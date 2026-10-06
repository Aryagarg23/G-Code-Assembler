// A robust FDM G-code reader: what was extruded, where, and how wide.
//
// Fixes over the hackathon reader (engine/legacy.mjs):
//  - G92 resets positions (E especially); relative/absolute XYZ (G90/G91) and
//    E (M82/M83, or G91/G90 when no M82/M83 was given) are tracked separately;
//    G20 inches are converted.
//  - G2/G3 arcs (I/J or R) are followed as arcs, split into short chords
//    (at most `arcTolerance` mm from the true arc), with E and Z shared out.
//  - Each extrusion keeps the Z it was printed at (the Z after the move).
//  - Bead width comes from the filament pushed out: the bead cross-section is
//    E * filament area / length, read as a stadium (a rectangle with
//    half-circle sides) of the layer height. Layer height is the gap to the
//    layer below.
//  - Only the part: slicer comments mark the start (first layer change), the
//    end and each feature type. Purge lines, wipe/prime towers, skirts, brims,
//    rafts and supports are left out (and counted). Files without such
//    comments keep every extrusion.
//  - Slicers: Bambu Studio / OrcaSlicer ("; CHANGE_LAYER", "; FEATURE:"),
//    PrusaSlicer / SuperSlicer (";LAYER_CHANGE", ";TYPE:"), Cura and
//    ideaMaker (";LAYER:n", ";TYPE:"), Simplify3D ("; layer n, Z = ...",
//    "; feature ..."). Firmware flavour does not matter beyond the standard
//    commands; M200 D (volumetric E, where E counts mm^3) is honoured.

// Feature names that are not the part, in any slicer's wording ("Skirt/Brim",
// "SUPPORT-INTERFACE", "Support material", "Prime tower", "prime pillar",
// "Wipe tower", "RAFT", "Custom" start/end code, "Ironing").
const EXCLUDED_FEATURE = /skirt|brim|support|prime|wipe|purge|raft|custom|ironing/i;
const LAYER_MARKER = /^(; ?CHANGE_LAYER|;LAYER_CHANGE|;LAYER:-?\d+|; layer \d+, Z = )/;
const END_MARKER = /^(; ?MACHINE_END_GCODE_START|;\s*filament end gcode|;End of Gcode|; ?EXECUTABLE_BLOCK_END|;TYPE:END|; layer end)/i;
const FEATURE = /^;\s*(?:FEATURE|TYPE)\s*:\s*(.+)$|^;\s*feature\s+(.+)$/i;

/**
 * @param {string} text  the G-code file
 * @param {{ arcTolerance?: number, filamentDiameter?: number, keepAll?: boolean }} options
 */
export function parseGcode(text, options = {}) {
  const arcTolerance = options.arcTolerance ?? 0.01;
  const settings = readSettings(text);
  const filamentDiameter = options.filamentDiameter ?? settings.filamentDiameter ?? 1.75;
  const filamentArea = Math.PI * (filamentDiameter / 2) ** 2;
  const hasMarkers = new RegExp(LAYER_MARKER.source, 'm').test(text);
  const keepAll = options.keepAll || !hasMarkers;

  const pos = { x: NaN, y: NaN, z: 0, e: 0 };
  let absXYZ = true;
  let absE = true;
  let eModeSet = false; // M82/M83 seen: G90/G91 no longer change E mode
  let unit = 1;
  let volumetric = false; // M200 D>0: E counts filament volume (mm^3), not length
  let started = keepAll;
  let ended = false;
  let feature = '';
  const moves = []; // [x0, y0, x1, y1, z, e, featureIndex]
  const features = [];
  const featureIndex = name => {
    let i = features.indexOf(name);
    if (i === -1) { features.push(name); i = features.length - 1; }
    return i;
  };
  const counts = { lines: 0, moves: 0, arcs: 0, retractions: 0, excluded: {}, outsidePart: 0, unknownCommands: 0 };

  const emit = (x0, y0, x1, y1, z, de) => {
    if (!(de > 1e-7)) return;
    const len = Math.hypot(x1 - x0, y1 - y0);
    if (!(len > 1e-5) || !Number.isFinite(x0) || !Number.isFinite(y0)) return;
    if (!started || ended) { counts.outsidePart += 1; return; }
    if (feature && EXCLUDED_FEATURE.test(feature)) {
      counts.excluded[feature] = (counts.excluded[feature] ?? 0) + 1;
      return;
    }
    moves.push(x0, y0, x1, y1, z, de, featureIndex(feature || 'Extrusion'));
  };

  for (const rawLine of text.split('\n')) {
    counts.lines += 1;
    const line = rawLine.trim();
    if (!line) continue;
    if (line[0] === ';') {
      if (!started && LAYER_MARKER.test(line)) started = true;
      if (started && END_MARKER.test(line)) ended = true;
      const m = FEATURE.exec(line);
      if (m) feature = (m[1] ?? m[2]).trim();
      continue;
    }
    // Strip comments, line numbers and checksums.
    let code = line;
    const semi = code.indexOf(';');
    if (semi !== -1) code = code.slice(0, semi);
    code = code.replace(/\*\d+\s*$/, '').replace(/^N\d+\s+/i, '').trim().toUpperCase();
    if (!code) continue;
    const cmd = /^([GMT])(\d+(?:\.\d+)?)/.exec(code);
    if (!cmd) { counts.unknownCommands += 1; continue; }
    const word = cmd[1] + Number(cmd[2]);
    const p = {};
    for (const m of code.slice(cmd[0].length).matchAll(/([A-Z])\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:E[-+]?\d+)?)/g)) p[m[1]] = Number(m[2]);

    switch (word) {
      case 'G90': absXYZ = true; if (!eModeSet) absE = true; continue;
      case 'G91': absXYZ = false; if (!eModeSet) absE = false; continue;
      case 'M82': absE = true; eModeSet = true; continue;
      case 'M83': absE = false; eModeSet = true; continue;
      case 'M200': volumetric = ('D' in p ? p.D > 0 : !('S' in p) || p.S > 0) && (p.D ?? 1) > 0; continue;
      case 'G20': unit = 25.4; continue;
      case 'G21': unit = 1; continue;
      case 'G92':
        if (!('X' in p || 'Y' in p || 'Z' in p || 'E' in p)) { pos.x = pos.y = pos.z = pos.e = 0; continue; }
        if ('X' in p) pos.x = p.X * unit;
        if ('Y' in p) pos.y = p.Y * unit;
        if ('Z' in p) pos.z = p.Z * unit;
        if ('E' in p) pos.e = p.E * unit;
        continue;
      case 'G28':
        // Homing: the named axes (or all) go to 0.
        if (!('X' in p || 'Y' in p || 'Z' in p)) { pos.x = pos.y = pos.z = 0; continue; }
        if ('X' in p) pos.x = 0;
        if ('Y' in p) pos.y = 0;
        if ('Z' in p) pos.z = 0;
        continue;
      case 'G0': case 'G1': case 'G2': case 'G3': break;
      default: continue;
    }

    counts.moves += 1;
    const x1 = moveTo(p, 'X', pos.x, absXYZ, unit), y1 = moveTo(p, 'Y', pos.y, absXYZ, unit), z1 = moveTo(p, 'Z', pos.z, absXYZ, unit);
    let de = 0;
    if ('E' in p) {
      const e = p.E * unit;
      de = absE ? e - pos.e : e;
      pos.e = absE ? e : pos.e + e;
    }
    if (de < 0) counts.retractions += 1;
    if (volumetric) de /= filamentArea; // store filament length either way

    if (word === 'G2' || word === 'G3') {
      counts.arcs += 1;
      const pts = arcPoints(pos.x, pos.y, x1, y1, p, unit, word === 'G2', arcTolerance);
      if (pts) {
        // Share E (and any helical Z) out by chord length.
        let total = 0;
        for (let i = 2; i < pts.length; i += 2) total += Math.hypot(pts[i] - pts[i - 2], pts[i + 1] - pts[i - 1]);
        let done = 0;
        for (let i = 2; i < pts.length; i += 2) {
          const l = Math.hypot(pts[i] - pts[i - 2], pts[i + 1] - pts[i - 1]);
          const zAt = pos.z + (z1 - pos.z) * ((done + l) / (total || 1));
          emit(pts[i - 2], pts[i - 1], pts[i], pts[i + 1], zAt, total ? de * l / total : 0);
          done += l;
        }
        pos.x = x1; pos.y = y1; pos.z = z1;
        continue;
      }
    }
    emit(pos.x, pos.y, x1, y1, z1, de);
    pos.x = x1; pos.y = y1; pos.z = z1;
  }

  return buildLayers(moves, features, { filamentArea, filamentDiameter, settings, counts, keepAll });
}

// Where an axis ends up after a move: the new value, an offset, or unchanged.
function moveTo(p, axis, current, absolute, unit) {
  if (!(axis in p)) return current;
  return absolute ? p[axis] * unit : current + p[axis] * unit;
}

// Points along a G2 (clockwise) / G3 arc from (x0,y0) to (x1,y1), as a flat
// [x, y, ...] list including both ends. null when the arc is malformed.
function arcPoints(x0, y0, x1, y1, p, unit, clockwise, tolerance) {
  if (!Number.isFinite(x0) || !Number.isFinite(y0)) return null;
  let cx, cy;
  if ('I' in p || 'J' in p) {
    cx = x0 + (p.I ?? 0) * unit;
    cy = y0 + (p.J ?? 0) * unit;
  } else if ('R' in p) {
    const r = p.R * unit;
    const dx = x1 - x0, dy = y1 - y0, d = Math.hypot(dx, dy);
    if (!(d > 0) || Math.abs(r) < d / 2) return null;
    const h = Math.sqrt(r * r - (d / 2) ** 2);
    // R > 0: the shorter arc; R < 0: the longer one.
    const sign = (clockwise ? -1 : 1) * (r > 0 ? 1 : -1);
    cx = (x0 + x1) / 2 - sign * h * dy / d;
    cy = (y0 + y1) / 2 + sign * h * dx / d;
  } else return null;
  const r = Math.hypot(x0 - cx, y0 - cy);
  if (!(r > 1e-6)) return null;
  let a0 = Math.atan2(y0 - cy, x0 - cx);
  let a1 = Math.atan2(y1 - cy, x1 - cx);
  let sweep = clockwise ? a0 - a1 : a1 - a0;
  if (sweep <= 1e-9) sweep += 2 * Math.PI; // same start and end: a full circle
  // Chord error r(1 - cos(step/2)) <= tolerance.
  const step = 2 * Math.acos(Math.max(-1, 1 - tolerance / r));
  const n = Math.max(1, Math.ceil(sweep / (step || sweep)));
  const out = [x0, y0];
  for (let i = 1; i < n; i++) {
    const a = a0 + (clockwise ? -1 : 1) * sweep * i / n;
    out.push(cx + r * Math.cos(a), cy + r * Math.sin(a));
  }
  out.push(x1, y1);
  return out;
}

// Slicer settings from the header/config comments, when present.
function readSettings(text) {
  const head = text.length > 400000 ? text.slice(0, 200000) + '\n' + text.slice(-200000) : text;
  const num = re => { const m = re.exec(head); return m ? Number(m[1]) : undefined; };
  return {
    filamentDiameter: num(/^;\s*filament_diameter\s*[:=]\s*([\d.]+)/im) ?? num(/^;\s*Filament diameter\s*[:=]\s*([\d.]+)/im),
    nozzleDiameter: num(/^;\s*nozzle_diameter\s*[:=]\s*([\d.]+)/im) ?? num(/^;\s*machine_nozzle_size\s*[:=]\s*([\d.]+)/im),
    layerHeight: num(/^;\s*layer_height\s*[:=]\s*([\d.]+)/im) ?? num(/^;Layer height:\s*([\d.]+)/im),
    lineWidth: num(/^;\s*line_width\s*[:=]\s*([\d.]+)/im) ?? num(/^;\s*extrusion_width\s*[:=]\s*([\d.]+)/im),
  };
}

// Group moves into layers by Z and give each bead its width.
function buildLayers(moves, features, { filamentArea, filamentDiameter, settings, counts, keepAll }) {
  const byZ = new Map();
  for (let i = 0; i < moves.length; i += 7) {
    const z = Math.round(moves[i + 4] * 1e4) / 1e4;
    if (!byZ.has(z)) byZ.set(z, []);
    byZ.get(z).push(i);
  }
  const zs = [...byZ.keys()].sort((a, b) => a - b);
  const nominalH = settings.layerHeight ?? 0.2;
  const nozzle = settings.nozzleDiameter ?? 0.4;
  const layers = [];
  let clamped = 0;
  let prevZ = null;
  const widths = [];
  let extruded = 0; // filament volume pushed out for the part, mm^3
  for (const z of zs) {
    // Layer height: the gap to the layer below. A first layer sits on the bed;
    // a gap far bigger than nominal (a part starting higher up) uses nominal.
    let h = prevZ === null ? z : z - prevZ;
    if (!(h > 0.01) || h > 2.5 * nominalH) h = Math.min(nominalH, z);
    prevZ = z;
    const list = byZ.get(z);
    const segs = new Float64Array(list.length * 5); // x0 y0 x1 y1 width
    let k = 0;
    for (const i of list) {
      const x0 = moves[i], y0 = moves[i + 1], x1 = moves[i + 2], y1 = moves[i + 3];
      const len = Math.hypot(x1 - x0, y1 - y0);
      extruded += moves[i + 5] * filamentArea;
      const area = moves[i + 5] * filamentArea / len; // bead cross-section, mm^2
      // Stadium of height h: area = (w - h) h + pi h^2 / 4.
      let w = area / h + h * (1 - Math.PI / 4);
      const lo = h, hi = 3 * nozzle;
      if (w < lo || w > hi) { clamped++; w = Math.min(hi, Math.max(lo, w)); }
      widths.push(w);
      segs[k++] = x0; segs[k++] = y0; segs[k++] = x1; segs[k++] = y1; segs[k++] = w;
    }
    layers.push({ z, h, segs });
  }
  const sorted = widths.slice().sort((a, b) => a - b);
  const hs = layers.map(l => l.h).sort((a, b) => a - b);
  return {
    layers,
    features,
    settings: { ...settings, filamentDiameter },
    stats: {
      ...counts,
      beads: widths.length,
      layers: layers.length,
      widthClamped: clamped,
      medianWidth: sorted.length ? sorted[sorted.length >> 1] : null,
      medianLayerHeight: hs.length ? hs[hs.length >> 1] : null,
      extrudedVolume: extruded,
      keptEverything: keepAll,
    },
  };
}
