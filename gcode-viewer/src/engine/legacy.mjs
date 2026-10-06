// The hackathon pipeline (Backend/flask_back.py, MakeUC 2024), ported to the
// browser line for line so the site can run it without the Flask server.
//
// It is kept as it was, quirks included, so the "Hackathon mesh" option shows
// what the original produced:
//  - every move that changes X or Y is a segment, and its Z is the Z *before*
//    the move; arcs (G2/G3) become one straight chord;
//  - G92 is ignored, so after an E reset in absolute mode the next moves read
//    as negative extrusion and are dropped;
//  - in relative E mode, a line without E re-adds the previous E value;
//  - each extruding segment becomes its own 0.4 x 0.2 box from z to z + 0.2,
//    so boxes overlap, leave wedge gaps at corners and are not one solid.
// engine/parse.mjs and engine/solid.mjs are the rewrite.

const NEG = -Infinity;
const LETTERS = { G: 0, X: 1, Y: 2, Z: 3, E: 4, F: 5, M: 6, T: 7 };

/** GcodeReader._read_fdm_regular. Returns { segs: Float64Array (7 per seg), nLayers }. */
export function legacyRead(text) {
  const lines = [];
  for (const raw of text.split('\n')) {
    let line = raw.trim();
    if (!line) continue;
    if (line[0] === 'G' || line[0] === 'M' || line[0] === 'T') {
      const idx = line.indexOf(';');
      if (idx !== -1) line = line.slice(0, idx);
      lines.push(line);
    }
  }
  const segs = [];
  let g = [NEG, NEG, NEG, NEG, NEG, NEG, NEG, NEG];
  let mxZ = NEG;
  let lastE = 0;
  let eRelative = false;
  let segId = 0;
  let lastWasExtruding = false;
  let lastX = null;
  let lastY = null;
  let nLayers = 0;
  for (const line of lines) {
    const old = g.slice();
    if (line.includes('G91')) { eRelative = true; continue; }
    if (line.includes('G90')) { eRelative = false; continue; }
    if (line.includes('M83')) { eRelative = true; continue; }
    if (line.includes('M82')) { eRelative = false; continue; }
    for (const token of line.split(/\s+/)) {
      if (!token) continue;
      const k = LETTERS[token[0]];
      if (k === undefined) continue;
      const rest = token.slice(1);
      if (token[0] === 'M' || token[0] === 'T') {
        // Python int(): whole decimal digits only (a sign is allowed).
        if (/^[+-]?\d+$/.test(rest.trim())) g[k] = parseInt(rest, 10);
      } else {
        const v = pyFloat(rest);
        if (v !== null) g[k] = v;
      }
    }
    let eMove;
    if (g[4] !== NEG) {
      if (eRelative) { eMove = g[4]; lastE += eMove; }
      else { eMove = g[4] - lastE; lastE = g[4]; }
    } else eMove = 0;
    const extruding = eMove > 0;
    const cx = g[1] !== NEG ? g[1] : lastX;
    const cy = g[2] !== NEG ? g[2] : lastY;
    const discontinuous = lastX !== null && lastY !== null
      ? Math.abs(cx - lastX) > 0.01 || Math.abs(cy - lastY) > 0.01 : false;
    if ((extruding !== lastWasExtruding && extruding) || (extruding && discontinuous)) segId += 1;
    lastWasExtruding = extruding;
    lastX = cx;
    lastY = cy;
    if ([0, 1, 2, 3].includes(g[0]) && (g[1] !== old[1] || g[2] !== old[2])) {
      if (g[3] > mxZ) { mxZ = g[3]; nLayers += 1; }
      segs.push(old[1], old[2], g[1], g[2], old[3], extruding ? 1 : 0, segId);
    }
  }
  return { segs: Float64Array.from(segs), nLayers, nSegs: segs.length / 7 };
}

// Python float(): accepts "1", ".5", "-2.", "1e3", "inf", "nan"; rejects "" and "1.2.3".
function pyFloat(s) {
  const t = s.trim();
  if (/^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/.test(t)) return Number(t);
  if (/^[+-]?(inf|infinity)$/i.test(t)) return t.startsWith('-') ? -Infinity : Infinity;
  if (/^[+-]?nan$/i.test(t)) return NaN;
  return null;
}

/** _create_extrusion_mesh + save_to_stl: triangles as Float32Array (9 per triangle). */
export function legacyMesh({ segs }, width = 0.4, height = 0.2) {
  // Group extruding segments by z, in first-seen order (Python dict order).
  const byZ = new Map();
  for (let i = 0; i < segs.length; i += 7) {
    if (!segs[i + 5]) continue;
    const z = segs[i + 4];
    if (!byZ.has(z)) byZ.set(z, []);
    byZ.get(z).push(i);
  }
  const out = [];
  for (const [z, list] of byZ) {
    for (const i of list) {
      const x0 = segs[i], y0 = segs[i + 1], x1 = segs[i + 2], y1 = segs[i + 3];
      const dx = x1 - x0, dy = y1 - y0;
      const length = Math.sqrt(dx * dx + dy * dy);
      if (length < 1e-6) continue;
      const nx = -dy / length * width / 2;
      const ny = dx / length * width / 2;
      const v = [
        [x0 - nx, y0 - ny, z], [x0 + nx, y0 + ny, z], [x1 + nx, y1 + ny, z], [x1 - nx, y1 - ny, z],
        [x0 - nx, y0 - ny, z + height], [x0 + nx, y0 + ny, z + height], [x1 + nx, y1 + ny, z + height], [x1 - nx, y1 - ny, z + height],
      ];
      for (const [a, b, c] of BOX_FACES) out.push(...v[a], ...v[b], ...v[c]);
    }
  }
  return Float32Array.from(out);
}

const BOX_FACES = [
  [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
  [0, 4, 1], [1, 4, 5], [1, 5, 2], [2, 5, 6], [2, 6, 3], [3, 6, 7], [3, 7, 0], [0, 7, 4],
];
