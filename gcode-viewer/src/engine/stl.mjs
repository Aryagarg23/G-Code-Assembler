// Binary STL out, and the model report the viewer's side panel shows.
// `analyze` is STLAutoAnalyzer.generate_report from Backend/flask_back.py,
// ported over the same float32 triangles numpy-stl would have read back.

const f = Math.fround;

/** Binary STL from triangles (Float32Array, 9 per triangle). Normals are the
 *  unit face normals; numpy-stl wrote unnormalised ones, which readers ignore. */
export function toBinaryStl(tris, name = 'gcode-assembler') {
  const n = tris.length / 9;
  const buf = new ArrayBuffer(84 + n * 50);
  const view = new DataView(buf);
  const header = `binary STL from G-Code Assembler: ${name}`.slice(0, 80);
  for (let i = 0; i < header.length; i++) view.setUint8(i, header.charCodeAt(i) & 0x7f);
  view.setUint32(80, n, true);
  let o = 84;
  for (let t = 0; t < n; t++) {
    const p = t * 9;
    const ax = tris[p + 3] - tris[p], ay = tris[p + 4] - tris[p + 1], az = tris[p + 5] - tris[p + 2];
    const bx = tris[p + 6] - tris[p], by = tris[p + 7] - tris[p + 1], bz = tris[p + 8] - tris[p + 2];
    let nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
    const len = Math.hypot(nx, ny, nz) || 1;
    nx /= len; ny /= len; nz /= len;
    view.setFloat32(o, nx, true); view.setFloat32(o + 4, ny, true); view.setFloat32(o + 8, nz, true);
    for (let k = 0; k < 9; k++) view.setFloat32(o + 12 + k * 4, tris[p + k], true);
    view.setUint16(o + 48, 0, true);
    o += 50;
  }
  return buf;
}

// numpy round (half to even) at `precision` decimals, in float32 like the input.
function npRound32(v, precision) {
  const s = 10 ** precision;
  const x = f(v * s);
  const r = Math.round(x);
  const even = Math.abs(x % 1) === 0.5 ? 2 * Math.round(x / 2) : r;
  return f(even / s);
}

function mostCommon(values, precision, threshold) {
  if (!values.length) return null;
  const counts = new Map();
  for (const v of values) {
    const r = npRound32(v, precision);
    counts.set(r, (counts.get(r) ?? 0) + 1);
  }
  let best = null, bestCount = -1;
  // np.unique sorts ascending and argmax takes the first maximum.
  for (const v of [...counts.keys()].sort((a, b) => a - b)) {
    if (!(v > threshold)) continue;
    if (counts.get(v) > bestCount) { best = v; bestCount = counts.get(v); }
  }
  return best;
}

const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
const std = a => { const m = mean(a); return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length); };

function bounds(tris) {
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < tris.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      const v = tris[i + k];
      if (v < min[k]) min[k] = v;
      if (v > max[k]) max[k] = v;
    }
  }
  return { min, max };
}

function triArea(tris, p) {
  const ax = tris[p + 3] - tris[p], ay = tris[p + 4] - tris[p + 1], az = tris[p + 5] - tris[p + 2];
  const bx = tris[p + 6] - tris[p], by = tris[p + 7] - tris[p + 1], bz = tris[p + 8] - tris[p + 2];
  return 0.5 * Math.hypot(ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx);
}

/** Eberly's polyhedral mass properties (numpy-stl get_mass_properties): volume
 *  and centre of gravity. Only meaningful for a closed mesh. */
export function massProperties(tris) {
  let i0 = 0, ix = 0, iy = 0, iz = 0;
  const sub = (w0, w1, w2) => {
    const t0 = w0 + w1, f1 = t0 + w2, t1 = w0 * w0, t2 = t1 + w1 * t0;
    return [f1, t2 + w2 * f1];
  };
  for (let p = 0; p < tris.length; p += 9) {
    const x0 = tris[p], y0 = tris[p + 1], z0 = tris[p + 2];
    const x1 = tris[p + 3], y1 = tris[p + 4], z1 = tris[p + 5];
    const x2 = tris[p + 6], y2 = tris[p + 7], z2 = tris[p + 8];
    const a1 = x1 - x0, b1 = y1 - y0, c1 = z1 - z0, a2 = x2 - x0, b2 = y2 - y0, c2 = z2 - z0;
    const d0 = b1 * c2 - b2 * c1, d1 = a2 * c1 - a1 * c2, d2 = a1 * b2 - a2 * b1;
    const [f1x, f2x] = sub(x0, x1, x2), [, f2y] = sub(y0, y1, y2), [, f2z] = sub(z0, z1, z2);
    i0 += d0 * f1x; ix += d0 * f2x; iy += d1 * f2y; iz += d2 * f2z;
  }
  const volume = i0 / 6;
  return { volume, cog: volume ? [ix / 24 / volume, iy / 24 / volume, iz / 24 / volume] : [0, 0, 0] };
}

/** STLAutoAnalyzer.generate_report. `params` overrides the detected print
 *  parameters (the solid mesh knows them from the G-code instead of guessing). */
export function analyze(tris, { name = 'model', sizeBytes = 0, params = null } = {}) {
  const n = tris.length / 9;
  const { min, max } = bounds(tris);
  const dims = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];

  let detected = params;
  if (!detected) {
    const zs = new Set();
    for (let i = 2; i < tris.length; i += 3) zs.add(tris[i]);
    const zSorted = [...zs].sort((a, b) => a - b);
    const zDiffs = [];
    for (let i = 1; i < zSorted.length; i++) zDiffs.push(f(zSorted[i] - zSorted[i - 1]));
    const widths = [];
    for (let p = 0; p < tris.length; p += 9) {
      for (let a = 0; a < 3; a++) {
        const b = (a + 1) % 3;
        const dx = f(tris[p + a * 3] - tris[p + b * 3]);
        const dy = f(tris[p + a * 3 + 1] - tris[p + b * 3 + 1]);
        const dz = f(tris[p + a * 3 + 2] - tris[p + b * 3 + 2]);
        if (Math.abs(dz) < 0.01) {
          const w = f(Math.sqrt(f(f(dx * dx) + f(dy * dy))));
          if (w > 0.01) widths.push(w);
        }
      }
    }
    let buildDirection = 'Z';
    if (dims[2] < dims[0] && dims[2] < dims[1]) buildDirection = dims[0] > dims[1] ? 'X' : 'Y';
    const valid = zDiffs.filter(d => d > 0.001);
    detected = {
      layer_height: mostCommon(zDiffs, 5, 0.001),
      extrusion_width: mostCommon(widths, 3, 0.01),
      build_direction: buildDirection,
      num_layers: zSorted.length,
      model_height: dims[2],
      layer_heights_stats: valid.length
        ? { min: Math.min(...valid), max: Math.max(...valid), mean: mean(valid), std: std(valid) }
        : { min: null, max: null, mean: null, std: null },
      dimensions: { x: dims[0], y: dims[1], z: dims[2] },
    };
  }

  let surface = 0;
  for (let p = 0; p < tris.length; p += 9) surface += triArea(tris, p);
  const { volume, cog } = massProperties(tris);

  // Layers: triangles grouped by round(mean z / layer height), as before.
  const layerAnalysis = [];
  const lh = detected.layer_height;
  if (lh && lh > 0) {
    const groups = new Map();
    for (let p = 0; p < tris.length; p += 9) {
      const zMean = f((f(f(tris[p + 2] + tris[p + 5]) + tris[p + 8])) / 3);
      const idx = pyRound(zMean / lh);
      if (!groups.has(idx)) groups.set(idx, []);
      groups.get(idx).push(p);
    }
    for (const idx of [...groups.keys()].sort((a, b) => a - b)) {
      const list = groups.get(idx);
      let area = 0;
      const lmin = [Infinity, Infinity, Infinity], lmax = [-Infinity, -Infinity, -Infinity];
      for (const p of list) {
        area += triArea(tris, p);
        for (let k = 0; k < 9; k++) {
          const v = tris[p + k], a = k % 3;
          if (v < lmin[a]) lmin[a] = v;
          if (v > lmax[a]) lmax[a] = v;
        }
      }
      layerAnalysis.push({ layer_index: idx, z_height: idx * lh, num_triangles: list.length, area_mm2: area, bounds_mm: { min: lmin, max: lmax } });
    }
  }

  // Quality: degenerate and small triangles, edge lengths.
  const modelSize = Math.hypot(...dims);
  const minArea = (modelSize * 0.0001) ** 2;
  let degenerate = 0, small = 0, eMin = Infinity, eMax = 0, eSum = 0, eSq = 0;
  for (let p = 0; p < tris.length; p += 9) {
    for (let a = 0; a < 3; a++) {
      const b = (a + 1) % 3;
      const len = Math.hypot(tris[p + b * 3] - tris[p + a * 3], tris[p + b * 3 + 1] - tris[p + a * 3 + 1], tris[p + b * 3 + 2] - tris[p + a * 3 + 2]);
      if (len < eMin) eMin = len;
      if (len > eMax) eMax = len;
      eSum += len;
      eSq += len * len;
    }
    const area = triArea(tris, p);
    if (area < 1e-10) degenerate++;
    else if (area < minArea) small++;
  }
  const nEdges = n * 3;
  const eMean = nEdges ? eSum / nEdges : 0;

  return {
    file_info: { path: name, size_bytes: sizeBytes },
    detected_parameters: detected,
    model_stats: {
      num_triangles: n,
      volume_mm3: volume,
      surface_area_mm2: surface,
      dimensions_mm: { x: dims[0], y: dims[1], z: dims[2] },
      center_of_gravity_mm: cog,
      bounding_box_mm: { min, max },
    },
    layer_analysis: layerAnalysis,
    quality_metrics: {
      degenerate_triangles: degenerate,
      small_triangles: small,
      edge_stats: { min_length: eMin, max_length: eMax, mean_length: eMean, std_length: Math.sqrt(Math.max(0, eSq / nEdges - eMean * eMean)), total_edges: nEdges },
    },
  };
}

// Python round(): half to even.
function pyRound(x) {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 ? 2 * Math.round(x / 2) : r;
}
