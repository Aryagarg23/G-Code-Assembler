// Which kind of bead each triangle of the mesh came from, so the viewer can
// colour walls, solid layers and infill apart the way a slicer preview does.
//
// A triangle takes the kind of the nearest bead centreline in the layer its
// centroid sits in. Beads are bucketed on a plan grid per layer, so each
// lookup checks a few cells instead of every bead.

const BUCKET = 2; // mm

export function triangleKinds(tris, layers) {
  const n = tris.length / 9;
  const out = new Uint8Array(n).fill(4);
  if (!layers.length) return out;
  const tops = layers.map(l => l.z);
  const index = layers.map(bucketLayer);
  for (let t = 0; t < n; t++) {
    const o = t * 9;
    const cx = (tris[o] + tris[o + 3] + tris[o + 6]) / 3;
    const cy = (tris[o + 1] + tris[o + 4] + tris[o + 7]) / 3;
    const cz = (tris[o + 2] + tris[o + 5] + tris[o + 8]) / 3;
    out[t] = nearestKind(index[layerAt(tops, cz)], cx, cy);
  }
  return out;
}

// First layer whose top is at or above z (the layer a point at height z is in).
export function layerAt(tops, z) {
  let lo = 0, hi = tops.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (tops[mid] < z - 1e-6) lo = mid + 1; else hi = mid;
  }
  return lo;
}

function bucketLayer({ segs, kinds }) {
  const cells = new Map();
  const n = segs.length / 5;
  for (let b = 0; b < n; b++) {
    const i = b * 5;
    const r = segs[i + 4] / 2;
    const x0 = Math.floor((Math.min(segs[i], segs[i + 2]) - r) / BUCKET), x1 = Math.floor((Math.max(segs[i], segs[i + 2]) + r) / BUCKET);
    const y0 = Math.floor((Math.min(segs[i + 1], segs[i + 3]) - r) / BUCKET), y1 = Math.floor((Math.max(segs[i + 1], segs[i + 3]) + r) / BUCKET);
    for (let gx = x0; gx <= x1; gx++) for (let gy = y0; gy <= y1; gy++) {
      const key = gx * 100003 + gy;
      let list = cells.get(key);
      if (!list) cells.set(key, (list = []));
      list.push(b);
    }
  }
  return { segs, kinds, cells };
}

function nearestKind({ segs, kinds, cells }, x, y) {
  const gx = Math.floor(x / BUCKET), gy = Math.floor(y / BUCKET);
  let best = Infinity, kind = 4;
  for (let ring = 0; ring <= 2 && best === Infinity; ring++) {
    for (let dx = -ring; dx <= ring; dx++) for (let dy = -ring; dy <= ring; dy++) {
      if (Math.max(Math.abs(dx), Math.abs(dy)) !== ring) continue;
      const list = cells.get((gx + dx) * 100003 + (gy + dy));
      if (!list) continue;
      for (const b of list) {
        const i = b * 5;
        const d = segDistance(x, y, segs[i], segs[i + 1], segs[i + 2], segs[i + 3]) - segs[i + 4] / 2;
        if (d < best) { best = d; kind = kinds[b]; }
      }
    }
  }
  return kind;
}

function segDistance(px, py, x0, y0, x1, y1) {
  const dx = x1 - x0, dy = y1 - y0;
  const len2 = dx * dx + dy * dy;
  const t = len2 > 0 ? Math.max(0, Math.min(1, ((px - x0) * dx + (py - y0) * dy) / len2)) : 0;
  return Math.hypot(px - x0 - t * dx, py - y0 - t * dy);
}

// Every layer's beads packed for the viewer's layer view: per bead
// x0 y0 x1 y1 width kind (mm), and where each layer starts.
export function packToolpaths(layers) {
  const total = layers.reduce((s, l) => s + l.segs.length / 5, 0);
  const beads = new Float32Array(total * 6);
  const starts = new Uint32Array(layers.length + 1);
  let b = 0;
  layers.forEach((l, li) => {
    starts[li] = b;
    for (let k = 0; k < l.segs.length / 5; k++, b++) {
      for (let c = 0; c < 5; c++) beads[b * 6 + c] = l.segs[k * 5 + c];
      beads[b * 6 + 5] = l.kinds[k];
    }
  });
  starts[layers.length] = b;
  return { beads, starts };
}
