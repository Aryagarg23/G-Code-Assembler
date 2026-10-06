// One closed solid from the beads, for simulation.
//
// Every bead is a stadium in cross-section (the shape a squashed round
// extrusion takes): as tall as its layer, as wide as engine/parse.mjs worked
// out from the filament, with half-circle sides. In plan it runs the length of
// its move with round ends, so beads join at corners instead of leaving wedge
// gaps. The bead spans from the layer below up to the nozzle height Z.
//
// The solid is the union of all beads. It is sampled as a field (positive
// inside: the bead's half-width at that height minus the distance to its
// centreline) on a grid: `cell` mm across, and `subSlices` samples per layer
// up the height. The surface is the zero level, extracted by marching
// tetrahedra over a Kuhn split of each cube, which is consistent between
// neighbouring cubes, so the mesh is closed (watertight) and every edge joins
// exactly two triangles. Each triangle faces outward.
//
// Accuracy: in plan the field is a true distance, so walls sit within a small
// fraction of a cell of the bead edge. Up the height the surface is placed
// between samples, so tops and bottoms are within half a sub-slice
// (h / (2 * subSlices)) of the true height.

// Cube corners as (di, dj, dk) offsets, and the 6 Kuhn tetrahedra of a cube:
// each walks 000 -> one axis -> two axes -> 111. Every tet edge then runs
// between corners whose offsets only increase.
const C = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]];
const TETS = [[0, 1, 3, 7], [0, 1, 5, 7], [0, 2, 3, 7], [0, 2, 6, 7], [0, 4, 5, 7], [0, 4, 6, 7]];

// CASES[tet][pattern]: the triangles for each inside/outside pattern of a
// tet's 4 corners, as corner pairs (low corner first) whose edges carry the
// vertices. Facing is worked out once here on edge midpoints, where no
// triangle is degenerate, so at run time it never depends on a sliver's
// shape: neighbouring triangles always agree and the mesh stays consistent.
const CASES = TETS.map(tet => {
  const out = [];
  for (let pattern = 0; pattern < 16; pattern++) {
    const inside = [], outside = [];
    tet.forEach((c, k) => ((pattern >> k) & 1 ? inside : outside).push(c));
    const tris = [];
    if (inside.length === 1 || inside.length === 3) {
      const [solo, rest] = inside.length === 1 ? [inside[0], outside] : [outside[0], inside];
      tris.push([[solo, rest[0]], [solo, rest[1]], [solo, rest[2]]]);
    } else if (inside.length === 2) {
      const [a, b] = inside, [c, d] = outside;
      tris.push([[a, c], [a, d], [b, d]], [[a, c], [b, d], [b, c]]);
    }
    out.push(tris.map(tri => orient(tri.map(([a, b]) => low(a, b)), inside, outside)));
  }
  return out;
});
function low(a, b) {
  return C[a][0] <= C[b][0] && C[a][1] <= C[b][1] && C[a][2] <= C[b][2] ? [a, b] : [b, a];
}
function orient(tri, inside, outside) {
  const mid = ([a, b]) => C[a].map((v, k) => (v + C[b][k]) / 2);
  const centroid = list => [0, 1, 2].map(k => list.reduce((s, c) => s + C[c][k], 0) / list.length);
  const [p, q, r] = tri.map(mid);
  const u = q.map((v, k) => v - p[k]), w = r.map((v, k) => v - p[k]);
  const n = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
  const ci = centroid(inside), co = centroid(outside);
  const dot = n[0] * (co[0] - ci[0]) + n[1] * (co[1] - ci[1]) + n[2] * (co[2] - ci[2]);
  const flat = dot >= 0 ? tri : [tri[0], tri[2], tri[1]];
  return flat.flat();
}

export const RESOLUTIONS = {
  coarse: { cell: 0.3, subSlices: 1 },
  medium: { cell: 0.2, subSlices: 2 },
  fine: { cell: 0.1, subSlices: 4 },
};


/**
 * @param {{ layers: { z: number, h: number, segs: Float64Array }[] }} parsed
 * @param {{ cell?: number, subSlices?: number, fill?: boolean, maxCellsPerSlice?: number, onProgress?: (f: number) => void }} options
 * @returns {{ positions: Float32Array, indices: Uint32Array, cell: number, subSlices: number, grid: number[] }}
 */
export function buildSolid(parsed, options = {}) {
  const layers = parsed.layers.filter(l => l.segs.length);
  let cell = options.cell ?? RESOLUTIONS.medium.cell;
  const subSlices = Math.max(1, Math.round(options.subSlices ?? RESOLUTIONS.medium.subSlices));
  const maxCells = options.maxCellsPerSlice ?? 1_500_000;
  const fill = options.fill ?? true;
  if (!layers.length) return { positions: new Float32Array(0), indices: new Uint32Array(0), cell, subSlices, grid: [0, 0, 0] };

  // Plan extent of every bead, widened by its half-width.
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const { segs } of layers) {
    for (let i = 0; i < segs.length; i += 5) {
      const r = segs[i + 4] / 2;
      minX = Math.min(minX, segs[i] - r, segs[i + 2] - r);
      maxX = Math.max(maxX, segs[i] + r, segs[i + 2] + r);
      minY = Math.min(minY, segs[i + 1] - r, segs[i + 3] - r);
      maxY = Math.max(maxY, segs[i + 1] + r, segs[i + 3] + r);
    }
  }
  // Keep a slice within budget by coarsening the cell if the part is large.
  while (((maxX - minX) / cell + 4) * ((maxY - minY) / cell + 4) > maxCells) cell *= 1.25;
  const ox = minX - 2 * cell, oy = minY - 2 * cell;
  const nx = Math.ceil((maxX - minX) / cell) + 5, ny = Math.ceil((maxY - minY) / cell) + 5;
  const nxy = nx * ny;
  // Empty space reads as "one cell outside", like a distance would. A huge
  // value would pin the surface onto the last sample and make slivers.
  const OUTSIDE = -cell;

  // Sample heights: subSlices per layer, at the middle of equal bands, plus an
  // empty slice below the first and above the last.
  const samples = [];
  for (const layer of layers) {
    const bottom = layer.z - layer.h;
    for (let s = 0; s < subSlices; s++) {
      const t = (s + 0.5) / subSlices * layer.h;
      samples.push({ z: bottom + t, layer, t });
    }
  }
  samples.sort((a, b) => a.z - b.z);
  const first = samples[0], last = samples[samples.length - 1];
  const dz0 = first.layer.h / subSlices, dz1 = last.layer.h / subSlices;
  const slices = [{ z: first.z - dz0, layer: null }, ...samples, { z: last.z + dz1, layer: null }];

  const positions = [];
  const indices = [];
  // Vertex ids on grid edges, for the slice the edge starts in. 7 edge
  // directions per grid point: +x, +y, +x+y (in the slice) and +z, +x+z,
  // +y+z, +x+y+z (up to the next slice).
  let edgesLo = new Int32Array(nxy * 7).fill(-1);
  let edgesHi = new Int32Array(nxy * 7).fill(-1);
  let lo = null;
  const total = slices.length - 1;

  function fieldFor(slice) {
    const out = new Float32Array(nxy).fill(OUTSIDE);
    if (!slice.layer) return out;
    const { h, segs } = slice.layer;
    const t = slice.t;
    // Stadium half-width at height t within a layer of height h, for width w:
    // (w - h)/2 + sqrt((h/2)^2 - (t - h/2)^2).
    const bulge = Math.sqrt(Math.max(0, (h / 2) ** 2 - (t - h / 2) ** 2));
    for (let i = 0; i < segs.length; i += 5) {
      const x0 = segs[i], y0 = segs[i + 1], x1 = segs[i + 2], y1 = segs[i + 3], w = segs[i + 4];
      const r = Math.max(0, (w - h) / 2) + bulge;
      if (r <= 0) continue;
      const reach = r + cell; // the field is needed a cell past the edge
      const i0 = Math.max(0, Math.floor((Math.min(x0, x1) - reach - ox) / cell));
      const i1 = Math.min(nx - 1, Math.ceil((Math.max(x0, x1) + reach - ox) / cell));
      const j0 = Math.max(0, Math.floor((Math.min(y0, y1) - reach - oy) / cell));
      const j1 = Math.min(ny - 1, Math.ceil((Math.max(y0, y1) + reach - oy) / cell));
      const dx = x1 - x0, dy = y1 - y0, ll = dx * dx + dy * dy;
      for (let j = j0; j <= j1; j++) {
        const py = oy + j * cell;
        for (let ii = i0; ii <= i1; ii++) {
          const px = ox + ii * cell;
          let u = ll > 0 ? ((px - x0) * dx + (py - y0) * dy) / ll : 0;
          u = u < 0 ? 0 : u > 1 ? 1 : u;
          const ex = px - (x0 + u * dx), ey = py - (y0 + u * dy);
          const v = r - Math.sqrt(ex * ex + ey * ey);
          const k = j * nx + ii;
          if (v > out[k]) out[k] = v;
        }
      }
    }
    // Never exactly zero: a vertex on a grid point would make slivers.
    for (let k = 0; k < nxy; k++) if (out[k] === 0) out[k] = -1e-7;
    return out;
  }

  // Fill: empty space that cannot reach the outside without crossing a bead
  // (infill pockets, the inside of walls) counts as solid, so what is left is
  // the part's outer boundary, which is what a flow or stress model meshes.
  // "Reach" is in 3D: a cabin or a chimney that opens to the air higher up
  // stays empty even where walls surround it in one layer.
  //
  // Each slice's empty cells are split into 4-connected regions; regions in
  // neighbouring slices that share a cell are joined (union-find), and so is
  // any region touching the edge of the grid. The first pass only records
  // that; the meshing pass labels each slice the same way again and fills the
  // regions that never joined the outside.
  const labelsA = new Int32Array(nxy), labelsB = new Int32Array(nxy), queue = new Int32Array(nxy);
  const parent = [0]; // region 0 is "outside"
  const find = x => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
  const union = (a, b) => { a = find(a); b = find(b); if (a !== b) parent[Math.max(a, b)] = Math.min(a, b); };
  // Add the empty, unlabelled 4-neighbours of cell k to region `id`.
  function spread(out, labels, id, k, i, j, tail) {
    if (i > 0 && !labels[k - 1] && out[k - 1] <= 0) { labels[k - 1] = id; queue[tail++] = k - 1; }
    if (i < nx - 1 && !labels[k + 1] && out[k + 1] <= 0) { labels[k + 1] = id; queue[tail++] = k + 1; }
    if (j > 0 && !labels[k - nx] && out[k - nx] <= 0) { labels[k - nx] = id; queue[tail++] = k - nx; }
    if (j < ny - 1 && !labels[k + nx] && out[k + nx] <= 0) { labels[k + nx] = id; queue[tail++] = k + nx; }
    return tail;
  }
  function label(out, labels) {
    labels.fill(0);
    for (let start = 0; start < nxy; start++) {
      if (labels[start] || out[start] > 0) continue;
      const id = parent.length;
      parent.push(id);
      labels[start] = id;
      let head = 0, tail = 0;
      queue[tail++] = start;
      while (head < tail) {
        const k = queue[head++], i = k % nx, j = (k - i) / nx;
        if (i === 0 || j === 0 || i === nx - 1 || j === ny - 1) union(id, 0);
        tail = spread(out, labels, id, k, i, j, tail);
      }
    }
  }
  if (fill) {
    let prev = labelsA, cur = labelsB;
    for (let s = 0; s < slices.length; s++) {
      const out = fieldFor(slices[s]);
      label(out, cur);
      // The empty slices above and below are open air.
      if (s === 0 || s === slices.length - 1) for (let k = 0; k < nxy; k++) if (cur[k]) union(cur[k], 0);
      if (s > 0) for (let k = 0; k < nxy; k++) if (cur[k] && prev[k]) union(cur[k], prev[k]);
      [prev, cur] = [cur, prev];
      options.onProgress?.(0.4 * (s + 1) / slices.length);
    }
  }
  // The meshing pass labels each slice again in the same order, so region
  // ids come out the same as in the first pass.
  let nextRegion = 1;
  function filled(out) {
    if (!fill) return out;
    const labels = labelsA;
    labels.fill(0);
    for (let start = 0; start < nxy; start++) {
      if (labels[start] || out[start] > 0) continue;
      const id = nextRegion++;
      labels[start] = id;
      let head = 0, tail = 0;
      queue[tail++] = start;
      const enclosed = find(id) !== 0;
      while (head < tail) {
        const k = queue[head++], i = k % nx, j = (k - i) / nx;
        if (enclosed) out[k] = cell;
        tail = spread(out, labels, id, k, i, j, tail);
      }
    }
    return out;
  }

  const val = new Float64Array(8), px = new Float64Array(8), py = new Float64Array(8), pz = new Float64Array(8);
  const gi = new Int32Array(8), gj = new Int32Array(8), gk = new Int32Array(8);

  lo = filled(fieldFor(slices[0]));
  for (let s = 0; s < total; s++) {
    const hi = filled(fieldFor(slices[s + 1]));
    const z0 = slices[s].z, z1 = slices[s + 1].z;
    for (let j = 0; j < ny - 1; j++) {
      for (let i = 0; i < nx - 1; i++) {
        const b = j * nx + i;
        const a0 = lo[b], a1 = lo[b + 1], a2 = lo[b + nx], a3 = lo[b + nx + 1];
        const a4 = hi[b], a5 = hi[b + 1], a6 = hi[b + nx], a7 = hi[b + nx + 1];
        const ins = (a0 > 0) + (a1 > 0) + (a2 > 0) + (a3 > 0) + (a4 > 0) + (a5 > 0) + (a6 > 0) + (a7 > 0);
        if (ins === 0 || ins === 8) continue;
        val[0] = a0; val[1] = a1; val[2] = a2; val[3] = a3; val[4] = a4; val[5] = a5; val[6] = a6; val[7] = a7;
        for (let c = 0; c < 8; c++) {
          gi[c] = i + C[c][0]; gj[c] = j + C[c][1]; gk[c] = C[c][2];
          px[c] = ox + gi[c] * cell; py[c] = oy + gj[c] * cell; pz[c] = gk[c] ? z1 : z0;
        }
        let mask = 0;
        for (let c = 0; c < 8; c++) if (val[c] > 0) mask |= 1 << c;
        for (let t = 0; t < 6; t++) {
          const tet = TETS[t];
          const pattern = ((mask >> tet[0]) & 1) | (((mask >> tet[1]) & 1) << 1) | (((mask >> tet[2]) & 1) << 2) | (((mask >> tet[3]) & 1) << 3);
          for (const [a, b, c, d, e, f] of CASES[t][pattern]) {
            const v0 = vertex(a, b), v1 = vertex(c, d), v2 = vertex(e, f);
            if (v0 !== v1 && v1 !== v2 && v0 !== v2) indices.push(v0, v1, v2);
          }
        }
      }
    }
    // Slide up: the top slice becomes the bottom one.
    lo = hi;
    const spare = edgesLo;
    edgesLo = edgesHi;
    edgesHi = spare.fill(-1);
    // In-slice edges of the new bottom slice were made as "Hi" in-slice edges;
    // vertical edges start fresh.
    for (let k = 0; k < nxy; k++) { const o = k * 7; edgesLo[o + 3] = edgesLo[o + 4] = edgesLo[o + 5] = edgesLo[o + 6] = -1; }
    options.onProgress?.((fill ? 0.4 : 0) + (fill ? 0.6 : 1) * (s + 1) / total);
  }

  // Vertex on the grid edge between corners a and b (a "below/left of" b in
  // every axis, as Kuhn edges always are), shared by every tet that uses it.
  function vertex(a, b) {
    const dir = (gi[b] - gi[a]) + 2 * (gj[b] - gj[a]) + 4 * (gk[b] - gk[a]); // 1..7
    const type = [-1, 0, 1, 2, 3, 4, 5, 6][dir];
    const slot = (gj[a] * nx + gi[a]) * 7 + type;
    const table = gk[a] ? edgesHi : edgesLo;
    if (table[slot] !== -1) return table[slot];
    const t = val[a] / (val[a] - val[b]);
    const id = positions.length / 3;
    positions.push(px[a] + (px[b] - px[a]) * t, py[a] + (py[b] - py[a]) * t, pz[a] + (pz[b] - pz[a]) * t);
    table[slot] = id;
    return id;
  }

  return { positions: Float32Array.from(positions), indices: Uint32Array.from(indices), cell, subSlices, grid: [nx, ny, slices.length] };
}

/** Indexed mesh to a flat triangle list (9 floats per triangle). */
export function toTriangles({ positions, indices }) {
  const out = new Float32Array(indices.length * 3);
  for (let t = 0; t < indices.length; t++) {
    const v = indices[t] * 3;
    out[t * 3] = positions[v]; out[t * 3 + 1] = positions[v + 1]; out[t * 3 + 2] = positions[v + 2];
  }
  return out;
}

/** Edge check: closed means every edge is used by exactly two triangles, once
 *  in each direction (which also makes the facing consistent). */
export function checkClosed({ indices }) {
  // Per undirected edge: uses a->b (low id first) and b->a, packed as fwd + 1000 * bwd.
  const uses = new Map();
  for (let t = 0; t < indices.length; t += 3) {
    for (let k = 0; k < 3; k++) {
      const a = indices[t + k], b = indices[t + (k + 1) % 3];
      const key = a < b ? a * 4294967296 + b : b * 4294967296 + a;
      uses.set(key, (uses.get(key) ?? 0) + (a < b ? 1 : 1000));
    }
  }
  let open = 0, bad = 0;
  for (const v of uses.values()) {
    if (v === 1001) continue;
    if (v === 1 || v === 1000) open++;
    else bad++;
  }
  return { edges: uses.size, openEdges: open, badEdges: bad, closed: open === 0 && bad === 0 };
}
