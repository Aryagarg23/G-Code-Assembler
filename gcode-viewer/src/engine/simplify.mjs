// Mesh simplification: quadric error edge collapse (Garland & Heckbert 1997).
//
// Each vertex carries a quadric, the sum of squared distances to the planes
// of the triangles around it. Collapsing an edge merges its two vertices into
// one, placed where the summed quadric is least; the cost is that sum, in
// mm^2: how far, squared, the new vertex sits from the surface it replaces.
// The cheapest edges go first, and simplification stops at the first edge
// whose cost passes the tolerance. Flat walls cost nothing, so they collapse
// to a few large triangles; curved and grooved surfaces keep their detail.
//
// A collapse is refused when it would:
//  - break the surface (the two vertices share a neighbour other than the two
//    across the edge: the "link condition", which keeps a closed manifold
//    closed and manifold);
//  - turn a triangle over or flatten it to nothing;
//  - leave a vertex joined to more than `maxValence` other vertices.

/**
 * @param {{ positions: Float32Array, indices: Uint32Array }} mesh  closed, indexed
 * @param {{ tolerance?: number, maxNormalTurn?: number, maxValence?: number }} options
 *   tolerance: largest allowed distance (mm) of a moved vertex from the
 *   surface it replaces; maxNormalTurn: cosine floor for a triangle's normal
 *   before vs after a collapse.
 */
export function simplify(mesh, options = {}) {
  const tol = options.tolerance ?? 0.02;
  const maxCost = tol * tol;
  const minCos = options.maxNormalTurn ?? 0.3;
  const maxValence = options.maxValence ?? 16;
  const nv = mesh.positions.length / 3;
  const nf = mesh.indices.length / 3;
  const P = Float64Array.from(mesh.positions);
  const F = Int32Array.from(mesh.indices);
  const faceAlive = new Uint8Array(nf).fill(1);
  const vertAlive = new Uint8Array(nv).fill(1);
  const stamp = new Int32Array(nv);
  const Q = new Float64Array(nv * 10);
  const mark = new Int32Array(nv);
  let epoch = 0;
  const nbA = new Int32Array(256), nbB = new Int32Array(256);
  const before = [0, 0, 0], after = [0, 0, 0];

  // Faces around each vertex.
  const vf = Array.from({ length: nv }, () => []);
  for (let f = 0; f < nf; f++) for (let k = 0; k < 3; k++) vf[F[f * 3 + k]].push(f);

  // Plane quadrics (unweighted: the cost is a sum of squared distances).
  const n = [0, 0, 0];
  for (let f = 0; f < nf; f++) {
    if (!normal(f, n)) continue;
    const a = F[f * 3] * 3;
    const d = -(n[0] * P[a] + n[1] * P[a + 1] + n[2] * P[a + 2]);
    const q = planeQuadric(n[0], n[1], n[2], d);
    for (let k = 0; k < 3; k++) addQ(F[f * 3 + k], q);
  }

  // Candidate edges in a binary heap keyed on cost (lazy: stale entries are
  // recognised by the vertex stamps and skipped).
  const heap = new Heap();
  const opt = [0, 0, 0];
  const pushEdge = (a, b) => {
    const cost = bestPosition(a, b, opt);
    if (cost <= maxCost) heap.push(cost, a, b, stamp[a], stamp[b], opt[0], opt[1], opt[2]);
  };
  for (let f = 0; f < nf; f++) {
    for (let k = 0; k < 3; k++) {
      const a = F[f * 3 + k], b = F[f * 3 + (k + 1) % 3];
      if (a < b) pushEdge(a, b); // each edge once (its other face sees b -> a)
    }
  }

  let faces = nf;
  const entry = { cost: 0, a: 0, b: 0, sa: 0, sb: 0, x: 0, y: 0, z: 0 };
  while (heap.pop(entry)) {
    const { a, b } = entry;
    if (!vertAlive[a] || !vertAlive[b] || stamp[a] !== entry.sa || stamp[b] !== entry.sb) continue;
    if (!collapse(a, b, entry.x, entry.y, entry.z)) continue;
    faces -= 2;
    const count = neighbours(a, nbA, ++epoch);
    for (let i = 0; i < count; i++) { const v = nbA[i]; pushEdge(Math.min(a, v), Math.max(a, v)); }
  }

  // Compact.
  const remap = new Int32Array(nv).fill(-1);
  const positions = [];
  const indices = [];
  for (let f = 0; f < nf; f++) {
    if (!faceAlive[f]) continue;
    for (let k = 0; k < 3; k++) {
      const v = F[f * 3 + k];
      if (remap[v] === -1) { remap[v] = positions.length / 3; positions.push(P[v * 3], P[v * 3 + 1], P[v * 3 + 2]); }
      indices.push(remap[v]);
    }
  }
  return { positions: Float32Array.from(positions), indices: Uint32Array.from(indices), facesBefore: nf, facesAfter: faces };

  // ---------------------------------------------------------------------------

  function normal(f, out) {
    const a = F[f * 3] * 3, b = F[f * 3 + 1] * 3, c = F[f * 3 + 2] * 3;
    return cross(P[a], P[a + 1], P[a + 2], P[b], P[b + 1], P[b + 2], P[c], P[c + 1], P[c + 2], out);
  }
  function cross(ax, ay, az, bx, by, bz, cx, cy, cz, out) {
    const ux = bx - ax, uy = by - ay, uz = bz - az, wx = cx - ax, wy = cy - ay, wz = cz - az;
    let x = uy * wz - uz * wy, y = uz * wx - ux * wz, z = ux * wy - uy * wx;
    const len = Math.hypot(x, y, z);
    if (!(len > 1e-14)) return 0;
    out[0] = x / len; out[1] = y / len; out[2] = z / len;
    return len;
  }
  function planeQuadric(a, b, c, d) {
    return [a * a, a * b, a * c, a * d, b * b, b * c, b * d, c * c, c * d, d * d];
  }
  function addQ(v, q) { const o = v * 10; for (let i = 0; i < 10; i++) Q[o + i] += q[i]; }
  function costAt(o1, o2, x, y, z) {
    const q = i => Q[o1 + i] + Q[o2 + i];
    return q(0) * x * x + 2 * q(1) * x * y + 2 * q(2) * x * z + 2 * q(3) * x
      + q(4) * y * y + 2 * q(5) * y * z + 2 * q(6) * y + q(7) * z * z + 2 * q(8) * z + q(9);
  }
  // Where the merged vertex should go, into out; returns the cost there. Solves
  // the 3x3 system of the summed quadric; if it is near singular (a flat or
  // straight region), the best of the two ends and the midpoint is used.
  function bestPosition(a, b, out) {
    const qa = a * 10, qb = b * 10;
    const q = i => Q[qa + i] + Q[qb + i];
    const m00 = q(0), m01 = q(1), m02 = q(2), m11 = q(4), m12 = q(5), m22 = q(7);
    const r0 = -q(3), r1 = -q(6), r2 = -q(8);
    const det = m00 * (m11 * m22 - m12 * m12) - m01 * (m01 * m22 - m12 * m02) + m02 * (m01 * m12 - m11 * m02);
    const ax = P[a * 3], ay = P[a * 3 + 1], az = P[a * 3 + 2];
    const bx = P[b * 3], by = P[b * 3 + 1], bz = P[b * 3 + 2];
    const edge2 = (ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2;
    if (Math.abs(det) > 1e-10) {
      const x = (r0 * (m11 * m22 - m12 * m12) - m01 * (r1 * m22 - m12 * r2) + m02 * (r1 * m12 - m11 * r2)) / det;
      const y = (m00 * (r1 * m22 - m12 * r2) - r0 * (m01 * m22 - m12 * m02) + m02 * (m01 * r2 - r1 * m02)) / det;
      const z = (m00 * (m11 * r2 - r1 * m12) - m01 * (m01 * r2 - r1 * m02) + r0 * (m01 * m12 - m11 * m02)) / det;
      // Only trust the optimum near the edge; far away it is a numerical artefact.
      const mx = (ax + bx) / 2, my = (ay + by) / 2, mz = (az + bz) / 2;
      if ((x - mx) ** 2 + (y - my) ** 2 + (z - mz) ** 2 <= edge2) {
        out[0] = x; out[1] = y; out[2] = z;
        return Math.max(0, costAt(qa, qb, x, y, z));
      }
    }
    let best = Infinity;
    for (const [x, y, z] of [[ax, ay, az], [bx, by, bz], [(ax + bx) / 2, (ay + by) / 2, (az + bz) / 2]]) {
      const c = costAt(qa, qb, x, y, z);
      if (c < best) { best = c; out[0] = x; out[1] = y; out[2] = z; }
    }
    return Math.max(0, best);
  }
  // Neighbours of v into nb[0..count), using a mark array instead of a Set.
  function neighbours(v, nb, markValue) {
    let count = 0;
    for (const f of vf[v]) {
      if (!faceAlive[f]) continue;
      for (let k = 0; k < 3; k++) {
        const u = F[f * 3 + k];
        if (u !== v && mark[u] !== markValue) { mark[u] = markValue; nb[count++] = u; }
      }
    }
    return count;
  }
  // Merge b into a at (x, y, z). Returns false (and changes nothing) when the
  // collapse is refused.
  function collapse(a, b, x, y, z) {
    let s0 = -1, s1 = -1, nShared = 0;
    for (const f of vf[a]) {
      if (faceAlive[f] && (F[f * 3] === b || F[f * 3 + 1] === b || F[f * 3 + 2] === b)) {
        if (nShared === 0) s0 = f; else s1 = f;
        nShared++;
      }
    }
    if (nShared !== 2) return false;
    // Link condition: a and b may share only the two corners across the edge.
    const o0 = opposite(s0, a, b), o1 = opposite(s1, a, b);
    if (o0 === o1) return false;
    const nb = neighbours(b, nbB, ++epoch);
    let common = 0;
    for (let i = 0; i < nb; i++) {
      const u = nbB[i];
      if (u !== a && isNeighbour(a, u)) { if (u !== o0 && u !== o1) return false; common++; }
    }
    if (common !== 2) return false;
    // Keep fans small: a vertex joining too many triangles makes slivers and
    // slows every later check around it.
    if (neighbours(a, nbA, ++epoch) + nb - 4 > maxValence) return false;
    // No triangle may turn over or collapse.
    for (let pass = 0; pass < 2; pass++) {
      const v = pass ? b : a;
      for (const f of vf[v]) {
        if (!faceAlive[f] || f === s0 || f === s1) continue;
        if (!normal(f, before)) continue;
        const i0 = F[f * 3], i1 = F[f * 3 + 1], i2 = F[f * 3 + 2];
        const p0 = i0 === a || i0 === b, p1 = i1 === a || i1 === b, p2 = i2 === a || i2 === b;
        const area = cross(
          p0 ? x : P[i0 * 3], p0 ? y : P[i0 * 3 + 1], p0 ? z : P[i0 * 3 + 2],
          p1 ? x : P[i1 * 3], p1 ? y : P[i1 * 3 + 1], p1 ? z : P[i1 * 3 + 2],
          p2 ? x : P[i2 * 3], p2 ? y : P[i2 * 3 + 1], p2 ? z : P[i2 * 3 + 2], after);
        if (!(area > 1e-10)) return false;
        if (before[0] * after[0] + before[1] * after[1] + before[2] * after[2] < minCos) return false;
      }
    }
    // Apply.
    faceAlive[s0] = 0;
    faceAlive[s1] = 0;
    const merged = [];
    for (const f of vf[a]) if (faceAlive[f]) merged.push(f);
    for (const f of vf[b]) {
      if (!faceAlive[f]) continue;
      for (let k = 0; k < 3; k++) if (F[f * 3 + k] === b) F[f * 3 + k] = a;
      merged.push(f);
    }
    vf[a] = merged;
    vf[b] = [];
    vertAlive[b] = 0;
    P[a * 3] = x; P[a * 3 + 1] = y; P[a * 3 + 2] = z;
    for (let i = 0; i < 10; i++) Q[a * 10 + i] += Q[b * 10 + i];
    stamp[a]++;
    return true;
  }
  function opposite(f, a, b) {
    for (let k = 0; k < 3; k++) { const u = F[f * 3 + k]; if (u !== a && u !== b) return u; }
    return -1;
  }
  function isNeighbour(a, u) {
    for (const f of vf[a]) if (faceAlive[f] && (F[f * 3] === u || F[f * 3 + 1] === u || F[f * 3 + 2] === u)) return true;
    return false;
  }
}

// Binary min-heap over parallel typed arrays, grown as needed.
class Heap {
  constructor(cap = 1 << 16) { this.n = 0; this.alloc(cap); }
  alloc(cap) {
    const old = this.cost;
    this.cap = cap;
    this.cost = new Float64Array(cap);
    this.data = new Int32Array(cap * 4); // a, b, stampA, stampB
    this.pos = new Float64Array(cap * 3);
    if (old) { this.cost.set(old); this.data.set(this.oldData); this.pos.set(this.oldPos); }
  }
  push(cost, a, b, sa, sb, x, y, z) {
    if (this.n === this.cap) { this.oldData = this.data; this.oldPos = this.pos; this.alloc(this.cap * 2); this.oldData = this.oldPos = null; }
    let i = this.n++;
    this.set(i, cost, a, b, sa, sb, x, y, z);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (this.cost[p] <= this.cost[i]) break;
      this.swap(i, p);
      i = p;
    }
  }
  pop(out) {
    if (!this.n) return false;
    out.cost = this.cost[0];
    out.a = this.data[0]; out.b = this.data[1]; out.sa = this.data[2]; out.sb = this.data[3];
    out.x = this.pos[0]; out.y = this.pos[1]; out.z = this.pos[2];
    this.n--;
    if (this.n) {
      this.copy(this.n, 0);
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1;
        let m = i;
        if (l < this.n && this.cost[l] < this.cost[m]) m = l;
        if (r < this.n && this.cost[r] < this.cost[m]) m = r;
        if (m === i) break;
        this.swap(i, m);
        i = m;
      }
    }
    return true;
  }
  set(i, cost, a, b, sa, sb, x, y, z) {
    this.cost[i] = cost;
    const d = i * 4; this.data[d] = a; this.data[d + 1] = b; this.data[d + 2] = sa; this.data[d + 3] = sb;
    const p = i * 3; this.pos[p] = x; this.pos[p + 1] = y; this.pos[p + 2] = z;
  }
  copy(from, to) {
    this.cost[to] = this.cost[from];
    for (let k = 0; k < 4; k++) this.data[to * 4 + k] = this.data[from * 4 + k];
    for (let k = 0; k < 3; k++) this.pos[to * 3 + k] = this.pos[from * 3 + k];
  }
  swap(i, j) {
    let t = this.cost[i]; this.cost[i] = this.cost[j]; this.cost[j] = t;
    for (let k = 0; k < 4; k++) { t = this.data[i * 4 + k]; this.data[i * 4 + k] = this.data[j * 4 + k]; this.data[j * 4 + k] = t; }
    for (let k = 0; k < 3; k++) { t = this.pos[i * 3 + k]; this.pos[i * 3 + k] = this.pos[j * 3 + k]; this.pos[j * 3 + k] = t; }
  }
}
