// G-code in, STL and a summary out. Runs in a Web Worker (engine/worker.js)
// in the browser, and directly in the tests. This replaces the Flask
// endpoints (POST /api/upload-gcode, GET /api/stl-file, GET /api/model-data).
//
// The summary holds only what someone converting a print to a mesh needs:
// whether the mesh is a closed solid (and so has a real volume), its size,
// how big the file is, what the G-code said about layers and beads, and what
// was left out. The hackathon report's guesses (build direction, layer height
// read back off the mesh, edge-length and sliver counts) are not shown.

import { legacyRead, legacyMesh } from './legacy.mjs';
import { parseGcode } from './parse.mjs';
import { buildSolid, toTriangles, checkClosed, RESOLUTIONS } from './solid.mjs';
import { massProperties, toBinaryStl } from './stl.mjs';
import { simplify } from './simplify.mjs';

/**
 * @param {string} text  G-code
 * @param {{ name?: string, method?: 'solid' | 'legacy', resolution?: keyof typeof RESOLUTIONS, fill?: boolean, tolerance?: number, onProgress?: (f: number) => void }} options
 *   tolerance: simplification limit in mm (0 = off; default 0.02)
 */
export function processGcode(text, options = {}) {
  const name = options.name ?? 'model.gcode';
  const method = options.method ?? 'solid';
  const started = Date.now();

  if (method === 'legacy') {
    const read = legacyRead(text);
    const tris = legacyMesh(read);
    const stl = toBinaryStl(tris, name);
    return {
      stl,
      summary: {
        method,
        closed: false,
        triangles: tris.length / 9,
        fileBytes: stl.byteLength,
        volume: null, // overlapping boxes: no meaningful volume
        size: extent(tris),
        layers: [],
        layerCount: read.nLayers,
        segments: read.nSegs,
        ms: Date.now() - started,
      },
    };
  }

  const parsed = parseGcode(text);
  if (!parsed.layers.length) throw new Error('No extrusion found in this file.');
  const fill = options.fill ?? true;
  const solid = buildSolid(parsed, { ...(RESOLUTIONS[options.resolution ?? 'coarse'] ?? RESOLUTIONS.coarse), fill, onProgress: f => options.onProgress?.(0.8 * f) });
  const tolerance = options.tolerance ?? 0.02;
  let mesh = solid;
  let simplified = null;
  if (tolerance > 0) {
    const small = simplify(solid, { tolerance });
    // Only keep the simplified mesh if it is still closed; otherwise say so
    // and hand over the full one.
    simplified = { tolerance, before: solid.indices.length / 3, kept: checkClosed(small).closed };
    if (simplified.kept) mesh = small;
    options.onProgress?.(1);
  }
  const tris = toTriangles(mesh);
  const stl = toBinaryStl(tris, name);
  const { closed } = checkClosed(mesh);
  const { stats, settings, layers } = parsed;
  return {
    stl,
    summary: {
      method,
      closed,
      triangles: mesh.indices.length / 3,
      simplified,
      fileBytes: stl.byteLength,
      volume: closed ? massProperties(tris).volume : null,
      size: extent(tris),
      cell: solid.cell,
      subSlices: solid.subSlices,
      filled: fill,
      layers: layers.map(l => ({ z: l.z, h: l.h, beads: l.segs.length / 5 })),
      layerCount: layers.length,
      layerHeight: stats.medianLayerHeight,
      beadWidth: stats.medianWidth,
      filament: stats.extrudedVolume,
      filamentDiameter: settings.filamentDiameter,
      leftOut: { ...stats.excluded, ...(stats.outsidePart ? { 'Start and end code': stats.outsidePart } : {}) },
      keptEverything: stats.keptEverything,
      ms: Date.now() - started,
    },
  };
}

function extent(tris) {
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < tris.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      if (tris[i + k] < min[k]) min[k] = tris[i + k];
      if (tris[i + k] > max[k]) max[k] = tris[i + k];
    }
  }
  return { x: max[0] - min[0], y: max[1] - min[1], z: max[2] - min[2] };
}

export { RESOLUTIONS };
