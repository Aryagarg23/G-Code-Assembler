// Engine tests: node --test test/ (from gcode-viewer/). Uses the challenge's
// own files in KV_Challenge/Public Materials, including the CAD models that
// came with them, as ground truth.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { legacyRead, legacyMesh } from '../src/engine/legacy.mjs';
import { parseGcode } from '../src/engine/parse.mjs';
import { buildSolid, toTriangles, checkClosed, RESOLUTIONS } from '../src/engine/solid.mjs';
import { analyze, massProperties } from '../src/engine/stl.mjs';
import { processGcode } from '../src/engine/pipeline.mjs';

const KV = new URL('../../KV_Challenge/Public Materials/', import.meta.url);
const gcode = name => fs.readFileSync(new URL(`${name}.gcode`, KV), 'utf8');
const close = (actual, expected, rel, label) => assert.ok(Math.abs(actual - expected) <= Math.abs(expected) * rel, `${label}: ${actual} vs ${expected}`);

function extent(tris) {
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < tris.length; i += 3) for (let k = 0; k < 3; k++) { min[k] = Math.min(min[k], tris[i + k]); max[k] = Math.max(max[k], tris[i + k]); }
  return max.map((v, k) => v - min[k]);
}
// Ground truth: the CAD STL Kinetic Vision supplied with each G-code file.
function truth(name) {
  const buf = fs.readFileSync(new URL(`${name}.stl`, KV));
  const n = buf.readUInt32LE(80);
  const tris = new Float32Array(n * 9);
  for (let t = 0; t < n; t++) for (let k = 0; k < 9; k++) tris[t * 9 + k] = buf.readFloatLE(84 + t * 50 + 12 + k * 4);
  return { volume: massProperties(tris).volume, size: extent(tris) };
}

// Why: the "Hackathon mesh" option claims to be what the MakeUC build made.
// These are the original Python's numbers (Backend/flask_back.py with
// numpy-stl, run 2026-10-06) for the same files.
const PYTHON = {
  SquarePrism: { segs: 57718, layers: 83, tris: 678216, volume: 20655.0, dirs: 'Y', zLevels: 129 },
  KV_Monogram: { segs: 39981, layers: 278, tris: 407376, volume: 30076.6, dirs: 'X', zLevels: 324 },
  '3DBenchy': { segs: 62480, layers: 198, tris: 662388, volume: 9990.3, dirs: 'X', zLevels: 244 },
};
for (const [name, py] of Object.entries(PYTHON)) {
  test(`hackathon port matches the original Python on ${name}`, () => {
    const read = legacyRead(gcode(name));
    assert.equal(read.nSegs, py.segs);
    assert.equal(read.nLayers, py.layers);
    const tris = legacyMesh(read);
    assert.equal(tris.length / 9, py.tris);
    const rep = analyze(tris);
    close(rep.model_stats.volume_mm3, py.volume, 1e-3, 'volume');
    assert.equal(rep.detected_parameters.build_direction, py.dirs);
    assert.equal(rep.detected_parameters.num_layers, py.zLevels);
    close(rep.detected_parameters.layer_height, 0.2, 1e-6, 'layer height');
    close(rep.detected_parameters.extrusion_width, 0.4, 1e-6, 'width');
  });
}

// Why: the rewrite exists because the hackathon mesh is not the part. The CAD
// model is the reference: a solid mesh must match its size and volume, and
// the hackathon mesh does not (it includes the purge line and is boxes).
for (const [name, volTol] of [['SquarePrism', 0.01], ['3DBenchy', 0.02]]) {
  test(`solid mesh of ${name} is closed and matches the CAD model; the hackathon mesh does not`, () => {
    const ref = truth(name);
    const parsed = parseGcode(gcode(name));
    const solid = buildSolid(parsed, RESOLUTIONS.coarse);
    assert.equal(checkClosed(solid).closed, true);
    const tris = toTriangles(solid);
    close(massProperties(tris).volume, ref.volume, volTol, 'solid volume');
    extent(tris).forEach((v, k) => assert.ok(Math.abs(v - ref.size[k]) < 0.1, `solid size ${k}: ${v} vs ${ref.size[k]}`));
    const legacy = legacyMesh(legacyRead(gcode(name)));
    assert.ok(Math.abs(massProperties(legacy).volume - ref.volume) > ref.volume * 0.3, 'hackathon volume is far off');
    assert.ok(extent(legacy).some((v, k) => Math.abs(v - ref.size[k]) > 10), 'hackathon size includes the purge line');
  });
}

// Why: a mesh for simulation must be closed: every edge shared by exactly two
// triangles, facing opposite ways. Sliver triangles once flipped the facing
// when a grid point sat on a bead edge (0.225 here, with 0.1 cells).
test('solid meshes are closed at every resolution, including beads that land on grid points', () => {
  const layer = (z, segs) => ({ z, h: 0.2, segs: Float64Array.from(segs) });
  const parts = [
    [layer(0.2, [0, 0, 10, 0, 0.45])],
    [layer(0.2, [0, 0, 10, 0, 0.45, 10, 0, 10, 10, 0.45]), layer(0.4, [0, 0, 10, 0, 0.45])],
  ];
  for (const layers of parts) for (const res of Object.values(RESOLUTIONS)) {
    for (const fill of [true, false]) {
      const result = checkClosed(buildSolid({ layers }, { ...res, cell: Math.min(res.cell, 0.1), fill }));
      assert.equal(result.closed, true, JSON.stringify(result));
    }
  }
});

// Why: a bead's shape comes from the filament pushed out. One straight bead
// must come back with the volume of filament that made it (stadium section
// times length, plus its round ends).
test('one bead has the volume of the filament that made it', () => {
  const area = Math.PI * (1.75 / 2) ** 2;
  const L = 20, h = 0.2, w = 0.45;
  const stadium = (w - h) * h + Math.PI * h * h / 4;
  const e = stadium * L / area;
  const text = `; CHANGE_LAYER\nG90\nM83\nG1 X0 Y0 Z0.2\nG1 X${L} Y0 E${e.toFixed(6)}\n`;
  const parsed = parseGcode(text);
  close(parsed.layers[0].segs[4], w, 1e-3, 'width from E');
  const tris = toTriangles(buildSolid(parsed, { cell: 0.05, subSlices: 8 }));
  // The two round ends together are the stadium spun round a vertical axis:
  // the integral of pi * r(t)^2 up the height, r(t) = (w - h)/2 + sqrt((h/2)^2 - (t - h/2)^2).
  let ends = 0;
  for (let i = 0; i < 1000; i++) {
    const t = (i + 0.5) / 1000 * h;
    ends += Math.PI * ((w - h) / 2 + Math.sqrt((h / 2) ** 2 - (t - h / 2) ** 2)) ** 2 * h / 1000;
  }
  close(massProperties(tris).volume, stadium * L + ends, 0.02, 'bead volume');
});

// Why: in relative E mode (M83, which Bambu, Prusa and others use) the
// hackathon reader re-added the last E on every following line, so travel
// moves with no E were drawn as beads.
test('travel moves are not beads (the hackathon reader drew them in relative E mode)', () => {
  const text = '; CHANGE_LAYER\nM83\nG1 X0 Y0 Z0.2\nG1 X10 Y0 E0.5\nG0 X40 Y30\nG1 X50 Y30 E0.5\n';
  const { layers } = parseGcode(text);
  assert.equal(layers[0].segs.length / 5, 2);
  const legacy = legacyRead(text);
  let extruding = 0;
  for (let i = 0; i < legacy.segs.length; i += 7) extruding += legacy.segs[i + 5];
  assert.equal(extruding, 3, 'hackathon reader counts the travel move');
});

// Why: after G92 E0 in absolute E mode the next E is measured from zero.
test('G92 E resets are followed in absolute E mode', () => {
  const layers = [0.2, 0.4, 0.6].map(z => `; CHANGE_LAYER\nG92 E0\nG1 X0 Y0 Z${z}\nG1 X10 Y0 E0.5\n`).join('');
  const parsed = parseGcode(`M82\nG90\n${layers}`);
  assert.equal(parsed.layers.length, 3);
  const widths = parsed.layers.map(l => l.segs[4]);
  assert.ok(widths.every(w => Math.abs(w - widths[0]) < 1e-9), 'same bead every layer');
});

test('relative positioning, inches and line numbers are read', () => {
  const text = '; CHANGE_LAYER\nG21\nM83\nN10 G1 X0 Y0 Z0.2*55\nG91\nG1 X10 E0.5\nG1 Y10 E0.5\nG90\nG20\nG1 X1 Y0 E0.02\n';
  const { layers } = parseGcode(text);
  const s = layers[0].segs;
  assert.deepEqual([s[2], s[3]], [10, 0]);
  assert.deepEqual([s[7], s[8]], [10, 10]);
  assert.ok(Math.abs(s[12] - 25.4) < 1e-9 && s[13] === 0);
});

// Why: Bambu Studio writes walls as arcs (9,299 in the Benchy). The hackathon
// reader drew each as one straight chord.
test('arcs are followed within tolerance, with I/J and with R', () => {
  const r = 10;
  for (const move of ['G3 X0 Y10 I-10 J0 E1', 'G3 X0 Y10 R10 E1']) {
    const text = `; CHANGE_LAYER\nM83\nG1 X10 Y0 Z0.2\n${move}\n`;
    const { layers } = parseGcode(text, { arcTolerance: 0.005 });
    const s = layers[0].segs;
    assert.ok(s.length / 5 > 10, move);
    for (let i = 0; i < s.length; i += 5) {
      const mx = (s[i] + s[i + 2]) / 2, my = (s[i + 1] + s[i + 3]) / 2;
      assert.ok(Math.abs(Math.hypot(mx, my) - r) < 0.006, `${move}: chord midpoint ${Math.hypot(mx, my)}`);
      assert.ok(Math.hypot(s[i], s[i + 1]) > r - 1e-9 && mx >= -1e-9 && my >= -1e-9, 'stays on the quarter circle');
    }
  }
});

// Why: the mesh should be the part, not the purge line, skirt or supports.
test('only the part is kept: start code, purge and supports are left out and counted', () => {
  const text = [
    'M83', 'G1 X0 Y0 Z0.2', 'G1 X50 Y0 E5 ; purge before the first layer',
    '; CHANGE_LAYER', '; FEATURE: Skirt', 'G1 X0 Y5 E1', '; FEATURE: Outer wall', 'G1 X10 Y5 E0.5',
    '; FEATURE: Support', 'G1 X10 Y20 E0.5', '; FEATURE: Inner wall', 'G1 X10 Y6 E0.5',
  ].join('\n');
  const { layers, stats } = parseGcode(text);
  assert.equal(layers[0].segs.length / 5, 2);
  assert.equal(stats.outsidePart, 1);
  assert.deepEqual(stats.excluded, { Skirt: 1, Support: 1 });
});

test('bead widths read off the challenge files match the slicer line widths', () => {
  const { stats } = parseGcode(gcode('3DBenchy'));
  // Bambu Studio set 0.42 to 0.5 mm line widths for this print.
  assert.ok(stats.medianWidth > 0.4 && stats.medianWidth < 0.5, String(stats.medianWidth));
  close(stats.medianLayerHeight, 0.2, 1e-6, 'layer height');
});

test('the STL file is binary with one 50-byte record per triangle', () => {
  const { stl, summary } = processGcode(gcode('SquarePrism'), { resolution: 'coarse' });
  const view = new DataView(stl);
  assert.equal(view.getUint32(80, true), summary.triangles);
  assert.equal(stl.byteLength, 84 + 50 * summary.triangles);
  assert.equal(summary.closed, true);
  assert.ok(summary.volume > 0);
  const legacy = processGcode(gcode('SquarePrism'), { method: 'legacy' }).summary;
  assert.equal(legacy.closed, false);
  assert.equal(legacy.volume, null);
});

// Why: people use more than Bambu Studio. Each slicer marks layers and
// section types its own way; the part must come out the same from each.
const SLICERS = {
  'PrusaSlicer / SuperSlicer': ['M83', 'G1 Z.2', 'G1 X0 Y0', 'G1 X40 Y0 E3 ; intro line', ';LAYER_CHANGE', ';Z:0.2', ';HEIGHT:0.2', ';TYPE:Skirt/Brim', 'G1 X0 Y5 E1', ';TYPE:External perimeter', 'G1 X10 Y5 E0.5', ';TYPE:Support material', 'G1 X10 Y20 E0.5', ';TYPE:Perimeter', 'G1 X10 Y6 E0.5', ';TYPE:Custom', '; filament end gcode', 'G1 X50 Y50 E2'],
  Cura: ['M82', 'G92 E0', 'G1 Z0.2', 'G1 X0 Y0', 'G1 X40 Y0 E3', ';LAYER_COUNT:1', ';LAYER:0', ';TYPE:SKIRT', 'G92 E0', 'G1 X0 Y5 E1', ';TYPE:WALL-OUTER', 'G1 X10 Y5 E1.5', ';TYPE:SUPPORT', 'G1 X10 Y20 E2', ';TYPE:WALL-INNER', 'G1 X10 Y6 E2.5', ';End of Gcode', 'G1 X50 Y50 E9'],
  Simplify3D: ['M83', 'G1 Z0.2', 'G1 X0 Y0', 'G1 X40 Y0 E3', '; layer 1, Z = 0.2', '; feature skirt', 'G1 X0 Y5 E1', '; feature outer perimeter', 'G1 X10 Y5 E0.5', '; feature support', 'G1 X10 Y20 E0.5', '; feature inner perimeter', 'G1 X10 Y6 E0.5', '; layer end', 'G1 X50 Y50 E2'],
  ideaMaker: ['M83', 'G1 Z0.2', 'G1 X0 Y0', 'G1 X40 Y0 E3', ';LAYER:0', ';TYPE:SKIRT', 'G1 X0 Y5 E1', ';TYPE:WALL-OUTER', 'G1 X10 Y5 E0.5', ';TYPE:SUPPORT', 'G1 X10 Y20 E0.5', ';TYPE:WALL-INNER', 'G1 X10 Y6 E0.5'],
};
for (const [slicer, lines] of Object.entries(SLICERS)) {
  test(`${slicer}: the part is kept, intro line, skirt, support and end code are not`, () => {
    const { layers } = parseGcode(lines.join('\n'));
    assert.equal(layers.length, 1);
    const s = layers[0].segs;
    assert.deepEqual([...s].filter((_, i) => i % 5 !== 4), [0, 5, 10, 5, 10, 20, 10, 6], slicer);
  });
}

test('a file with no slicer comments keeps every extrusion', () => {
  const { layers, stats } = parseGcode('M83\nG1 Z0.2\nG1 X0 Y0\nG1 X10 Y0 E0.5\nG1 X10 Y10 E0.5\n');
  assert.equal(layers[0].segs.length / 5, 2);
  assert.equal(stats.keptEverything, true);
});

// Why: with M200 D the firmware reads E as mm^3 of filament; reading it as a
// length would make every bead about 2.4 times too wide (1 / (pi * 0.875^2)).
test('volumetric E (M200 D) gives the same bead as the length form', () => {
  const area = Math.PI * (1.75 / 2) ** 2;
  const asLength = parseGcode('; CHANGE_LAYER\nM83\nG1 X0 Y0 Z0.2\nG1 X10 Y0 E0.4\n').layers[0].segs[4];
  const asVolume = parseGcode(`; CHANGE_LAYER\nM83\nM200 D1.75\nG1 X0 Y0 Z0.2\nG1 X10 Y0 E${0.4 * area}\n`).layers[0].segs[4];
  close(asVolume, asLength, 1e-9, 'width');
});
