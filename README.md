# G-Code Assembler

A web tool that turns G-code into a watertight STL: upload a `.gcode` file, get a real-time 3D preview and a mesh you can download.

Built in 36-ish hours at MakeUC 2024 (November 2024). Winner — Kinetic Vision Challenge (3D Printing / 3D Modeling).

## What it does

Most slicers go the other direction: mesh in, G-code out. This goes backwards. You give it a `.gcode` file — the actual toolpath a printer would run — and it reconstructs a solid mesh from the extrusion moves, then lets you spin it around in the browser before you download it.

The motivating case is the Kinetic Vision challenge itself: you're handed G-code and need to get back a model you can inspect and modify, without a printer on hand to just run it and look.

There's also a path that skips the browser entirely: a Jupyter notebook that runs the same G-code-to-STL pipeline for people who just want the file.

## How it works

- `Backend/flask_back.py` is the whole backend. A `GcodeReader` class parses FDM G-code line by line, tracking absolute/relative extrusion mode, and turns each extruding move into a line segment tagged with its layer and a segment id.
- Each segment becomes a rectangular prism (width x height x length) with 8 vertices and 12 triangular faces, extruded along the toolpath — that's the watertight mesh, written out with `numpy-stl`.
- `STLAutoAnalyzer` reads a generated STL back and infers print parameters from the geometry alone: layer height from the spacing between Z levels, extrusion width from horizontal edge lengths, build direction, volume, surface area, and per-layer triangle/area stats.
- Flask exposes three endpoints: `POST /api/upload-gcode` (parse and mesh), `GET /api/stl-file` (download the result), `GET /api/model-data` (the analyzer's report, for the UI).
- `gcode-viewer/` is the React front end — `react-router` for an upload page and a viewer page, `@react-three/fiber` and `three.js` for the interactive STL preview.
- `Backend/test.ipynb` runs the same `GcodeReader` → STL pipeline outside the web app, for generating files directly.

## In the browser (2026)

`gcode-viewer/` now runs entirely in the browser; the Flask server is not needed. Use it at [aryagarg23.com/gcode-to-stl](https://aryagarg23.com/gcode-to-stl).

- `src/engine/legacy.mjs` is `flask_back.py` ported line for line ("Hackathon mesh"): same segments, layers and triangle counts as the Python on the challenge files (checked in `test/engine.test.mjs`).
- `src/engine/parse.mjs` is a new reader: G92 resets, relative XYZ/E, inches, arcs (G2/G3, I/J or R), volumetric E (M200), and bead width from the filament each move pushes out. It keeps only the part, using the comments Bambu Studio/OrcaSlicer, PrusaSlicer/SuperSlicer, Cura, ideaMaker and Simplify3D write: start/end code, purge lines, skirts, brims, rafts, supports and wipe/prime towers are left out. Files without such comments keep every extrusion.
- `src/engine/solid.mjs` builds one closed solid ("Solid"): every bead is a rounded stadium in cross-section with round ends, unioned on a grid and meshed by marching tetrahedra. Every edge joins exactly two triangles, so the volume is real and the mesh can go into a simulation. "Fill enclosed spaces" makes infill pockets solid (the outer boundary); off, it is the part as printed.
- `src/engine/simplify.mjs` then shrinks the solid by quadric error edge collapse: triangles merge where the surface moves less than the tolerance (0.02 mm by default), never at the cost of opening the mesh or turning a triangle over. If a simplified mesh ever fails the closed check, the full one is kept and the app says so.

Against the CAD models Kinetic Vision supplied with the challenge (coarse setting):

| | Hackathon mesh | Solid | CAD |
|---|---|---|---|
| SquarePrism volume | 20,655 mm³ | 62,255 mm³ | 62,500 mm³ |
| SquarePrism size | 128.5 × 144.2 × 25.2 | 24.99 × 99.99 × 24.97 | 25 × 100 × 25 |
| 3DBenchy volume | 9,990 mm³ | 15,442 mm³ | 15,551 mm³ |
| 3DBenchy size | 133.5 × 109.7 × 48.2 | 59.98 × 31.01 × 47.98 | 60 × 31 × 48 |
| Closed | no | yes | |
| SquarePrism triangles | 678,216 | 189,108 (1,276,868 before simplifying) | |
| 3DBenchy triangles | 662,388 | 307,972 (1,145,840 before simplifying) | |

The hackathon size includes the purge line; its volume is of overlapping boxes. Limits: tops and bottoms sit within half a sample of the true height (0.1 mm at coarse); detail finer than a cell is smoothed.

```sh
cd gcode-viewer
npm install
npm start            # http://localhost:3000
npm run test:engine  # engine tests against the challenge files
```

## Run the original (2024) locally

The 2024 version split the work between the React page and a Flask server. The server is still here:

```sh
python -m pip install -r requirements.txt
python Backend/flask_back.py   # http://localhost:5001
```

The 2024 front end that called it is commit `04623fd` (`git checkout 04623fd -- gcode-viewer`). It processes one model at a time in the server process, and only regular FDM G-code.

## Prototype

A standalone script that shows the two ideas the real backend relies on — segments grouped by layer, and each segment carrying an extrusion length that maps to print time — on a toy square-spiral toolpath instead of a real upload. Illustrative only, not the production parser.

The diagram script needs Python, NumPy, and Matplotlib. Run from the repository root:

```sh
python -m pip install numpy matplotlib
python prototype/gcode_prototype.py
```

It writes the two diagrams to `prototype/figures/`. The script generates a 7-layer inward square spiral, colors each layer's path (folding back to a 4-color palette past layer 4), and estimates per-layer print time from segment length at a constant feedrate.

![Toy square-spiral toolpath, colored by layer](https://vircgxpcwyvniemqmdyi.supabase.co/storage/v1/object/public/media/writing/G-Code-Assembler/toolpath.png)
![Per-layer extrusion time profile](https://vircgxpcwyvniemqmdyi.supabase.co/storage/v1/object/public/media/writing/G-Code-Assembler/extrusion_profile.png)

## Team

Solo build by Arya ([@Aryagarg23](https://github.com/Aryagarg23)) — backend, frontend, and the mesh reconstruction.

## Links

- Devpost: https://devpost.com/software/g-code-assembler
- Demo video: https://youtu.be/q7wP98uev2o
- Writeup: https://aryagarg23.com/writing/g-code-assembler
- Site: https://aryagarg23.com
- Devpost profile: https://devpost.com/Aryagarg23

## More hackathon builds

- [Gyrus](https://github.com/Aryagarg23/Gyrus) — agentic browser that supports curiosity instead of replacing it (WeaveHacks 2025)
- [WhiteBox](https://github.com/Aryagarg23/WhiteBox) — traceable GraphRAG over medical literature (Future of Data 2024, 1st place)
- [Terminally-Addicted](https://github.com/Aryagarg23/Terminally-Addicted) — Spotify, GitHub, GPT and YouTube without leaving the terminal (HackOHI/O 2024)
- [Memento](https://github.com/Aryagarg23/Memento) — digital memory journal for Alzheimer's patients and caregivers (RevolutionUC 2024, 3rd overall)
- [Buycott](https://github.com/Aryagarg23/Buycott) — barcode scan -> parent company -> NLP stance on social issues (MakeUC 2023, 1st overall)
- [SignLink](https://github.com/Aryagarg23/SignLink) — video calls with real-time ASL fingerspelling to text (BoilerMake X 2023)
- [Kuka Arm Viz](https://github.com/Aryagarg23/Visualizing-Kuka-7-Node-Robot-Arm) — interactive 7-DOF robot arm in WebGL with inverse kinematics (RevolutionUC 2023)
- [Hi-Five](https://github.com/Aryagarg23/Hi-Five) — anonymous friend-matching on OCEAN personality vectors (SASEhack 2024)
- [Friction](https://github.com/Aryagarg23/Friction) — speculative OS + hardware that protects flow state with physical friction (Fig Build 2026)
