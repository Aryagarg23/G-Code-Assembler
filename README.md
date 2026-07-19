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

## Prototype

A standalone script that shows the two ideas the real backend relies on — segments grouped by layer, and each segment carrying an extrusion length that maps to print time — on a toy square-spiral toolpath instead of a real upload. Illustrative only, not the production parser.

Run it:

```bash
cd prototype
MPLCONFIGDIR=/path/to/mplcache python gcode_prototype.py
```

It generates a 7-layer inward square spiral, colors each layer's path (folding back to a 4-color palette past layer 4), and estimates per-layer print time from segment length at a constant feedrate.

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
