# BRIEF — G-Code-Assembler (repo: Aryagarg23/G-Code-Assembler)
Slug: g-code-assembler. Hackathon: MakeUC 2024, November 2024. Prizes: Winner — Kinetic Vision Challenge (3D Printing / 3D Modeling). Solo build by Arya.
What: web tool assembling G-code with real-time preview and watertight STL export — Flask backend parses G-code + mesh transforms, React front-end (three.js) for interactive viz, plus a Jupyter path for direct STL generation. Demo: https://youtu.be/q7wP98uev2o
Devpost: https://devpost.com/software/g-code-assembler
Prototype idea: parse a small embedded G-code snippet (or generate a square-spiral toolpath), chart 1: 2D toolpath plot colored by layer (max 4 colors then fold), chart 2: per-layer extrusion length/time profile (bar/step).
