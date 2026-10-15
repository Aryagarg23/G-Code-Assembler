import React, { useEffect, useMemo, useRef, useState } from 'react';
import { KIND_COLORS } from '../lib/kinds';
import * as THREE from 'three';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Grid } from '@react-three/drei';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader';

// The viewer shows the part up to the selected layer, coloured by what each
// bead is (outer wall, inner wall, solid layers, sparse infill), with the
// selected layer's toolpaths drawn on top of the cut like a slicer preview.
//
// Clipping a mesh only removes triangles, so on its own the cut would show the
// inside of a thin shell and the part would look hollow. The cut is capped
// with the stencil technique (three.js webgl_clipping_stencil): back faces add
// one, front faces subtract one, and a plane drawn at the cut fills exactly
// the pixels left non-zero, which are where the cut passes through material.
// The cap carries a texture of the layer's beads, so it reads as the layer.
const CAP = 1.2; // world units; the model is scaled to fit in 1
const TEX = 2048; // cap texture size in pixels

function stencilMaterials(plane) {
  const base = {
    depthWrite: false,
    depthTest: false,
    colorWrite: false,
    stencilWrite: true,
    stencilFunc: THREE.AlwaysStencilFunc,
    clippingPlanes: [plane],
  };
  const back = new THREE.MeshBasicMaterial({
    ...base,
    side: THREE.BackSide,
    stencilFail: THREE.IncrementWrapStencilOp,
    stencilZFail: THREE.IncrementWrapStencilOp,
    stencilZPass: THREE.IncrementWrapStencilOp,
  });
  const front = new THREE.MeshBasicMaterial({
    ...base,
    side: THREE.FrontSide,
    stencilFail: THREE.DecrementWrapStencilOp,
    stencilZFail: THREE.DecrementWrapStencilOp,
    stencilZPass: THREE.DecrementWrapStencilOp,
  });
  return { back, front };
}

// mm (G-code coordinates) to world: the mesh is centred in plan, its base on
// the grid, scaled by s, and turned Z-up to Y-up.
export function cutHeight(layerTopMm, xf) {
  return (layerTopMm - xf.minZ) * xf.s;
}

export function drawLayer(ctx, toolpaths, layerIndex, xf) {
  ctx.fillStyle = '#cfd4dc';
  ctx.fillRect(0, 0, TEX, TEX);
  if (!toolpaths) return;
  const { beads, starts } = toolpaths;
  const k = xf.s * TEX / CAP; // pixels per mm
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  for (let b = starts[layerIndex]; b < starts[layerIndex + 1]; b++) {
    const o = b * 6;
    ctx.strokeStyle = KIND_COLORS[beads[o + 5]] ?? KIND_COLORS[4];
    ctx.lineWidth = Math.max(1, beads[o + 4] * k);
    ctx.beginPath();
    ctx.moveTo(TEX / 2 + (beads[o] - xf.cx) * k, TEX / 2 - (beads[o + 1] - xf.cy) * k);
    ctx.lineTo(TEX / 2 + (beads[o + 2] - xf.cx) * k, TEX / 2 - (beads[o + 3] - xf.cy) * k);
    ctx.stroke();
  }
}

function Model({ url, kinds, toolpaths, layers, currentLayer }) {
  const meshRef = useRef();
  const stencilBackRef = useRef();
  const stencilFrontRef = useRef();
  const capRef = useRef();
  const xfRef = useRef(null);
  const canvas = useMemo(() => {
    const c = document.createElement('canvas');
    c.width = c.height = TEX;
    return c;
  }, []);
  const texture = useMemo(() => {
    const t = new THREE.CanvasTexture(canvas);
    t.colorSpace = THREE.SRGBColorSpace;
    t.anisotropy = 8;
    return t;
  }, [canvas]);
  // Bumped when a model finishes loading, so the cut is applied to it.
  const [loaded, setLoaded] = useState(0);

  useEffect(() => {
    const loader = new STLLoader();
    loader.load(url, (geometry) => {
      geometry.computeBoundingBox();
      const box = geometry.boundingBox;
      if (!box) return;
      const size = box.getSize(new THREE.Vector3());
      const s = 1 / Math.max(size.x, size.y, size.z);
      xfRef.current = { cx: (box.min.x + box.max.x) / 2, cy: (box.min.y + box.max.y) / 2, minZ: box.min.z, s };
      geometry.center();

      // One colour per triangle, from the kind of the bead it came from.
      const n = geometry.attributes.position.count;
      const colors = new Float32Array(n * 3);
      // THREE.Color already converts hex (sRGB) into the linear working space
      // under three's colour management, which is what vertex colours expect.
      const palette = KIND_COLORS.map(c => new THREE.Color(c));
      const fallback = new THREE.Color('#3b82f6');
      const perTri = kinds && kinds.length * 3 === n;
      for (let v = 0; v < n; v++) {
        const c = perTri ? palette[kinds[(v / 3) | 0]] ?? palette[4] : fallback;
        colors[v * 3] = c.r; colors[v * 3 + 1] = c.g; colors[v * 3 + 2] = c.b;
      }
      geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

      const all = [meshRef, stencilBackRef, stencilFrontRef].map(r => r.current);
      if (all.some(m => !m) || !capRef.current) return;
      for (const m of all) {
        m.geometry = geometry;
        m.scale.set(s, s, s);
        m.rotation.x = -Math.PI / 2; // STL is Z-up; three.js is Y-up
        m.position.y = (size.z * s) / 2;
      }
      meshRef.current.material = new THREE.MeshStandardMaterial({ vertexColors: true, metalness: 0.05, roughness: 0.75 });
      capRef.current.material = new THREE.MeshStandardMaterial({
        map: texture,
        metalness: 0.05,
        roughness: 0.75,
        polygonOffset: true, // win against the part's own top face at the top layer
        polygonOffsetFactor: -1,
        polygonOffsetUnits: -4,
        stencilWrite: true,
        stencilRef: 0,
        stencilFunc: THREE.NotEqualStencilFunc,
        stencilFail: THREE.ReplaceStencilOp,
        stencilZFail: THREE.ReplaceStencilOp,
        stencilZPass: THREE.ReplaceStencilOp,
      });
      setLoaded(x => x + 1);
    });
  }, [url, kinds, texture]);

  useEffect(() => {
    const mesh = meshRef.current, xf = xfRef.current;
    if (!mesh || !xf || !mesh.material || !mesh.material.isMeshStandardMaterial) return;
    if (currentLayer === null || !layers || !layers[currentLayer]) return;
    const y = cutHeight(layers[currentLayer].z, xf);
    const plane = new THREE.Plane(new THREE.Vector3(0, -1, 0), y);
    mesh.material.clippingPlanes = [plane];
    mesh.material.needsUpdate = true;
    const { back, front } = stencilMaterials(plane);
    stencilBackRef.current.material = back;
    stencilFrontRef.current.material = front;
    capRef.current.position.y = y;
    drawLayer(canvas.getContext('2d'), toolpaths, currentLayer, xf);
    texture.needsUpdate = true;
  }, [currentLayer, layers, toolpaths, loaded, canvas, texture]);

  return (
    <>
      <mesh ref={stencilBackRef} renderOrder={1} />
      <mesh ref={stencilFrontRef} renderOrder={1} />
      <mesh
        ref={capRef}
        visible={loaded > 0}
        renderOrder={1.1}
        rotation={[-Math.PI / 2, 0, 0]}
        onAfterRender={(renderer) => renderer.clearStencil()}
      >
        <planeGeometry args={[CAP, CAP]} />
      </mesh>
      <mesh ref={meshRef} renderOrder={2} />
    </>
  );
}

function Scene({ fileUrl, kinds, toolpaths, layers, currentLayer, controlsRef }) {
  return (
    <Canvas
      shadows
      gl={{ stencil: true, localClippingEnabled: true }}
      camera={{ position: [1.6, 1.3, 1.9], fov: 50 }}
      style={{ background: '#f3f4f6' }}
    >
      {/* Ambient light for soft illumination */}
      <ambientLight intensity={0.7} />
      {/* Directional light for strong shadows */}
      <directionalLight position={[5, 10, 5]} intensity={1.2} castShadow />

      <Model url={fileUrl} kinds={kinds} toolpaths={toolpaths} layers={layers} currentLayer={currentLayer} />

      {/* Ground Plane */}
      <mesh receiveShadow position={[0, -0.01, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[20, 20]} />
        <meshStandardMaterial color="#e0e0e0" />
      </mesh>

      <Grid
        renderOrder={-1}
        position={[0, 0, 0]}
        infiniteGrid
        cellSize={0.5}
        cellThickness={0.5}
        cellColor="#6b7280"
        sectionSize={2}
        sectionThickness={1}
        sectionColor="#374151"
        fadeDistance={30}
        fadeStrength={1}
        followCamera={false}
      />

      <OrbitControls ref={controlsRef} enableDamping dampingFactor={0.05} enableZoom={true} enablePan={true} enableRotate={true} />
    </Canvas>
  );
}

export function ModelViewer({ fileData, kinds, toolpaths, layers, currentLayer, controlsRef }) {
  return (
    <div className="w-full h-[600px] rounded-lg overflow-hidden">
      {fileData ? (
        <Scene fileUrl={fileData} kinds={kinds} toolpaths={toolpaths} layers={layers} currentLayer={currentLayer} controlsRef={controlsRef} />
      ) : (
        <div className="w-full h-full flex items-center justify-center bg-gray-100">
          <p className="text-gray-500">No model loaded</p>
        </div>
      )}
    </div>
  );
}
