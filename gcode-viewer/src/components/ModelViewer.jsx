import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Grid } from '@react-three/drei';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader';

// The viewer shows the part up to the selected layer: blue below it, the layer
// itself in red. Clipping a mesh only removes triangles, so on its own the cut
// shows the inside of a thin shell and the part looks hollow. The red layer's
// top is therefore capped with the stencil technique (three.js
// webgl_clipping_stencil): back faces add one, front faces subtract one, and a
// plane drawn at the cut fills exactly the pixels left non-zero, which are the
// pixels where the cut passes through solid material.
function capMaterials(plane) {
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

export function layerPlanes(currentLayer, layerHeight, scale) {
  const h = layerHeight * scale;
  const below = currentLayer * h; // top of the blue part, bottom of the red layer
  const top = (currentLayer + 1) * h; // top of the red layer, where the cap goes
  return {
    blue: new THREE.Plane(new THREE.Vector3(0, -1, 0), below),
    redBottom: new THREE.Plane(new THREE.Vector3(0, 1, 0), -below),
    redTop: new THREE.Plane(new THREE.Vector3(0, -1, 0), top),
    capY: top,
  };
}

function Model({ url, currentLayer, layerHeight }) {
  const blueMeshRef = useRef();
  const redMeshRef = useRef();
  const stencilBackRef = useRef();
  const stencilFrontRef = useRef();
  const capRef = useRef();
  // Bumped when a model finishes loading: loading swaps in fresh materials, so
  // the layer clipping has to be applied again (before, the first view showed
  // the whole model in the highlight colour until the slider moved).
  const [loaded, setLoaded] = useState(0);

  useEffect(() => {
    const loader = new STLLoader();
    loader.load(url, (geometry) => {
      geometry.center();
      geometry.computeBoundingBox();
      const boundingBox = geometry.boundingBox;
      if (!boundingBox) return;

      const size = boundingBox.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z);
      const scaleFactor = 1 / maxDim;
      // STL is Z-up (the printer's build direction); three.js is Y-up. Turn
      // the model so its layers stack upward, base on the grid.
      const scaledHeight = size.z * scaleFactor;

      const meshes = [blueMeshRef, redMeshRef, stencilBackRef, stencilFrontRef].map(r => r.current);
      if (meshes.some(m => !m) || !capRef.current) return;
      for (const m of meshes) {
        m.geometry = geometry;
        m.scale.set(scaleFactor, scaleFactor, scaleFactor);
        m.rotation.x = -Math.PI / 2;
        m.position.y = scaledHeight / 2;
      }

      blueMeshRef.current.material = new THREE.MeshStandardMaterial({ color: 0x3b82f6, metalness: 0.4, roughness: 0.5 });
      redMeshRef.current.material = new THREE.MeshStandardMaterial({ color: 0xff0000, metalness: 0.4, roughness: 0.5 });
      capRef.current.material = new THREE.MeshStandardMaterial({
        color: 0xff0000,
        metalness: 0.4,
        roughness: 0.5,
        stencilWrite: true,
        stencilRef: 0,
        stencilFunc: THREE.NotEqualStencilFunc,
        stencilFail: THREE.ReplaceStencilOp,
        stencilZFail: THREE.ReplaceStencilOp,
        stencilZPass: THREE.ReplaceStencilOp,
      });
      setLoaded(n => n + 1);
    });
  }, [url]);

  useEffect(() => {
    const blue = blueMeshRef.current;
    if (!blue || !blue.material || !blue.material.isMeshStandardMaterial) return;
    if (currentLayer === null || layerHeight === null) return;
    const p = layerPlanes(currentLayer, layerHeight, blue.scale.y);

    blue.material.clippingPlanes = [p.blue];
    redMeshRef.current.material.clippingPlanes = [p.redBottom, p.redTop];
    const { back, front } = capMaterials(p.redTop);
    stencilBackRef.current.material = back;
    stencilFrontRef.current.material = front;
    capRef.current.position.y = p.capY;
    for (const m of [blue.material, redMeshRef.current.material]) m.needsUpdate = true;
  }, [currentLayer, layerHeight, loaded]);

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
        <planeGeometry args={[4, 4]} />
      </mesh>
      <mesh ref={blueMeshRef} renderOrder={2} />
      <mesh ref={redMeshRef} renderOrder={2} />
    </>
  );
}

function Scene({ fileUrl, currentLayer, layerHeight, controlsRef }) {
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

      <Model url={fileUrl} currentLayer={currentLayer} layerHeight={layerHeight} />

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

export function ModelViewer({ fileData, currentLayer, layerHeight, controlsRef }) {
  return (
    <div className="w-full h-[600px] rounded-lg overflow-hidden">
      {fileData ? (
        <Scene fileUrl={fileData} currentLayer={currentLayer} layerHeight={layerHeight} controlsRef={controlsRef} />
      ) : (
        <div className="w-full h-full flex items-center justify-center bg-gray-100">
          <p className="text-gray-500">No model loaded</p>
        </div>
      )}
    </div>
  );
}
