import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { Canvas, useThree } from '@react-three/fiber';
import { OrbitControls, Grid } from '@react-three/drei';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader';

function Model({ url, buildDirection, currentLayer, layerHeight }) {
  const blueMeshRef = useRef();
  const redMeshRef = useRef();
  const { gl } = useThree();

  useEffect(() => {
    const loader = new STLLoader();
    loader.load(url, (geometry) => {
      geometry.center();

      // Calculate the bounding box to get size for scaling and positioning
      geometry.computeBoundingBox();
      const boundingBox = geometry.boundingBox;
      if (!boundingBox) return;

      const size = boundingBox.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z);
      const scaleFactor = 1 / maxDim;

      // Set materials
      const blueMaterial = new THREE.MeshStandardMaterial({
        color: 0x3b82f6, // Blue color for the main model
        metalness: 0.4,
        roughness: 0.5,
        clippingPlanes: [],
      });

      const redMaterial = new THREE.MeshStandardMaterial({
        color: 0xff0000, // Red color for the highlighted layer
        metalness: 0.4,
        roughness: 0.5,
        clippingPlanes: [],
      });

      // Apply geometry and transformation to blue and red meshes
      if (blueMeshRef.current && redMeshRef.current) {
        blueMeshRef.current.geometry = geometry;
        redMeshRef.current.geometry = geometry.clone();

        blueMeshRef.current.material = blueMaterial;
        redMeshRef.current.material = redMaterial;

        blueMeshRef.current.scale.set(scaleFactor, scaleFactor, scaleFactor);
        redMeshRef.current.scale.set(scaleFactor, scaleFactor, scaleFactor);

        // Apply the same rotation based on build direction to both models
        if (buildDirection === 'X') {
          blueMeshRef.current.rotation.x = -Math.PI / 2;
          redMeshRef.current.rotation.x = -Math.PI / 2;
        } else if (buildDirection === 'Y') {
          blueMeshRef.current.rotation.x = Math.PI / 2;
          redMeshRef.current.rotation.x = Math.PI / 2;
        }

        // Move models up by half of their height to align the base with the origin
        const scaledHeight = size.z * scaleFactor;
        blueMeshRef.current.position.y = scaledHeight / 2;
        redMeshRef.current.position.y = scaledHeight / 2;
      }
    });
  }, [url, buildDirection]);

  // Update clipping planes for blue and red models
  useEffect(() => {
    if (blueMeshRef.current && redMeshRef.current && currentLayer !== null && layerHeight !== null) {
      const scaleFactor = blueMeshRef.current.scale.y;
      const adjustedLayerHeight = layerHeight * scaleFactor;

      // Blue model: clipping everything below current layer
      const blueClipPlane = new THREE.Plane(new THREE.Vector3(0, -1, 0), currentLayer * adjustedLayerHeight);

      // Red model: clipping everything except the current layer (inverted mask)
      const redClipPlaneTop = new THREE.Plane(new THREE.Vector3(0, 1, 0), -(currentLayer * adjustedLayerHeight));
      const redClipPlaneBottom = new THREE.Plane(new THREE.Vector3(0, -1, 0), (currentLayer + 1) * adjustedLayerHeight);

      // Enable local clipping
      gl.localClippingEnabled = true;

      // Apply clipping planes
      blueMeshRef.current.material.clippingPlanes = [blueClipPlane];
      blueMeshRef.current.material.needsUpdate = true;

      redMeshRef.current.material.clippingPlanes = [redClipPlaneTop, redClipPlaneBottom];
      redMeshRef.current.material.needsUpdate = true;
    }
  }, [currentLayer, layerHeight, gl]);

  return (
    <>
      <mesh ref={blueMeshRef} />
      <mesh ref={redMeshRef} />
    </>
  );
}

function Scene({ fileUrl, buildDirection, currentLayer, layerHeight, controlsRef }) {
  return (
    <Canvas
      shadows
      camera={{ position: [0, 0, 5], fov: 50 }}
      style={{ background: '#f3f4f6' }}
    >
      {/* Ambient light for soft illumination */}
      <ambientLight intensity={0.7} />
      {/* Directional light for strong shadows */}
      <directionalLight position={[5, 10, 5]} intensity={1.2} castShadow />

      <Model url={fileUrl} buildDirection={buildDirection} currentLayer={currentLayer} layerHeight={layerHeight} />

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

export function ModelViewer({ fileData, buildDirection = 'Z', currentLayer, layerHeight, controlsRef }) {
  return (
    <div className="w-full h-[600px] rounded-lg overflow-hidden">
      {fileData ? (
        <Scene fileUrl={fileData} buildDirection={buildDirection} currentLayer={currentLayer} layerHeight={layerHeight} controlsRef={controlsRef} />
      ) : (
        <div className="w-full h-full flex items-center justify-center bg-gray-100">
          <p className="text-gray-500">No model loaded</p>
        </div>
      )}
    </div>
  );
}
