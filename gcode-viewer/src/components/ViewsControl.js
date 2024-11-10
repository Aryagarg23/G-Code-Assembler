import React from 'react';
import { RotateCw, ZoomIn, ZoomOut } from 'lucide-react';

export function ViewerControls({ controlsRef }) {
  const handleReset = () => {
    if (controlsRef.current) {
      controlsRef.current.reset(); // Reset the camera position and target
    }
  };

  const handleZoomIn = () => {
    if (controlsRef.current) {
      controlsRef.current.dollyIn(1.5); // Zoom in - use dollyIn method of OrbitControls
      controlsRef.current.update(); // Ensure the controls are updated
    }
  };

  const handleZoomOut = () => {
    if (controlsRef.current) {
      controlsRef.current.dollyOut(1.5); // Zoom out - use dollyOut method of OrbitControls
      controlsRef.current.update(); // Ensure the controls are updated
    }
  };

  return (
    <div className="flex space-x-2">
      <button onClick={handleReset} className="p-2 hover:bg-gray-100 rounded-full" title="Reset View">
        <RotateCw size={20} />
      </button>
      <button onClick={handleZoomIn} className="p-2 hover:bg-gray-100 rounded-full" title="Zoom In">
        <ZoomIn size={20} />
      </button>
      <button onClick={handleZoomOut} className="p-2 hover:bg-gray-100 rounded-full" title="Zoom Out">
        <ZoomOut size={20} />
      </button>
    </div>
  );
}
