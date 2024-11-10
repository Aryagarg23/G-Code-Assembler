import React, { useState, useEffect, useRef } from 'react';
import { useLocation } from 'react-router-dom';
import { Card, CardContent, CardHeader } from "./components/ui/card";
import { Slider } from "./components/ui/slider";
import { ModelViewer } from './components/ModelViewer';  // Updated import
import { ViewerControls } from './components/ViewsControl'; // Import ViewerControls
import logoImage from './logo.svg';

function ViewerPage() {
  const location = useLocation();
  const [currentLayer, setCurrentLayer] = useState(0);
  const [modelData, setModelData] = useState(null);
  const [fileData, setFileData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const controlsRef = useRef(); // Create a reference for OrbitControls

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Fetch model data
        const modelResponse = await fetch('http://localhost:5001/api/model-data');
        if (!modelResponse.ok) {
          throw new Error(`Failed to fetch model data: ${modelResponse.statusText}`);
        }
        const modelData = await modelResponse.json();
        setModelData(modelData);

        // Fetch uploaded file
        const fileResponse = await fetch('http://localhost:5001/api/stl-file');
        if (!fileResponse.ok) {
          throw new Error('Failed to fetch file. Please upload a file first.');
        }
        
        const fileBlob = await fileResponse.blob();
        const fileUrl = URL.createObjectURL(fileBlob);
        setFileData(fileUrl);
        
      } catch (error) {
        console.error('Error fetching data:', error);
        setError(error.message);
      } finally {
        setLoading(false);
      }
    };

    fetchData();

    return () => {
      if (fileData) {
        URL.revokeObjectURL(fileData);
      }
    };
  }, []);

  const handleLayerChange = (value) => {
    setCurrentLayer(value[0]);
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-red-600">Error: {error}</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white shadow-sm border-b">
        <div className="mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <img 
                src={logoImage} 
                alt="KV Logo" 
                className="h-8 w-8"
              />
              <span className="ml-2 text-xl font-semibold text-gray-900">Print View Portal</span>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-[1600px] mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex gap-8">
          {/* Main Viewer Card */}
          <div className="flex-grow">
            <Card className="h-full">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <h2 className="text-xl font-semibold">Model Viewer</h2>
                  {/* Use ViewerControls here, pass the ref to allow interaction with controls */}
                  <ViewerControls controlsRef={controlsRef} />
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  {/* 3D Viewer */}
                  <ModelViewer 
                    fileData={fileData}
                    buildDirection={modelData?.detected_parameters?.build_direction || 'Z'}
                    currentLayer={currentLayer}
                    layerHeight={modelData?.detected_parameters?.layer_height || 0.2}
                    controlsRef={controlsRef} // Pass controlsRef to ModelViewer
                  />

                  {/* Layer Controls */}
                  <div className="space-y-4">
                    <div className="flex justify-between text-sm text-gray-600">
                      <span>Layer: {currentLayer}</span>
                      <span>Height: {(currentLayer * (modelData?.detected_parameters?.layer_height || 0.2)).toFixed(2)}mm</span>
                    </div>
                    <Slider
                      value={[currentLayer]}
                      onValueChange={handleLayerChange}
                      max={modelData?.detected_parameters?.num_layers - 1 || 0}
                      step={1}
                      className="w-full"
                    />
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Side Panel */}
          
<div className="w-80">
  <div className="space-y-6">
    {/* Model Details Card */}
    <Card>
<CardHeader>
  <div className="flex items-center justify-between">
    <h3 className="text-lg font-medium">Model Details</h3>
    <button
      id = 'jensenButton'
      className="bg-green-600 hover:bg-green-700 p-3 border-2 border-green-800 rounded-md transition duration-200 ease-in-out transform hover:scale-85 shadow-md"
      title="Download Model"
      onClick={() => {
        if (fileData) {
          const link = document.createElement('a');
          link.href = fileData;
          link.download = 'model.stl'; // Default name for the downloaded file
          link.click();
        }
      }}
    >
      <svg
        xmlns="http://www.w3.org/2000/svg"
        className="h-6 w-6 text-white"
        viewBox="0 0 24 24"
        fill="currentColor"
      >
        <path
          d="M12 16l4-4h-3V4h-2v8H8l4 4zM20 18H4v2h16v-2z"
        />
      </svg>
    </button>
  </div>
</CardHeader>





      <CardContent>
        {modelData && (
          <div className="space-y-4">
            {/* Print Parameters */}
            <div>
              <h4 className="text-sm font-medium mb-2">Print Parameters</h4>
              <div className="text-sm text-gray-600 space-y-1">
                <div className="flex justify-between">
                  <span>Layer Height:</span>
                  <span>{modelData.detected_parameters.layer_height.toFixed(3)}mm</span>
                </div>
                <div className="flex justify-between">
                  <span>Extrusion Width:</span>
                  <span>{modelData.detected_parameters.extrusion_width.toFixed(3)}mm</span>
                </div>
                <div className="flex justify-between">
                  <span>Total Layers:</span>
                  <span>{modelData.detected_parameters.num_layers}</span>
                </div>
                <div className="flex justify-between">
                  <span>Build Direction:</span>
                  <span>{modelData.detected_parameters.build_direction}</span>
                </div>
              </div>
            </div>

            {/* Model Statistics */}
            <div>
              <h4 className="text-sm font-medium mb-2">Model Statistics</h4>
              <div className="text-sm text-gray-600 space-y-1">
                <div className="flex justify-between">
                  <span>Triangles:</span>
                  <span>{modelData.model_stats.num_triangles.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span>Volume:</span>
                  <span>{modelData.model_stats.volume_mm3.toFixed(2)}mm³</span>
                </div>
                <div className="flex justify-between">
                  <span>Surface Area:</span>
                  <span>{modelData.model_stats.surface_area_mm2.toFixed(2)}mm²</span>
                </div>
              </div>
            </div>

            {/* Dimensions */}
            <div>
              <h4 className="text-sm font-medium mb-2">Dimensions</h4>
              <div className="text-sm text-gray-600 space-y-1">
                <div className="flex justify-between">
                  <span>X:</span>
                  <span>{modelData.model_stats.dimensions_mm.x.toFixed(2)}mm</span>
                </div>
                <div className="flex justify-between">
                  <span>Y:</span>
                  <span>{modelData.model_stats.dimensions_mm.y.toFixed(2)}mm</span>
                </div>
                <div className="flex justify-between">
                  <span>Z:</span>
                  <span>{modelData.model_stats.dimensions_mm.z.toFixed(2)}mm</span>
                </div>
              </div>
            </div>

            {/* Quality Metrics */}
            <div>
              <h4 className="text-sm font-medium mb-2">Quality Metrics</h4>
              <div className="text-sm text-gray-600 space-y-1">
                <div className="flex justify-between">
                  <span>Degenerate Triangles:</span>
                  <span>{modelData.quality_metrics.degenerate_triangles}</span>
                </div>
                <div className="flex justify-between">
                  <span>Small Triangles:</span>
                  <span>{modelData.quality_metrics.small_triangles}</span>
                </div>
                <div className="flex justify-between">
                  <span>Min Edge Length:</span>
                  <span>{modelData.quality_metrics.edge_stats.min_length.toFixed(3)}mm</span>
                </div>
                <div className="flex justify-between">
                  <span>Max Edge Length:</span>
                  <span>{modelData.quality_metrics.edge_stats.max_length.toFixed(3)}mm</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>

    {/* Layer Analysis Card */}
    <Card>
      <CardHeader>
        <h3 className="text-lg font-medium">Current Layer Info (RED)</h3>
      </CardHeader>
      <CardContent>
        {modelData && modelData.layer_analysis && modelData.layer_analysis[currentLayer] && (
          <div className="space-y-2 text-sm text-gray-600">
            <div className="flex justify-between">
              <span>Height:</span>
              <span>{modelData.layer_analysis[currentLayer].z_height.toFixed(2)}mm</span>
            </div>
            <div className="flex justify-between">
              <span>Area:</span>
              <span>{modelData.layer_analysis[currentLayer].area_mm2.toFixed(2)}mm²</span>
            </div>
            <div className="flex justify-between">
              <span>Triangles:</span>
              <span>{modelData.layer_analysis[currentLayer].num_triangles}</span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  </div>
</div>
        </div>
      </div>
    </div>
  );
}

export default ViewerPage;
