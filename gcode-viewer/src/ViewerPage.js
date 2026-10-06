import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, CardContent, CardHeader } from "./components/ui/card";
import { Slider } from "./components/ui/slider";
import { ModelViewer } from './components/ModelViewer';
import { ViewerControls } from './components/ViewsControl';
import logoImage from './logo.svg';
import { currentModel } from './engine/session';

const num = (v, d) => (typeof v === 'number' && Number.isFinite(v) ? v.toFixed(d) : '–');
const mb = bytes => `${(bytes / 1048576).toFixed(1)} MB`;

function Row({ label, children }) {
  return (
    <div className="flex justify-between gap-4">
      <span>{label}</span>
      <span className="text-right text-gray-900">{children}</span>
    </div>
  );
}

function Section({ title, children }) {
  return (
    <div>
      <h4 className="text-sm font-medium mb-2">{title}</h4>
      <div className="text-sm text-gray-600 space-y-1">{children}</div>
    </div>
  );
}

function ViewerPage() {
  const navigate = useNavigate();
  const [currentLayer, setCurrentLayer] = useState(0);
  const [model, setModel] = useState(null);
  const controlsRef = useRef();

  // The model was built in the browser (engine/session.js) on the upload page.
  useEffect(() => { setModel(currentModel()); }, []);

  if (!model) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center text-gray-700">
        <span>No model yet. <button className="underline text-blue-600" onClick={() => navigate('/')}>Upload a G-code file</button></span>
      </div>
    );
  }

  const s = model.summary;
  const solid = s.method === 'solid';
  const layerCount = Math.max(1, s.layerCount);
  const layer = s.layers[currentLayer];
  const layerHeight = s.layerHeight || 0.2;

  const download = () => {
    const link = document.createElement('a');
    link.href = model.url;
    link.download = model.name;
    link.click();
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white shadow-sm border-b">
        <div className="mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <img src={logoImage} alt="KV Logo" className="h-8 w-8" />
              <span className="ml-2 text-xl font-semibold text-gray-900">Print View Portal</span>
            </div>
            <div className="flex items-center">
              <button className="text-sm text-blue-600 hover:text-blue-500" onClick={() => navigate('/')}>← Another file</button>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-[1600px] mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex gap-8">
          <div className="flex-grow">
            <Card className="h-full">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <h2 className="text-xl font-semibold">{model.file}</h2>
                  <ViewerControls controlsRef={controlsRef} />
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  <ModelViewer
                    fileData={model.url}
                    buildDirection="Z"
                    currentLayer={currentLayer}
                    layerHeight={layerHeight}
                    controlsRef={controlsRef}
                  />
                  <div className="space-y-4">
                    <div className="flex justify-between text-sm text-gray-600">
                      <span>Layer {currentLayer + 1} of {layerCount}</span>
                      <span>{layer ? `Z ${num(layer.z, 2)} mm` : `About ${num((currentLayer + 1) * layerHeight, 2)} mm up`}</span>
                    </div>
                    <Slider
                      value={[currentLayer]}
                      onValueChange={v => setCurrentLayer(v[0])}
                      max={layerCount - 1}
                      step={1}
                      className="w-full"
                    />
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="w-80">
            <div className="space-y-6">
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-medium">STL</h3>
                    <button
                      className="bg-green-600 hover:bg-green-700 text-white text-sm px-3 py-2 rounded-md"
                      title="Download the STL"
                      onClick={download}
                    >
                      Download
                    </button>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <Section title="Mesh">
                      <Row label="Closed solid">{s.closed ? 'Yes' : 'No'}</Row>
                      <Row label="Volume">{s.volume === null ? 'not defined (not closed)' : `${num(s.volume / 1000, 2)} cm³`}</Row>
                      <Row label="Size">{num(s.size.x, 2)} × {num(s.size.y, 2)} × {num(s.size.z, 2)} mm</Row>
                      <Row label="Triangles">{s.triangles.toLocaleString()}</Row>
                      <Row label="File">{mb(s.fileBytes)}</Row>
                      {solid && <Row label="Detail">{num(s.cell, 2)} mm, {s.subSlices} per layer</Row>}
                      {solid && <Row label="Inside">{s.filled ? 'filled' : 'as printed'}</Row>}
                    </Section>
                    {solid ? (
                      <Section title="From the G-code">
                        <Row label="Layers">{s.layerCount} × {num(s.layerHeight, 2)} mm</Row>
                        <Row label="Bead width">{num(s.beadWidth, 2)} mm (median)</Row>
                        <Row label="Filament used">{num(s.filament / 1000, 2)} cm³</Row>
                        <Row label="Left out">{Object.keys(s.leftOut).length ? Object.entries(s.leftOut).map(([k, v]) => `${k} (${v})`).join(', ') : (s.keptEverything ? 'nothing (no slicer markers found)' : 'nothing')}</Row>
                      </Section>
                    ) : (
                      <Section title="Hackathon mesh">
                        <p>One 0.4 × 0.2 mm box per move, as built at MakeUC 2024. The boxes overlap and are not one solid, so there is no volume, and purge lines and start code are included.</p>
                        <Row label="Moves">{s.segments.toLocaleString()}</Row>
                      </Section>
                    )}
                  </div>
                </CardContent>
              </Card>

              {layer && (
                <Card>
                  <CardHeader>
                    <h3 className="text-lg font-medium">Layer {currentLayer + 1} <span className="text-sm font-normal text-red-600">(red)</span></h3>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-1 text-sm text-gray-600">
                      <Row label="Top (nozzle) Z">{num(layer.z, 2)} mm</Row>
                      <Row label="Thickness">{num(layer.h, 2)} mm</Row>
                      <Row label="Beads">{layer.beads.toLocaleString()}</Row>
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ViewerPage;
