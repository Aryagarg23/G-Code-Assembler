import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Upload } from 'lucide-react';
import logoImage from './logo.svg';
import { runInWorker } from './engine/session';

// The challenge's sample files, served next to the app.
const SAMPLES = [
  { file: 'SquarePrism.gcode', label: 'Square prism' },
  { file: 'KV_Monogram.gcode', label: 'KV monogram' },
  { file: '3DBenchy.gcode', label: '3DBenchy' },
];

const METHODS = [
  { id: 'solid', label: 'Solid', note: 'One closed surface. Each bead is a rounded stadium as wide as the filament pushed out. Supports, purge lines and skirts are left out.' },
  { id: 'legacy', label: 'Hackathon mesh', note: 'What the MakeUC 2024 build made: a 0.4 x 0.2 mm box per move, overlapping, not one solid.' },
];

const RESOLUTIONS = [
  { id: 'coarse', label: 'Coarse', note: '0.3 mm cells, 1 sample per layer' },
  { id: 'medium', label: 'Medium', note: '0.2 mm cells, 2 samples per layer' },
  { id: 'fine', label: 'Fine', note: '0.1 mm cells, 4 samples per layer (slow, big files)' },
];

// Simplification: merge triangles where the surface moves less than this.
const SIMPLIFY = [
  { value: 0.02, label: '0.02 mm', note: 'About 4 to 7 times fewer triangles' },
  { value: 0.05, label: '0.05 mm', note: 'Fewer still; flat walls go to a few triangles' },
  { value: 0, label: 'Off', note: 'Every triangle from the grid' },
];

function UploadPage() {
  const [isDragging, setIsDragging] = useState(false);
  const [file, setFile] = useState(null);
  const [uploadError, setUploadError] = useState(null);
  const [method, setMethod] = useState('solid');
  const [resolution, setResolution] = useState('coarse');
  const [fill, setFill] = useState(true);
  const [tolerance, setTolerance] = useState(0.02);
  const [progress, setProgress] = useState(null);
  const navigate = useNavigate();

  const accept = (f) => {
    if (f?.name.toLowerCase().endsWith('.gcode')) {
      setFile(f);
      setUploadError(null);
    } else {
      setUploadError('Please upload a .gcode file');
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    accept(e.dataTransfer.files[0]);
  };

  const handleFileSelect = (e) => accept(e.target.files[0]);

  const loadSample = async (name) => {
    setUploadError(null);
    try {
      const res = await fetch(`${process.env.PUBLIC_URL}/samples/${name}`);
      if (!res.ok) throw new Error(`Could not load ${name}`);
      setFile(new File([await res.blob()], name));
    } catch (error) {
      setUploadError(error.message);
    }
  };

  const handleSubmit = async () => {
    if (!file) return;
    setUploadError(null);
    setProgress(0);
    try {
      await runInWorker(file, { method, resolution, fill, tolerance }, setProgress);
      navigate('/viewer');
    } catch (error) {
      console.error('Processing error:', error);
      setUploadError(error.message || 'Failed to process the file. Please try again.');
    } finally {
      setProgress(null);
    }
  };

  const busy = progress !== null;

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
              <span className="ml-2 text-xl font-semibold text-gray-900">GCode Assembly Portal</span>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-2xl mx-auto pt-10 px-4 pb-16">
        <div
          className={`mt-8 border-2 border-dashed rounded-lg p-12 text-center transition-colors ${
            isDragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300'
          }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <Upload className="mx-auto h-12 w-12 text-gray-400" />
          <div className="mt-4">
            <label htmlFor="file-upload" className="cursor-pointer">
              <span className="text-blue-600 hover:text-blue-500">Upload a file</span>
              <input
                id="file-upload"
                type="file"
                className="sr-only"
                accept=".gcode"
                onChange={handleFileSelect}
              />
            </label>
            <p className="text-gray-500 mt-1">or drag and drop</p>
            <p className="text-sm text-gray-500 mt-2">GCode files only. Nothing leaves your browser.</p>
          </div>
        </div>

        <div className="mt-4 text-sm text-gray-600">
          <span>Or try a file from the challenge: </span>
          {SAMPLES.map((s, i) => (
            <React.Fragment key={s.file}>
              {i > 0 && <span> · </span>}
              <button type="button" className="text-blue-600 hover:text-blue-500 underline" onClick={() => loadSample(s.file)}>{s.label}</button>
            </React.Fragment>
          ))}
        </div>

        <div className="mt-6 bg-white border rounded-lg p-4 space-y-4 text-sm">
          <fieldset>
            <legend className="font-medium text-gray-900 mb-2">Mesh</legend>
            {METHODS.map(m => (
              <label key={m.id} className="flex gap-2 items-start mb-2">
                <input type="radio" name="method" value={m.id} checked={method === m.id} onChange={() => setMethod(m.id)} className="mt-1" />
                <span><span className="text-gray-900">{m.label}</span><span className="block text-gray-500">{m.note}</span></span>
              </label>
            ))}
          </fieldset>
          {method === 'solid' && (
            <>
              <fieldset>
                <legend className="font-medium text-gray-900 mb-2">Resolution</legend>
                <div className="flex flex-wrap gap-4">
                  {RESOLUTIONS.map(r => (
                    <label key={r.id} className="flex gap-2 items-start">
                      <input type="radio" name="resolution" value={r.id} checked={resolution === r.id} onChange={() => setResolution(r.id)} className="mt-1" />
                      <span><span className="text-gray-900">{r.label}</span><span className="block text-gray-500">{r.note}</span></span>
                    </label>
                  ))}
                </div>
              </fieldset>
              <label className="flex gap-2 items-start">
                <input type="checkbox" checked={fill} onChange={e => setFill(e.target.checked)} className="mt-1" />
                <span><span className="text-gray-900">Fill enclosed spaces</span><span className="block text-gray-500">Infill pockets become solid, so the mesh is the part's outer boundary. Off: the part as printed, infill gaps and all.</span></span>
              </label>
              <fieldset>
                <legend className="font-medium text-gray-900 mb-2">Simplify</legend>
                <div className="flex flex-wrap gap-4">
                  {SIMPLIFY.map(o => (
                    <label key={o.value} className="flex gap-2 items-start">
                      <input type="radio" name="simplify" checked={tolerance === o.value} onChange={() => setTolerance(o.value)} className="mt-1" />
                      <span><span className="text-gray-900">{o.label}</span><span className="block text-gray-500">{o.note}</span></span>
                    </label>
                  ))}
                </div>
              </fieldset>
            </>
          )}
        </div>

        {uploadError && (
          <div className="mt-4 p-3 bg-red-50 text-red-700 rounded-md text-sm">
            {uploadError}
          </div>
        )}

        {file && (
          <div className="mt-4">
            <p className="text-sm text-gray-600">Selected file: {file.name}</p>
            <button
              onClick={handleSubmit}
              disabled={busy}
              className="mt-2 w-full bg-blue-600 text-white rounded-md py-2 px-4 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors disabled:opacity-60"
            >
              {busy ? `Processing… ${Math.round(progress * 100)}%` : 'Upload and Process'}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default UploadPage;
