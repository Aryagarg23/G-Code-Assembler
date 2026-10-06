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
  { id: 'solid', label: 'Solid', note: 'One closed surface, for measuring and simulation. Supports, purge lines and skirts are left out.' },
  { id: 'legacy', label: 'Hackathon mesh', note: 'What the MakeUC 2024 build made: overlapping boxes, not one solid.' },
];

const RESOLUTIONS = [
  { id: 'coarse', label: 'Coarse', note: '0.3 mm' },
  { id: 'medium', label: 'Medium', note: '0.2 mm, slower' },
  { id: 'fine', label: 'Fine', note: '0.1 mm, slow, big files' },
];

// Simplification: merge triangles where the surface moves less than this.
const SIMPLIFY = [
  { value: 0.02, label: '0.02 mm' },
  { value: 0.05, label: '0.05 mm' },
  { value: 0, label: 'Off' },
];

// Choosing a file or a sample converts it straight away with the options
// above it. (It used to only select the file, and the button that started the
// conversion sat out of sight below the options, so a click looked like it
// did nothing.)
function UploadPage() {
  const [isDragging, setIsDragging] = useState(false);
  const [method, setMethod] = useState('solid');
  const [resolution, setResolution] = useState('coarse');
  const [fill, setFill] = useState(true);
  const [tolerance, setTolerance] = useState(0.02);
  const [status, setStatus] = useState(null); // { name, step, progress } while converting
  const [error, setError] = useState(null);
  const navigate = useNavigate();
  const busy = status !== null;

  const convert = async (file) => {
    if (busy) return;
    if (!file?.name.toLowerCase().endsWith('.gcode')) {
      setError('Please choose a .gcode file.');
      return;
    }
    setError(null);
    setStatus({ name: file.name, step: 'Reading the G-code', progress: 0 });
    try {
      await runInWorker(file, { method, resolution, fill, tolerance }, (progress, step) =>
        setStatus(s => ({ ...s, progress, step: step ?? s.step })));
      navigate('/viewer');
    } catch (e) {
      console.error('Processing error:', e);
      setError(`${file.name}: ${e.message || 'could not be converted.'}`);
      setStatus(null);
    }
  };

  const loadSample = async (name) => {
    if (busy) return;
    setError(null);
    setStatus({ name, step: 'Downloading the sample', progress: 0 });
    try {
      const res = await fetch(`${process.env.PUBLIC_URL}/samples/${name}`);
      if (!res.ok) throw new Error(`could not load the sample (HTTP ${res.status})`);
      const file = new File([await res.blob()], name);
      setStatus(null);
      await convert(file);
    } catch (e) {
      setError(`${name}: ${e.message}`);
      setStatus(null);
    }
  };

  const radios = (name, options, value, set) => (
    <div className="flex flex-wrap gap-x-4 gap-y-1">
      {options.map(o => {
        const v = o.id ?? o.value;
        return (
          <label key={String(v)} className="flex gap-1.5 items-center">
            <input type="radio" name={name} checked={value === v} onChange={() => set(v)} disabled={busy} />
            <span className="text-gray-900">{o.label}</span>
            {o.note && <span className="text-gray-500">({o.note})</span>}
          </label>
        );
      })}
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white shadow-sm border-b">
        <div className="mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <img src={logoImage} alt="KV Logo" className="h-8 w-8" />
              <span className="ml-2 text-xl font-semibold text-gray-900">GCode Assembly Portal</span>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-3xl mx-auto pt-8 px-4 pb-16">
        <div className="bg-white border rounded-lg p-4 space-y-3 text-sm">
          <div className="grid grid-cols-[6.5rem_1fr] gap-y-3 items-start">
            <span className="font-medium text-gray-900">Mesh</span>
            {radios('method', METHODS.map(m => ({ ...m, note: undefined })), method, setMethod)}
            {method === 'solid' && (
              <>
                <span className="font-medium text-gray-900">Resolution</span>
                {radios('resolution', RESOLUTIONS, resolution, setResolution)}
                <span className="font-medium text-gray-900">Simplify</span>
                {radios('simplify', SIMPLIFY, tolerance, setTolerance)}
                <span className="font-medium text-gray-900">Inside</span>
                <label className="flex gap-1.5 items-center">
                  <input type="checkbox" checked={fill} onChange={e => setFill(e.target.checked)} disabled={busy} />
                  <span className="text-gray-900">Fill enclosed spaces</span>
                  <span className="text-gray-500">(off: the part as printed, infill gaps and all)</span>
                </label>
              </>
            )}
          </div>
          <p className="text-gray-500">{METHODS.find(m => m.id === method).note}</p>
        </div>

        <div
          className={`mt-4 border-2 border-dashed rounded-lg p-10 text-center transition-colors ${
            isDragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300'
          } ${busy ? 'opacity-60' : ''}`}
          onDragOver={e => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={e => { e.preventDefault(); setIsDragging(false); convert(e.dataTransfer.files[0]); }}
        >
          <Upload className="mx-auto h-10 w-10 text-gray-400" />
          <div className="mt-3">
            <label htmlFor="file-upload" className={busy ? 'cursor-default' : 'cursor-pointer'}>
              <span className="text-blue-600 hover:text-blue-500">Choose a G-code file</span>
              <input
                id="file-upload"
                type="file"
                className="sr-only"
                accept=".gcode"
                disabled={busy}
                onChange={e => { convert(e.target.files[0]); e.target.value = ''; }}
              />
            </label>
            <p className="text-gray-500 mt-1">or drop it here. It converts as soon as you choose it.</p>
            <p className="text-sm text-gray-500 mt-1">Nothing leaves your browser.</p>
          </div>
          <div className="mt-4 text-sm text-gray-600">
            <span>Or convert a file from the challenge: </span>
            {SAMPLES.map((s, i) => (
              <React.Fragment key={s.file}>
                {i > 0 && <span> · </span>}
                <button type="button" disabled={busy} className="text-blue-600 hover:text-blue-500 underline disabled:opacity-50" onClick={() => loadSample(s.file)}>{s.label}</button>
              </React.Fragment>
            ))}
          </div>
        </div>

        <div className="mt-4 min-h-[3.5rem]" aria-live="polite">
          {status && (
            <div className="p-3 bg-blue-50 text-blue-900 rounded-md text-sm">
              <div className="flex justify-between">
                <span>{status.name}: {status.step}…</span>
                <span>{Math.round(status.progress * 100)}%</span>
              </div>
              <div className="mt-2 h-1.5 bg-blue-100 rounded">
                <div className="h-1.5 bg-blue-600 rounded transition-all" style={{ width: `${Math.max(3, status.progress * 100)}%` }} />
              </div>
            </div>
          )}
          {error && <div className="p-3 bg-red-50 text-red-700 rounded-md text-sm" role="alert">{error}</div>}
        </div>
      </div>
    </div>
  );
}

export default UploadPage;
