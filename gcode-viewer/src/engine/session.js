// The current model, kept in memory between the upload and viewer pages.
// This is what the Flask server's current_stl_path global used to hold.

let current = null;

export function runInWorker(file, options, onProgress) {
  return file.text().then(text => new Promise((resolve, reject) => {
    const worker = new Worker(new URL('./worker.js', import.meta.url));
    worker.onmessage = ({ data }) => {
      if (data.progress !== undefined) { onProgress?.(data.progress); return; }
      worker.terminate();
      if (data.error) { reject(new Error(data.error)); return; }
      const { stl, summary } = data.result;
      const url = URL.createObjectURL(new Blob([stl], { type: 'model/stl' }));
      if (current) URL.revokeObjectURL(current.url);
      const base = file.name.replace(/\.gcode$/i, '');
      current = { url, summary, file: file.name, name: `${base}-${summary.method === 'legacy' ? 'hackathon' : 'solid'}.stl` };
      resolve(current);
    };
    worker.onerror = e => { worker.terminate(); reject(new Error(e.message || 'Processing failed')); };
    worker.postMessage({ text, name: file.name, ...options });
  }));
}

export const currentModel = () => current;
