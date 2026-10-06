/* eslint-disable no-restricted-globals */
// Runs the pipeline off the main thread so the page stays responsive while a
// big file is meshed. Messages: { text, name, method, resolution, fill } in;
// { progress } updates, then { result } or { error } out.
import { processGcode } from './pipeline.mjs';

self.onmessage = ({ data }) => {
  try {
    const result = processGcode(data.text, { ...data, onProgress: f => self.postMessage({ progress: f }) });
    self.postMessage({ result }, [result.stl]);
  } catch (e) {
    self.postMessage({ error: String(e?.message ?? e) });
  }
};
