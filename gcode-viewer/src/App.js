import React from 'react';
import { MemoryRouter as Router, Routes, Route } from 'react-router-dom';
import UploadPage from './UploadPage';
import ViewerPage from './ViewerPage';

// MemoryRouter: the app runs in the browser with no server and can sit under
// any path (aryagarg23.com serves it at /embeds/gcode-assembler/), so the two
// pages switch without changing the URL.
function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<UploadPage />} />
        <Route path="/viewer" element={<ViewerPage />} />
      </Routes>
    </Router>
  );
}

export default App;
