import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Upload } from 'lucide-react';
import logoImage from './logo.svg';

function UploadPage() {
  const [isDragging, setIsDragging] = useState(false);
  const [file, setFile] = useState(null);
  const [uploadError, setUploadError] = useState(null);
  const navigate = useNavigate();

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
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile?.name.endsWith('.gcode')) {
      setFile(droppedFile);
      setUploadError(null);
    } else {
      setUploadError('Please upload a .gcode file');
    }
  };

  const handleFileSelect = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile?.name.endsWith('.gcode')) {
      setFile(selectedFile);
      setUploadError(null);
    } else {
      setUploadError('Please upload a .gcode file');
    }
  };

  const handleSubmit = async () => {
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:5001/api/upload-gcode', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      
      if (response.ok) {
        navigate('/viewer');  // Immediate navigation on success
      } else {
        throw new Error(data.error || 'Upload failed');
      }

    } catch (error) {
      console.error('Upload error:', error);
      setUploadError(error.message || 'Failed to upload file. Please try again.');
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
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

      <div className="max-w-2xl mx-auto pt-10 px-4">
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
            <p className="text-sm text-gray-500 mt-2">GCode files only</p>
          </div>
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
              className="mt-2 w-full bg-blue-600 text-white rounded-md py-2 px-4 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors"
            >
              Upload and Process
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default UploadPage;