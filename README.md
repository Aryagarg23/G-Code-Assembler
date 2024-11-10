# G-Code Assembler and Viewer

This project allows users to assemble G-code and view it through a web interface and download the result as a .STL file, featuring a Flask backend and a React front-end for visualizing G-code commands. The entire project can be started with a single command.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Alternative Usage with `test.ipynb`](#alternative-usage-with-testipynb)
- [Troubleshooting](#troubleshooting)
- [Notes](#notes)
- [License](#license)

## Prerequisites
1. **Python** (3.x): Ensure Python is installed and added to your PATH. [Download Python here](https://www.python.org/downloads/)
2. **Node.js and npm**: This project requires Node.js and npm for managing front-end dependencies.
   - [Download Node.js and npm](https://nodejs.org/en/download/prebuilt-installer) and ensure they’re added to your PATH.

## Installation and Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/Aryagarg23/G-Code-Assembler.git
   cd G-Code-Assembler
   ```

2. **Run the Project Setup**:
   Execute the following command based on your operating system:

   - **Linux/Mac**:
     ```bash
     ./run_project.sh
     ```
   - **Windows**:
     Double-click `run_project.bat` in the project directory, or open Command Prompt, navigate to the directory, and run:
     ```cmd
     run_project.bat
     ```

   This command will automatically:
   - Install Python dependencies.
   - Start the Flask backend.
   - Install npm dependencies and start the front-end server.

## Usage
After running the setup script, your project’s components should now be running:
- The **Flask backend** serves API endpoints at `http://localhost:5000`.
- The **React front-end** is available at `http://localhost:3000`.

### To view the project:
1. Open a web browser and go to `http://localhost:3000` to see the G-code viewer.
2. The backend Flask server will run concurrently, handling API requests in the background.

## Alternative Usage with `test.ipynb`

If you only need to generate STL files and don’t require the web interface, you can use the Jupyter notebook `test.ipynb` to work with STL files directly:

1. **Install Dependencies**: 
   From the `G-Code-Assembler` directory, install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Jupyter Notebook**:
   Open `test.ipynb` using Jupyter Notebook or Jupyter Lab:
   ```bash
   jupyter notebook test.ipynb
   ```
   
   This notebook allows you to generate STL files without needing to set up or run the front end. Simply run each cell in `test.ipynb` to execute the required operations.

## Troubleshooting
- **Module Installation Issues**:
  - If you encounter an error with `from stl import mesh`, please refer to this issue: [numpy-stl Issue #55](https://github.com/wolph/numpy-stl/issues/55).
- **Node.js Not Found**:
  - If the script indicates that `npm` is missing, install it manually from the link in the Prerequisites and ensure it's added to your PATH.

## Notes
- During the npm installation, you may see warnings about deprecated packages. These are non-critical, and the application should still run without issues.
- For any further customization or additional troubleshooting, please refer to the documentation of [Flask](https://flask.palletsprojects.com/) and [React](https://reactjs.org/).

## License
This project is licensed under the MIT License.
