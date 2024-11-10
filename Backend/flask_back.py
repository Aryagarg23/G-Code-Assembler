import collections
from enum import Enum
import math
import os.path
import pprint
import statistics
import sys
import tempfile
import numpy as np
from stl import mesh
from collections import defaultdict
# third party library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# maximum element length in meshing
MAX_ELEMENT_LENGTH = 1 # FDM regular

# set true to add axis-label and title
FIG_INFO = False

# MARGIN RATIO
MARGIN_RATIO = 0.2

class LayerError(Exception):
    """ layer number error """
    pass

class GcodeType(Enum):
    """ enum of GcodeType """
    FDM_REGULAR = 1
    FDM_STRATASYS = 2
    LPBF_REGULAR = 3
    LPBF_SCODE = 4

    @classmethod
    def has_value(cls, value):
        return any(value == item.value for item in cls)

class GcodeReader:
    def __init__(self, filename, filetype=GcodeType.FDM_REGULAR):
        if not os.path.exists(filename):
            print("{} does not exist!".format(filename))
            sys.exit(1)
        self.filename = filename
        self.filetype = filetype
        self.n_segs = 0  # number of line segments
        self.segs = None  # list of line segments [(x0, y0, x1, y1, z, e, segment_id)]
        self.n_layers = 0  # number of layers
        self.seg_index_bars = []
        self.subpath_index_bars = []
        self.summary = None
        self.lengths = None
        self.subpaths = None
        self.xyzlimits = None
        self.current_e = 0  # Track current E position
        self.is_absolute_e = True  # Track if E is absolute or relative
        self._read()

    def _read(self):
        """read the file and populate variables"""
        if self.filetype == GcodeType.FDM_REGULAR:
            self._read_fdm_regular()
        else:
            print("file type is not supported")
            sys.exit(1)
        self.xyzlimits = self._compute_xyzlimits(self.segs)

    def _compute_xyzlimits(self, seg_list):
        """compute axis limits of a segments list"""
        xmin, xmax = float('inf'), -float('inf')
        ymin, ymax = float('inf'), -float('inf')
        zmin, zmax = float('inf'), -float('inf')
        for x0, y0, x1, y1, z, e, _ in seg_list:  # Updated to handle 7 elements
            xmin = min(x0, x1) if min(x0, x1) < xmin else xmin
            ymin = min(y0, y1) if min(y0, y1) < ymin else ymin
            zmin = z if z < zmin else zmin
            xmax = max(x0, x1) if max(x0, x1) > xmax else xmax
            ymax = max(y0, y1) if max(y0, y1) > ymax else ymax
            zmax = z if z > zmax else zmax
        return (xmin, xmax, ymin, ymax, zmin, zmax)

    def _read_fdm_regular(self):
        """read FDM regular gcode type"""
        with open(self.filename, 'r') as infile:
            lines = (line.strip() for line in infile.readlines() if line.strip())
            new_lines = []
            for line in lines:
                if line.startswith(('G', 'M', 'T')):
                    idx = line.find(';')
                    if idx != -1:
                        line = line[:idx]
                    new_lines.append(line)
            lines = new_lines

        self.segs = []
        temp = -float('inf')
        gxyzef = [temp, temp, temp, temp, temp, temp, temp, temp]
        d = {'G': 0, 'X': 1, 'Y': 2, 'Z': 3, 'E': 4, 'F': 5, 'M': 6, 'T': 7}
        seg_count = 0
        mx_z = -math.inf
        last_e = 0
        e_relative_mode = False
        current_segment_id = 0
        last_was_extruding = False
        last_x = None
        last_y = None

        for line in lines:
            old_gxyzef = gxyzef[:]
            
            if "G91" in line:
                e_relative_mode = True
                continue
            elif "G90" in line:
                e_relative_mode = False
                continue
            elif "M83" in line:
                e_relative_mode = True
                continue
            elif "M82" in line:
                e_relative_mode = False
                continue

            for token in line.split():
                command_type = token[0]
                if command_type in d:
                    try:
                        if command_type in ['G', 'X', 'Y', 'Z', 'E', 'F']:
                            gxyzef[d[command_type]] = float(token[1:])
                        elif command_type in ['M', 'T']:
                            gxyzef[d[command_type]] = int(token[1:])
                    except ValueError:
                        print(f"Warning: Could not convert token '{token}'")

            if gxyzef[4] != temp:
                if e_relative_mode:
                    e_movement = gxyzef[4]
                    last_e += e_movement
                else:
                    e_movement = gxyzef[4] - last_e
                    last_e = gxyzef[4]
            else:
                e_movement = 0

            is_extruding = e_movement > 0
            
            # Check for discontinuity
            current_x = gxyzef[1] if gxyzef[1] != temp else last_x
            current_y = gxyzef[2] if gxyzef[2] != temp else last_y
            
            if last_x is not None and last_y is not None:
                is_discontinuous = (abs(current_x - last_x) > 0.01 or 
                                 abs(current_y - last_y) > 0.01)
            else:
                is_discontinuous = False
            
            # Increment segment ID for new extrusion paths or discontinuities
            if ((is_extruding != last_was_extruding and is_extruding) or 
                (is_extruding and is_discontinuous)):
                current_segment_id += 1
            
            last_was_extruding = is_extruding
            last_x = current_x
            last_y = current_y

            if (gxyzef[0] in [0, 1, 2, 3] and gxyzef[1:3] != old_gxyzef[1:3]):
                if gxyzef[3] > mx_z:
                    mx_z = gxyzef[3]
                    self.n_layers += 1
                    self.seg_index_bars.append(seg_count)

                x0, y0, z = old_gxyzef[1:4]
                x1, y1 = gxyzef[1:3]
                self.segs.append((x0, y0, x1, y1, z, is_extruding, current_segment_id))
                seg_count += 1

        self.n_segs = len(self.segs)
        self.segs = np.array(self.segs)
        self.seg_index_bars.append(self.n_segs)
        assert(len(self.seg_index_bars) - self.n_layers == 1)

    def create_axis(self, figsize=(8, 8), projection='2d'):
        """create axis based on figure size and projection"""
        projection = projection.lower()
        if projection not in ['2d', '3d']:
            raise ValueError
        if projection == '2d':
            fig, ax = plt.subplots(figsize=figsize)
        else:  # '3d'
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
        return fig, ax

    def save_figure(self, fig, filename, dpi=200):
        """save figure to a file"""
        _, ext = os.path.splitext(filename)
        ext = ext[1:]  # Remove the dot
        fig.savefig(filename, format=ext, dpi=dpi, bbox_inches='tight')
        print('saving to {:s} with {:d} DPI'.format(filename, dpi))

    def _compute_subpaths(self):
        """compute subpaths with segment IDs"""
        if not self.subpaths:
            self.subpaths = []
            self.subpath_index_bars = [0]
            x0, y0, x1, y1, z, e, seg_id = self.segs[0, :]
            xs, ys, zs, es, seg_ids = [x0, x1], [y0, y1], [z, z], [e, e], [seg_id, seg_id]
            mx_z = zs[-1]
            for x0, y0, x1, y1, z, e, seg_id in self.segs[1:, :]:
                if x0 != xs[-1] or y0 != ys[-1] or z != zs[-1]:
                    self.subpaths.append((xs, ys, zs, es, seg_ids))
                    if z > mx_z:
                        mx_z = z
                        self.subpath_index_bars.append(len(self.subpaths))
                    xs, ys, zs, es, seg_ids = [x0, x1], [y0, y1], [z, z], [e, e], [seg_id, seg_id]
                else:
                    xs.append(x1)
                    ys.append(y1)
                    zs.append(z)
                    es.append(e)
                    seg_ids.append(seg_id)
            if len(xs) != 0:
                self.subpaths.append((xs, ys, zs, es, seg_ids))
            self.subpath_index_bars.append(len(self.subpaths))

    def plot_layers_to_png(self, min_layer, max_layer=None, temp_dir=None):
        """Generate and save PNGs of each layer with segment coloring"""
        if max_layer is None:
            max_layer = self.n_layers + 1
        if (min_layer >= max_layer or min_layer < 1 or max_layer > self.n_layers + 1):
            raise LayerError("Layer number is invalid!")

        if temp_dir is None:
            temp_dir = tempfile.mkdtemp()

        self._compute_subpaths()

        for layer in range(min_layer, max_layer):
            fig, ax = self.create_axis(projection='2d')
            left, right = (self.subpath_index_bars[layer - 1],
                          self.subpath_index_bars[layer])
            
            # Create a colormap for segments
            unique_segments = set()
            for subpath in self.subpaths[left:right]:
                unique_segments.update(subpath[4])  # seg_ids
            colormap = plt.cm.get_cmap('tab20')(np.linspace(0, 1, len(unique_segments)))
            color_dict = dict(zip(sorted(unique_segments), colormap))
            
            for xs, ys, _, es, seg_ids in self.subpaths[left:right]:
                for i in range(len(xs)-1):
                    if es[i]:  # If extruding
                        color = color_dict[seg_ids[i]]
                    else:
                        color = 'lightgray'  # Non-extruding moves
                    ax.plot(xs[i:i+2], ys[i:i+2], color=color, alpha=0.7 if es[i] else 0.3)

            ax.axis('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')

            outfile = os.path.join(temp_dir, f"layer_{layer:03d}.png")
            self.save_figure(fig, outfile, dpi=200)
            plt.close(fig)
        print(f"Saved all layer PNGs to: {temp_dir}")

    def plot(self, ax=None):
        """plot the whole part in 3D with segment coloring"""
        if not ax:
            fig, ax = self.create_axis(projection='3d')
        assert(self.n_segs > 0)
        self._compute_subpaths()
        
        # Create a colormap for segments
        unique_segments = set()
        for subpath in self.subpaths:
            unique_segments.update(subpath[4])  # seg_ids
        colormap = plt.cm.get_cmap('tab20')(np.linspace(0, 1, len(unique_segments)))
        color_dict = dict(zip(sorted(unique_segments), colormap))
        
        for xs, ys, zs, es, seg_ids in self.subpaths:
            for i in range(len(xs)-1):
                if es[i]:  # If extruding
                    color = color_dict[seg_ids[i]]
                else:
                    color = 'lightgray'  # Non-extruding moves
                ax.plot(xs[i:i+2], ys[i:i+2], zs[i:i+2], 
                       color=color, alpha=0.7 if es[i] else 0.3)
        
        xmin, xmax, ymin, ymax, _, _ = self.xyzlimits
        ax.set_xlim(add_margin_to_axis_limits(xmin, xmax))
        ax.set_ylim(add_margin_to_axis_limits(ymin, ymax))
        return fig, ax

    def generate_3d_dataframe(self):
        """Generate a dataframe of all 3D path data with extrusion status and segment IDs"""
        self._compute_subpaths()
        data = []
        for xs, ys, zs, es, seg_ids in self.subpaths:
            for x, y, z, e, seg_id in zip(xs, ys, zs, es, seg_ids):
                data.append({
                    'X': x,
                    'Y': y,
                    'Z': z,
                    'E': bool(e),
                    'segment_id': int(seg_id)
                })
        return pd.DataFrame(data)
    def _create_extrusion_mesh(self, width=0.4, height=0.2):
        """Convert extrusion paths into triangulated mesh data"""
        vertices = []
        faces = []
        vertex_count = 0
        
        # Group segments by layer
        layer_segments = defaultdict(list)
        for x0, y0, x1, y1, z, is_extruding, seg_id in self.segs:
            if is_extruding:  # Only process extrusion moves
                layer_segments[z].append((x0, y0, x1, y1))

        # Process each layer
        for z, segments in layer_segments.items():
            for x0, y0, x1, y1 in segments:
                # Calculate segment direction and perpendicular
                dx = x1 - x0
                dy = y1 - y0
                length = np.sqrt(dx*dx + dy*dy)
                if length < 1e-6:  # Skip tiny segments
                    continue
                    
                # Unit vector perpendicular to segment
                nx = -dy / length * width/2
                ny = dx / length * width/2

                # Create vertices for the segment (rectangular prism)
                segment_vertices = [
                    # Bottom vertices
                    [x0 - nx, y0 - ny, z],
                    [x0 + nx, y0 + ny, z],
                    [x1 + nx, y1 + ny, z],
                    [x1 - nx, y1 - ny, z],
                    # Top vertices
                    [x0 - nx, y0 - ny, z + height],
                    [x0 + nx, y0 + ny, z + height],
                    [x1 + nx, y1 + ny, z + height],
                    [x1 - nx, y1 - ny, z + height],
                ]
                vertices.extend(segment_vertices)

                # Define triangles for this segment using vertex indices
                segment_faces = [
                    # Bottom face
                    [vertex_count + 0, vertex_count + 1, vertex_count + 2],
                    [vertex_count + 0, vertex_count + 2, vertex_count + 3],
                    # Top face
                    [vertex_count + 4, vertex_count + 6, vertex_count + 5],
                    [vertex_count + 4, vertex_count + 7, vertex_count + 6],
                    # Side faces
                    [vertex_count + 0, vertex_count + 4, vertex_count + 1],
                    [vertex_count + 1, vertex_count + 4, vertex_count + 5],
                    [vertex_count + 1, vertex_count + 5, vertex_count + 2],
                    [vertex_count + 2, vertex_count + 5, vertex_count + 6],
                    [vertex_count + 2, vertex_count + 6, vertex_count + 3],
                    [vertex_count + 3, vertex_count + 6, vertex_count + 7],
                    [vertex_count + 3, vertex_count + 7, vertex_count + 0],
                    [vertex_count + 0, vertex_count + 7, vertex_count + 4],
                ]
                faces.extend(segment_faces)
                vertex_count += 8

        return np.array(vertices), np.array(faces)

    def save_to_stl(self, filename, extrusion_width=0.4, layer_height=0.2):
        """Save the G-code paths as an STL file"""
        print("Generating triangulated mesh...")
        
        # Generate vertices and faces for all extrusion paths
        vertices, faces = self._create_extrusion_mesh(width=extrusion_width, height=layer_height)
        
        if len(faces) == 0:
            print("No extrusion paths found to create STL")
            return

        # Create the mesh
        print(f"Creating STL mesh with {len(vertices)} vertices and {len(faces)} faces...")
        
        # Initialize the mesh
        stl_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
        
        # Add faces to mesh
        for i, face in enumerate(faces):
            for j in range(3):
                stl_mesh.vectors[i][j] = vertices[face[j]]
        
        # Save the mesh to file
        print(f"Saving STL to {filename}...")
        stl_mesh.save(filename)
        print("STL file saved successfully.")

    def run_all(self, temp_dir=None, stl_output=None):
        """Run all tasks: plot layers to PNG and optionally save STL"""
        print("Running all tasks...")
        
        #if temp_dir:
            #self.plot_layers_to_png(min_layer=1, max_layer=self.n_layers + 1, temp_dir=temp_dir)
        
        if stl_output:
            self.save_to_stl(stl_output)
            print(f"STL file saved to: {stl_output}")
        
        print("All tasks completed.")

def add_margin_to_axis_limits(min_v, max_v, margin_ratio=MARGIN_RATIO):
    """compute new min_v and max_v based on margin"""
    dv = (max_v - min_v) * margin_ratio
    return (min_v - dv, max_v + dv)

import numpy as np
from stl import mesh
import os

class STLAutoAnalyzer:
    def __init__(self, stl_path):
        """Initialize STL analyzer with file path"""
        if not os.path.exists(stl_path):
            raise FileNotFoundError(f"STL file not found: {stl_path}")
        
        self.stl_path = stl_path
        self.mesh = mesh.Mesh.from_file(stl_path)
        
        # Auto-detect parameters
        self.detected_params = self._detect_printing_parameters()
        
    def _compute_most_common_value(self, values, precision=5, threshold=0.001):
        """Helper function to find most common value in array"""
        if len(values) == 0:
            return None
            
        # Round values to specified precision and find unique values
        rounded = np.round(values, precision)
        unique_vals, counts = np.unique(rounded, return_counts=True)
        
        # Filter out very small values
        mask = unique_vals > threshold
        if not np.any(mask):
            return None
            
        unique_vals = unique_vals[mask]
        counts = counts[mask]
        
        if len(counts) == 0:
            return None
            
        # Return the most common value
        return float(unique_vals[np.argmax(counts)])

    def _detect_printing_parameters(self):
        """Automatically detect printing parameters from the mesh"""
        vectors = self.mesh.vectors
        
        # Get all Z coordinates and sort them
        z_coords = np.unique(vectors[:, :, 2].flatten())
        z_coords = np.sort(z_coords)
        z_diffs = np.diff(z_coords)
        
        # Detect layer height using most common Z difference
        layer_height = self._compute_most_common_value(z_diffs, precision=5, threshold=0.001)
        
        # Detect extrusion width from X-Y plane
        widths = []
        for triangle in vectors:
            # Calculate edge lengths in XY plane
            for i in range(3):
                j = (i + 1) % 3
                dx = triangle[i][0] - triangle[j][0]
                dy = triangle[i][1] - triangle[j][1]
                dz = triangle[i][2] - triangle[j][2]
                
                # Only consider roughly horizontal edges
                if abs(dz) < 0.01:  # Nearly horizontal
                    width = np.sqrt(dx*dx + dy*dy)
                    if width > 0.01:  # Ignore tiny edges
                        widths.append(width)
        
        extrusion_width = self._compute_most_common_value(np.array(widths), precision=3, threshold=0.01)
        
        # Calculate model dimensions
        mins = np.min(vectors, axis=(0, 1))
        maxs = np.max(vectors, axis=(0, 1))
        dimensions = maxs - mins
        
        # Detect build direction based on layer analysis
        z_range = dimensions[2]
        y_range = dimensions[1]
        x_range = dimensions[0]
        
        build_direction = 'Z'  # Default
        if z_range < x_range and z_range < y_range:
            if x_range > y_range:
                build_direction = 'X'
            else:
                build_direction = 'Y'
        
        # Calculate layer height statistics if we found valid layers
        layer_stats = {
            'min': None,
            'max': None,
            'mean': None,
            'std': None
        }
        
        if len(z_diffs) > 0:
            valid_diffs = z_diffs[z_diffs > 0.001]
            if len(valid_diffs) > 0:
                layer_stats.update({
                    'min': float(np.min(valid_diffs)),
                    'max': float(np.max(valid_diffs)),
                    'mean': float(np.mean(valid_diffs)),
                    'std': float(np.std(valid_diffs))
                })
        
        return {
            'layer_height': layer_height,
            'extrusion_width': extrusion_width,
            'build_direction': build_direction,
            'num_layers': len(z_coords),
            'model_height': float(maxs[2] - mins[2]),
            'layer_heights_stats': layer_stats,
            'dimensions': {
                'x': float(x_range),
                'y': float(y_range),
                'z': float(z_range)
            }
        }
    
    def get_basic_stats(self):
        """Get basic statistics about the STL file"""
        # Calculate basic measurements
        try:
            volume, cog, inertia = self.mesh.get_mass_properties()
        except:
            volume = 0
            cog = np.zeros(3)
            
        # Get bounding box
        vectors = self.mesh.vectors
        mins = np.min(vectors, axis=(0, 1))
        maxs = np.max(vectors, axis=(0, 1))
        
        # Calculate surface area
        surface_area = 0
        for triangle in vectors:
            edge1 = triangle[1] - triangle[0]
            edge2 = triangle[2] - triangle[0]
            surface_area += 0.5 * np.linalg.norm(np.cross(edge1, edge2))
        
        return {
            'num_triangles': len(vectors),
            'volume_mm3': float(volume),
            'surface_area_mm2': float(surface_area),
            'dimensions_mm': {
                'x': float(maxs[0] - mins[0]),
                'y': float(maxs[1] - mins[1]),
                'z': float(maxs[2] - mins[2])
            },
            'center_of_gravity_mm': cog.tolist(),
            'bounding_box_mm': {
                'min': mins.tolist(),
                'max': maxs.tolist()
            }
        }
    
    def analyze_layers(self):
        """Analyze the model layer by layer using detected parameters"""
        layer_height = self.detected_params['layer_height']
        if layer_height is None or layer_height <= 0:
            return []
            
        # Group triangles by layer
        vectors = self.mesh.vectors
        layer_map = {}
        
        for triangle in vectors:
            z_mean = np.mean(triangle[:, 2])
            layer_idx = round(z_mean / layer_height)
            if layer_idx not in layer_map:
                layer_map[layer_idx] = []
            layer_map[layer_idx].append(triangle)
        
        layer_stats = []
        for layer_idx in sorted(layer_map.keys()):
            layer_triangles = np.array(layer_map[layer_idx])
            
            # Calculate layer area
            layer_area = 0
            for triangle in layer_triangles:
                edge1 = triangle[1] - triangle[0]
                edge2 = triangle[2] - triangle[0]
                layer_area += 0.5 * np.linalg.norm(np.cross(edge1, edge2))
            
            # Calculate layer bounds
            mins = np.min(layer_triangles, axis=(0, 1))
            maxs = np.max(layer_triangles, axis=(0, 1))
            
            layer_stats.append({
                'layer_index': int(layer_idx),
                'z_height': float(layer_idx * layer_height),
                'num_triangles': len(layer_triangles),
                'area_mm2': float(layer_area),
                'bounds_mm': {
                    'min': mins.tolist(),
                    'max': maxs.tolist()
                }
            })
        
        return layer_stats
    
    def check_mesh_quality(self):
        """Check mesh quality metrics"""
        vectors = self.mesh.vectors
        
        # Auto-detect thresholds based on model size
        dims = np.max(vectors, axis=(0, 1)) - np.min(vectors, axis=(0, 1))
        model_size = np.sqrt(np.sum(dims**2))
        min_area_threshold = (model_size * 0.0001)**2  # Scale with model size
        
        quality_stats = {
            'degenerate_triangles': 0,
            'small_triangles': 0,
            'edge_stats': {
                'min_length': float('inf'),
                'max_length': 0,
                'mean_length': 0,
                'total_edges': 0
            }
        }
        
        all_edge_lengths = []
        
        for triangle in vectors:
            # Calculate edges
            edges = np.array([
                triangle[1] - triangle[0],
                triangle[2] - triangle[1],
                triangle[0] - triangle[2]
            ])
            
            # Calculate edge lengths
            edge_lengths = np.sqrt(np.sum(edges**2, axis=1))
            all_edge_lengths.extend(edge_lengths)
            
            # Calculate area
            area = 0.5 * np.linalg.norm(np.cross(edges[0], edges[1]))
            
            # Check for degenerate and small triangles
            if area < 1e-10:
                quality_stats['degenerate_triangles'] += 1
            elif area < min_area_threshold:
                quality_stats['small_triangles'] += 1
        
        # Update edge statistics
        all_edge_lengths = np.array(all_edge_lengths)
        quality_stats['edge_stats'].update({
            'min_length': float(np.min(all_edge_lengths)),
            'max_length': float(np.max(all_edge_lengths)),
            'mean_length': float(np.mean(all_edge_lengths)),
            'std_length': float(np.std(all_edge_lengths)),
            'total_edges': len(all_edge_lengths)
        })
        
        return quality_stats

    def generate_report(self):
        """Generate a comprehensive report of the STL file"""
        report = {
            'file_info': {
                'path': self.stl_path,
                'size_bytes': os.path.getsize(self.stl_path)
            },
            'detected_parameters': self.detected_params,
            'model_stats': self.get_basic_stats(),
            'layer_analysis': self.analyze_layers(),
            'quality_metrics': self.check_mesh_quality()
        }
        return report

def print_stl_analysis(stl_path):
    """Print formatted analysis of an STL file"""
    analyzer = STLAutoAnalyzer(stl_path)
    report = analyzer.generate_report()
    
    print("\n=== STL File Analysis ===")
    print(f"\nFile: {report['file_info']['path']}")
    print(f"File Size: {report['file_info']['size_bytes'] / 1024:.1f} KB")
    
    print("\nDetected Parameters:")
    params = report['detected_parameters']
    if params['layer_height']:
        print(f"- Layer Height: {params['layer_height']:.3f} mm")
    else:
        print("- Layer Height: Not detected")
    
    if params['extrusion_width']:
        print(f"- Extrusion Width: {params['extrusion_width']:.3f} mm")
    else:
        print("- Extrusion Width: Not detected")
        
    print(f"- Build Direction: {params['build_direction']}")
    print(f"- Number of Layers: {params['num_layers']}")
    
    print("\nModel Statistics:")
    stats = report['model_stats']
    print(f"- Number of Triangles: {stats['num_triangles']:,}")
    print(f"- Volume: {stats['volume_mm3']:.2f} mm³")
    print(f"- Surface Area: {stats['surface_area_mm2']:.2f} mm²")
    print(f"- Dimensions (mm): X={stats['dimensions_mm']['x']:.2f}, "
          f"Y={stats['dimensions_mm']['y']:.2f}, "
          f"Z={stats['dimensions_mm']['z']:.2f}")
    
    print("\nQuality Metrics:")
    quality = report['quality_metrics']
    print(f"- Degenerate Triangles: {quality['degenerate_triangles']}")
    print(f"- Small Triangles: {quality['small_triangles']}")
    print(f"- Edge Length Statistics (mm):")
    print(f"  - Min: {quality['edge_stats']['min_length']:.3f}")
    print(f"  - Max: {quality['edge_stats']['max_length']:.3f}")
    print(f"  - Mean: {quality['edge_stats']['mean_length']:.3f}")
    
    return report

from flask import Flask, request, jsonify, send_file
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configure directories
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads', 'gcode')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'output')
LAYER_IMAGES_FOLDER = os.path.join(BASE_DIR, 'Layer_images')

# Create necessary directories
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, LAYER_IMAGES_FOLDER]:
    os.makedirs(folder, exist_ok=True)

ALLOWED_EXTENSIONS = {'gcode'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 * 1024

# Global variables
current_file_path = None
current_stl_path = None
@app.route('/api/model-data', methods=['GET'])
def get_model_data():
    global current_stl_path
    
    if not current_stl_path or not os.path.exists(current_stl_path):
        return jsonify({'error': 'No processed model available'}), 404

    try:
        # Analyze the STL file
        analyzer = STLAutoAnalyzer(current_stl_path)
        report = analyzer.generate_report()
        
        # Format the data for the frontend
        model_data = {
            'file_info': {
                'path': report['file_info']['path'],
                'size_bytes': report['file_info']['size_bytes']
            },
            'detected_parameters': {
                'layer_height': report['detected_parameters']['layer_height'],
                'extrusion_width': report['detected_parameters']['extrusion_width'],
                'build_direction': report['detected_parameters']['build_direction'],
                'num_layers': report['detected_parameters']['num_layers'],
                'model_height': report['detected_parameters']['model_height'],
                'layer_heights_stats': report['detected_parameters']['layer_heights_stats'],
                'dimensions': report['detected_parameters']['dimensions']
            },
            'model_stats': {
                'num_triangles': report['model_stats']['num_triangles'],
                'volume_mm3': report['model_stats']['volume_mm3'],
                'surface_area_mm2': report['model_stats']['surface_area_mm2'],
                'dimensions_mm': report['model_stats']['dimensions_mm'],
                'center_of_gravity_mm': report['model_stats']['center_of_gravity_mm'],
                'bounding_box_mm': report['model_stats']['bounding_box_mm']
            },
            'quality_metrics': {
                'degenerate_triangles': report['quality_metrics']['degenerate_triangles'],
                'small_triangles': report['quality_metrics']['small_triangles'],
                'edge_stats': {
                    'min_length': report['quality_metrics']['edge_stats']['min_length'],
                    'max_length': report['quality_metrics']['edge_stats']['max_length'],
                    'mean_length': report['quality_metrics']['edge_stats']['mean_length'],
                    'total_edges': report['quality_metrics']['edge_stats']['total_edges'],
                    'std_length': report['quality_metrics']['edge_stats']['std_length']
                }
            },
            'layer_analysis': report['layer_analysis']
        }
        
        return jsonify(model_data)
    except Exception as e:
        print(f"Error analyzing STL: {str(e)}")
        return jsonify({'error': f'Error analyzing model: {str(e)}'}), 500
    
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_gcode_file(gcode_path):
    try:
        # Create GcodeReader instance
        gcode_reader = GcodeReader(
            filename=gcode_path,
            filetype=GcodeType.FDM_REGULAR
        )

        # Generate STL file name based on input file
        stl_filename = os.path.splitext(os.path.basename(gcode_path))[0] + '.stl'
        stl_path = os.path.join(OUTPUT_FOLDER, stl_filename)

        # Run analysis and save STL
        gcode_reader.run_all(
            temp_dir=LAYER_IMAGES_FOLDER,
            stl_output=stl_path
        )

        return stl_path
    except Exception as e:
        print(f"Error processing GCODE: {str(e)}")
        raise

@app.route('/api/upload-gcode', methods=['POST'])
def upload_gcode():
    global current_file_path, current_stl_path
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only .gcode files are allowed'}), 400
    
    try:
        # Save the GCODE file
        filename = file.filename
        gcode_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(gcode_path)
        current_file_path = gcode_path
        
        print(f"GCODE file saved at: {current_file_path}")

        # Process the GCODE and generate STL
        stl_path = process_gcode_file(gcode_path)
        current_stl_path = stl_path
        
        print(f"STL file generated at: {current_stl_path}")
        
        return jsonify({
            'message': 'File uploaded and processed successfully',
            'gcode_filename': filename,
            'gcode_path': gcode_path,
            'stl_path': stl_path
        }), 200
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': f'Upload/processing failed: {str(e)}'}), 500

@app.route('/api/stl-file', methods=['GET'])
def get_stl_file():
    global current_stl_path
    
    try:
        if not current_stl_path:
            return jsonify({'error': 'No STL file has been generated yet'}), 404
            
        if not os.path.exists(current_stl_path):
            return jsonify({'error': f'STL file not found at path: {current_stl_path}'}), 404
            
        return send_file(
            current_stl_path,
            mimetype='application/sla',
            as_attachment=True,
            download_name=os.path.basename(current_stl_path)
        )
    except Exception as e:
        return jsonify({'error': f'Error retrieving STL file: {str(e)}'}), 500


if __name__ == '__main__':
    print(f"Server starting...")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"Layer images folder: {LAYER_IMAGES_FOLDER}")
    app.run(debug=True, port=5001)