"""
Illustrative only: not the real backend. The real G-code parser and STL mesher live in
Backend/flask_back.py (GcodeReader / STLAutoAnalyzer) and run on uploaded .gcode files.
This script fakes a small multi-layer toolpath on toy data to show the same two ideas
that matter in the real pipeline: segments are grouped by layer, and each segment has an
extrusion length that maps to print time.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("/home/arya/projects/hackathons/.style/garg-paper.mplstyle")

PALETTE = ["#3b42db", "#c2491d", "#6f2f96", "#77701c"]
FEEDRATE_MM_S = 60.0  # toy constant feedrate, stand-in for the F value in real G-code
LAYER_HEIGHT_MM = 0.2
N_LAYERS = 7


def square_spiral(initial_length=20.0, decrement=1.6, min_length=1.0):
    """turtle-style inward square spiral, one (x, y) toolpath per layer"""
    x, y = 0.0, 0.0
    points = [(x, y)]
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    length = initial_length
    dir_idx = 0
    turns_at_length = 0
    while length > min_length:
        dx, dy = directions[dir_idx % 4]
        x += dx * length
        y += dy * length
        points.append((x, y))
        dir_idx += 1
        turns_at_length += 1
        if turns_at_length == 2:
            length -= decrement
            turns_at_length = 0
    return np.array(points)


def build_layers(n_layers=N_LAYERS):
    """stack the same spiral at increasing z, one toy 'layer' per print layer"""
    base = square_spiral()
    layers = []
    for layer_idx in range(n_layers):
        z = layer_idx * LAYER_HEIGHT_MM
        # slight shrink per layer so the stack reads as a real print, not a repeat
        xy = base * (1.0 - 0.03 * layer_idx)
        layers.append({"z": z, "xy": xy})
    return layers


def layer_extrusion_length(xy):
    deltas = np.diff(xy, axis=0)
    seg_lengths = np.sqrt((deltas ** 2).sum(axis=1))
    return seg_lengths.sum()


def plot_toolpath(layers, outfile):
    fig, ax = plt.subplots(figsize=(6, 6))
    for layer_idx, layer in enumerate(layers):
        color = PALETTE[layer_idx % len(PALETTE)]  # fold back to 4 colors past layer 4
        xy = layer["xy"]
        ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=1.5,
                label=f"layer {layer_idx}", alpha=0.85)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(f"Toy square-spiral toolpath: {len(layers)} layers, colored by layer")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    fig.savefig(outfile, dpi=200)
    plt.close(fig)
    print(f"saved {outfile}")


def plot_extrusion_profile(layers, outfile):
    lengths = np.array([layer_extrusion_length(l["xy"]) for l in layers])
    times = lengths / FEEDRATE_MM_S
    layer_idx = np.arange(len(layers))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(layer_idx, times, color=PALETTE[0])
    ax.set_xlabel("layer index")
    ax.set_ylabel("estimated print time (s)")
    total_s = times.sum()
    ax.set_title(f"Per-layer extrusion time at F{FEEDRATE_MM_S:.0f} mm/s: {total_s:.1f}s total")
    fig.savefig(outfile, dpi=200)
    plt.close(fig)
    print(f"saved {outfile}")


if __name__ == "__main__":
    layers = build_layers()
    plot_toolpath(layers, "prototype/figures/toolpath.png")
    plot_extrusion_profile(layers, "prototype/figures/extrusion_profile.png")
