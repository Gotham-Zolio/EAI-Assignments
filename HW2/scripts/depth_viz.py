# Embodied AI 2025 Fall – HW2
# Part 2.2 – Sensors: Depth Visualization (ImageIO)
# ---------------------------------------------------------------
# Loads depth frames from .npz, maps them to color using a consistent
# normalization and colormap, and exports an MP4 animation.
# ---------------------------------------------------------------

import dataclasses
import pathlib
import numpy as np
import tyro
import imageio
import matplotlib


@dataclasses.dataclass
class Args:
    """Command-line arguments for depth visualization."""
    input: pathlib.Path
    output: pathlib.Path
    fps: int = 60
    colormap: str = "plasma"  # options: jet, plasma, viridis, inferno
    normalize_min: float | None = None
    normalize_max: float | None = None


def main(args: Args):
    # ============================================================
    # Load Data
    # ============================================================
    data = np.load(args.input, allow_pickle=True)
    if "depth_image_data" not in data:
        raise ValueError("Missing 'depth_image_data' in input file.")
    frames = np.array(data["depth_image_data"], dtype=object)
    frames = [np.asarray(f, dtype=np.float32) for f in frames if f is not None]
    if not frames:
        raise ValueError("No valid depth frames found.")

    print(f"Loaded {len(frames)} frames from {args.input}")

    # ============================================================
    # Normalization
    # ============================================================
    vmin = args.normalize_min or min(np.nanmin(f) for f in frames)
    vmax = args.normalize_max or max(np.nanmax(f) for f in frames)
    print(f"Depth range: [{vmin:.3f}, {vmax:.3f}]")

    # ============================================================
    # Colormap Setup (modern API)
    # ============================================================
    cmap = matplotlib.colormaps.get_cmap(args.colormap)

    # ============================================================
    # Render and Write Video
    # ============================================================
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(args.output, fps=args.fps, codec="libx264") as writer:
        for f in frames:
            norm = np.clip((f - vmin) / (vmax - vmin), 0, 1)
            rgb = (cmap(norm)[..., :3] * 255).astype(np.uint8)
            writer.append_data(rgb)

    print(f"Saved animation to {args.output}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
