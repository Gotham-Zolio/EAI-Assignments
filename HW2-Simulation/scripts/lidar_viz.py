import dataclasses
import tyro
import numpy as np
import matplotlib.pyplot as plt
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg


@dataclasses.dataclass
class Args:
    """Visualization arguments."""
    input: str
    output: str
    fps: int = 60
    elev: int = 20
    azim_speed: float = 0.5  # deg/frame


def main(args: Args):
    # ============================================================
    # Load Data
    # ============================================================
    data = np.load(args.input, allow_pickle=True)
    lidar_frames = data["lidar_data"]

    # ============================================================
    # Setup Figure
    # ============================================================
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_zlim(0, 3)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("LiDAR Point Cloud Animation")

    scatter = ax.scatter([], [], [], s=1, c="tab:blue")
    
    # Use a canvas that doesn't require a GUI
    canvas = FigureCanvasAgg(fig)

    print(f"Rendering {len(lidar_frames)} frames to {args.output}...")

    # ============================================================
    # Render Loop
    # ============================================================
    # Use imageio to write video, avoiding system ffmpeg dependency issues with matplotlib
    with imageio.get_writer(args.output, fps=args.fps, codec="libx264") as writer:
        for i, points in enumerate(lidar_frames):
            ax.clear()
            ax.set_xlim(-8, 8)
            ax.set_ylim(-8, 8)
            ax.set_zlim(0, 3)
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_zlabel("Z (m)")
            ax.set_title("LiDAR Point Cloud Animation")

            if points is not None and points.size > 0:
                # Ensure points are float and valid
                pts = np.asarray(points, dtype=np.float32)
                if pts.ndim == 2 and pts.shape[1] == 3:
                    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, c="tab:blue")

            ax.view_init(elev=args.elev, azim=i * args.azim_speed)
            
            # Draw the frame
            canvas.draw()
            
            # Convert to image
            s, (width, height) = canvas.print_to_buffer()
            image = np.frombuffer(s, np.uint8).reshape((height, width, 4))
            
            # Write RGB (remove alpha channel)
            writer.append_data(image[:, :, :3])
            
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(lidar_frames)} frames")

    print(f"Saved LiDAR animation to {args.output}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
