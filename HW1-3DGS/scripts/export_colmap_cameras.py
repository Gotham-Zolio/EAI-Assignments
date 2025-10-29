#!/usr/bin/env python3
"""
Export cameras from a COLMAP sparse model directory (e.g. sparse/0) to
individual NumPy .npy files compatible with the project's `render.py`.

Usage:
    python scripts/export_colmap_cameras.py /path/to/sparse/0 /path/to/output_dir

Dependencies:
    pip install pycolmap numpy

The script writes one .npy file per image containing a dict with keys:
    R (3x3 float32), T (3,), FoVx (float), FoVy (float), image_width (int), image_height (int)

"""
import sys
import os
import numpy as np

try:
    import pycolmap
except Exception as e:
    print("pycolmap is required. Install with: pip install pycolmap")
    raise


def qvec_to_rotmat(qvec):
    # pycolmap provides qvec2rotmat, but in case it's not available, implement fallback
    try:
        return pycolmap.utils.qvec2rotmat(qvec)
    except Exception:
        w, x, y, z = qvec
        R = np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
            [2*(x*y + w*z),     1 - 2*(x*x + z*z),     2*(y*z - w*x)],
            [2*(x*z - w*y),         2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ], dtype=np.float32)
        return R


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/export_colmap_cameras.py /path/to/sparse/0 /path/to/output_dir")
        sys.exit(1)

    model_path = sys.argv[1]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading COLMAP model from {model_path}")
    model = pycolmap.Reconstruction(model_path)

    for image_id, image in model.images.items():
        qvec = image.qvec
        tvec = image.tvec
        R = qvec_to_rotmat(qvec).astype(np.float32)
        T = np.array(tvec, dtype=np.float32)

        cam = model.cameras[image.camera_id]
        width = cam.width
        height = cam.height

        # try to get fx, fy from known camera models
        fx = None
        fy = None
        if cam.model.startswith('SIMPLE_PINHOLE'):
            fx = cam.params[0]
            fy = fx
        elif cam.model.startswith('PINHOLE'):
            fx = cam.params[0]
            fy = cam.params[1]
        elif cam.model.startswith('SIMPLE_RADIAL') or cam.model.startswith('RADIAL'):
            fx = cam.params[0]
            fy = fx
        else:
            # fallback: approximate using width and 60 deg FOV
            fx = cam.params[0] if len(cam.params) > 0 else max(width, height) / (2.0 * np.tan(np.deg2rad(60.0)/2.0))
            fy = fx

        FoVx = 2.0 * np.arctan(width / (2.0 * fx))
        FoVy = 2.0 * np.arctan(height / (2.0 * fy))

        cam_dict = {
            "R": R,
            "T": T,
            "FoVx": float(FoVx),
            "FoVy": float(FoVy),
            "image_width": int(width),
            "image_height": int(height)
        }

        out_path = os.path.join(out_dir, f"{image.name}.npy")
        np.save(out_path, cam_dict)
        print("Wrote", out_path)

    print("Done exporting cameras.")


if __name__ == '__main__':
    main()
