# Embodied AI — Course Assignments (EAI-Assignments)

This repository collects course assignments, code, data pointers, and documentation for the "Embodied Artificial Intelligence" course. It is organized by assignment and will continue to grow as new homeworks are added. The goal is to provide reproducible experiments, clean code, and clear write-ups for each assignment.

---

## Repository layout (top-level)

- `HW1-3DGS/` — Homework 1: 3D Gaussian Splatting implementation, rendering harness, report and assets.
  - `report/` — LaTeX report and images.
  - `renders/` — Output renders (example results and placeholders).
  - `submodules/` — Third-party code (e.g., CUDA rasterizer). Built artifacts may appear here after build.
  - `assets/` — Data assets used for HW1 (e.g., `points3D.ply`, camera files).
  - `scripts/` — Helper scripts (ignored by git; not tracked). Used for local reproducibility; do not commit secrets here.
  - `render.py`, `gaussian_model.py`, `utils/`, etc. — Core code used to run the pipeline.

- `requirements.txt` — Python dependencies used across assignments (try to keep these minimal and pinned per-assignment where necessary).

- (future) `HW2-.../`, `HW3-.../` — Additional homework folders will follow the same structure.

---

## High-level goals and conventions

- Each homework lives in its own top-level folder `HWn-Name/`.
- Include a `README.md` inside each homework folder describing the assignment-specific steps, expected outputs, and dataset pointers.
- Keep heavy data out of the git history when possible. Large files should be stored externally (e.g., via Google Drive, institutional storage) and referenced in the homework README with download instructions.
- Use reproducible commands and document the environment (OS, Python, CUDA versions) in each homework's README.

---

## Quick start (Windows, PowerShell)

These steps show how to set up and run the HW1 example locally on a Windows machine with PowerShell (adapt commands for Linux/macOS if needed).

1. Create and activate a Python virtual environment (recommended):

```powershell
# from repository root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

2. Build the CUDA rasterizer extension (if you have CUDA and nvcc installed):

```powershell
cd HW1-3DGS\submodules\diff-gaussian-rasterization
python setup.py build
python setup.py install
cd ..\..\
```

3. Quick functional test (from repo root):

```powershell
python HW1-3DGS\render.py --model_path HW1-3DGS\assets\gs_cloud.ply --sh_degree 1
```

4. If you want to skip the COLMAP reconstruction (heavy), use the provided `points3D.ply` in `HW1-3DGS/assets/` as the input for initialization.

---

## HW1-specific notes (3D Gaussian Splatting)

- Report: `HW1-3DGS/report/report.tex` and generated PDF. The report contains derivations, implementation notes, and an experiment protocol.
- Key implementation files:
  - `HW1-3DGS/gaussian_model.py` — Gaussian parameter handling (scales, quaternions, get_covariance, pruning).
  - `HW1-3DGS/utils/sh_utils.py` — Spherical harmonics evaluation (SH degree 0/1).
  - `HW1-3DGS/render.py` — Rendering harness that loads the model, computes covariances, and calls the rasterizer.
  - `HW1-3DGS/submodules/diff-gaussian-rasterization/` — CUDA extension source (rasterization kernel).

- Typical workflow:
  1. (Optional) Run COLMAP to reconstruct `points3D.ply` (heavy; can be skipped if `points3D.ply` is provided).
  2. Build the rasterizer.
  3. Run `render.py` with desired options (SH degree, prune mode, thresholds).
  4. Save renders into `HW1-3DGS/renders/` and compute metrics (PSNR/SSIM) against the provided ground truth if available.

- Reproducibility: record the git commit hash used for experiments and the exact command-line parameters in the report.

---

## Reproducing COLMAP run (if you choose to)

COLMAP can be expensive. If you want to run the automatic reconstructor to regenerate `points3D.ply` for HW1, run from PowerShell (example):

```powershell
$DATASET = "$(pwd)\HW1-3DGS\datasets\fruit"
colmap automatic_reconstructor --workspace_path $DATASET --image_path "$DATASET\images" 2>&1 | Tee-Object -FilePath "$DATASET\colmap_autorecon.log"
```

Notes:
- Watch `colmap_autorecon.log` for per-stage timings (feature extraction, matching, BA). Matching and BA are typically the most time-consuming.
- If you don't have CUDA-enabled feature extraction in COLMAP, CPU runs will be slower.

---

## Adding future homeworks (suggested template)

When adding `HWn-Name/` for a new assignment, follow this checklist:

- Add folder `HWn-Name/` with subfolders:
  - `code/` or root scripts for the assignment
  - `assets/` (small required files, keep large files external)
  - `report/` (LaTeX or markdown report)
  - `renders/` (output images)
  - `README.md` (assignment-specific instructions and quick commands)

- Provide a reproducible `requirements- HWn.txt` if the assignment needs special packages.
- Provide a short `run.sh` / `run.ps1` script to execute the main experiment with documented defaults.

---

## Common troubleshooting

- "ImportError" for the rasterizer on Windows: ensure the compiled extension matches your Python and CUDA versions. Rebuild the extension after activating the same venv that you run Python with.
- LaTeX fails to compile the report: ensure system fonts (or TeX packages) used by the report are available and escape underscores with `\texttt{...}` or verbatim blocks.
- COLMAP failures: check image naming, supported image formats, and sufficient features; review `colmap_autorecon.log` to find failing images.

---

## Tests and CI

We recommend adding minimal unit tests for critical numerical code (e.g., covariance symmetry, small SH checks). A lightweight `pytest` setup in each homework folder is suggested. If you'd like, I can add example tests for HW1.

---

## Contribution and style

- Keep code readable and documented. Use type hints where helpful.
- For new assignments, open a branch `hwN-yourname` and make a clear PR describing functionality and experiments.
- Avoid committing large binary data. Use `.gitignore` to exclude temporary files and `scripts/`.

---

## License & Acknowledgements

- This repository is for course use. If you plan to publish code derived from this repo, check with course policies and acknowledge appropriate sources (COLMAP, diff-gaussian-rasterization authors, etc.).

---

If you'd like, I can:
- Add a `README.md` inside `HW1-3DGS/` with an abbreviated quick-run section (I see the folder already has instructions; I can standardize it).
- Create the suggested `pytest` unit tests for `get_covariance` and `eval_sh`.
- Add a small `run.ps1` script that performs the build + baseline render and stores outputs in `renders/`.