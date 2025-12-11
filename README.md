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

- `HW2-Simulation/` — Homework 2: Robotics Simulation Workflow. Simulation, sensor integration, RL training, and maze navigation.
  - `report/` — LaTeX report and images.
  - `assets/` — Robot models, maze layouts, and sensor data.
  - `part1/` — Hello World scene construction (table, robots, objects, camera).
  - `part2/` — Sensor integration (IMU, LiDAR, Depth Camera) and visualization scripts.
  - `part3/` — RL locomotion training, evaluation, and maze navigation demo.
  - `scripts/` — Visualization and helper scripts (e.g., depth_viz.py, lidar_viz.py).
  - `urdf/` — URDF files for all robots used in simulation.
  - `go2_env.py`, `go2_train.py`, `go2_eval.py`, `go2_maze.py`, `go2_maze_run.py` — Core code for RL and navigation tasks.

- `HW3-MBRL/` — Homework 3: Model-Based Reinforcement Learning (MBRL) for continuous control.
  - `src/` — Source code for MBRL algorithms, environment wrappers, and training scripts.
  - `logs/`, `outputs/`, `runs/` — Training logs, evaluation outputs, and experiment runs (ignored by git).
  - `EAI_hw3.pdf` — Assignment report.
  - `cfg.yaml` — Configuration file for experiments.
  - `requirements.txt` — Python dependencies for HW3.

- `requirements.txt` — Python dependencies used across assignments (try to keep these minimal and pinned per-assignment where necessary).

---

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


## HW2-specific notes (Robotics Simulation Workflow)

- Report: `HW2/report/report.tex` and generated PDF. Contains scene setup, sensor integration, RL training, and navigation analysis.
- Key implementation files:
  - `HW2/part1/hello_world.py` — Genesis scene construction (table, robots, objects, camera).
  - `HW2/part2/imu.py`, `HW2/part2/lidar.py` — Sensor integration and data recording.
  - `HW2/part3/go2_env.py` — RL environment for Go2 quadruped.
  - `HW2/part3/go2_train.py` — PPO training script for locomotion policy.
  - `HW2/part3/go2_eval.py` — Policy evaluation and velocity tracking plots/videos.
  - `HW2/part3/go2_maze.py`, `HW2/part3/go2_maze_run.py` — Maze environment and navigation demo.
  - `HW2/scripts/depth_viz.py`, `HW2/scripts/lidar_viz.py` — Visualization of sensor data.

- Typical workflow:
  1. Run `part1/hello_world.py` to construct and record the basic scene.
  2. Use `part2/imu.py` and `part2/lidar.py` to attach sensors and record data; visualize with provided scripts.
  3. Train the RL policy with `part3/go2_train.py --max_iterations 1001`.
  4. Evaluate the trained policy with `part3/go2_eval.py --ckpt 1000` and generate plots/videos.
  5. Run the maze navigation demo with `part3/go2_maze_run.py --ckpt 1000` to record top-down, depth, and LiDAR videos.

- Reproducibility: record the git commit hash and all command-line parameters in the report. All scripts use Tyro CLI for configuration.

---

## HW3-specific notes (Model-Based Reinforcement Learning)

- Report: `HW3-MBRL/EAI_hw3.pdf` — Contains methodology, experiment results, and analysis for model-based RL tasks.
- Key implementation files:
  - `HW3-MBRL/src/agent.py` — MBRL agent implementation (e.g., PETS, MBPO, or similar algorithms).
  - `HW3-MBRL/src/env.py` — Environment wrappers and utilities for continuous control tasks.
  - `HW3-MBRL/src/train.py` — Training script for running experiments.
  - `HW3-MBRL/src/logger.py`, `helper.py` — Logging and helper utilities.
  - `HW3-MBRL/cfg.yaml` — Experiment configuration (hyperparameters, environment selection, etc.).
  - `HW3-MBRL/requirements.txt` — Python dependencies for HW3.

- Typical workflow:
  1. Edit `cfg.yaml` to set up experiment parameters (environment, agent, training steps, etc.).
  2. Run `python src/train.py --config cfg.yaml` to start training.
  3. Monitor logs and outputs in `logs/`, `outputs/`, or `runs/` directories.
  4. Analyze results and plots for evaluation.

- Reproducibility: record the git commit hash and all command-line/configuration parameters in the report. Use provided scripts and config files for consistent experiments.

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