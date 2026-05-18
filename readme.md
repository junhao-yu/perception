# Fast-FoundationStereo: Real-Time Zero-Shot Stereo Matching

This is the official implementation of our paper accepted to CVPR 2026.

[[Paper]](https://arxiv.org/abs/2512.11130) [[Website]](https://nvlabs.github.io/Fast-FoundationStereo/)


# Environment setup

- Option 1: Docker
```bash
docker build --network host -t ffs -f docker/dockerfile .
bash docker/run_container.sh
```

- Option 2: pip
```bash
conda create -n ffs python=3.12 && conda activate ffs
pip install torch==2.6.0 torchvision==0.21.0 xformers --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

# Weights

Download from [here](https://drive.google.com/drive/folders/1HuTt7UIp7gQsMiDvJwVuWmKpvFzIIMap?usp=drive_link) and put under `weights/` (e.g. `./weights/23-36-37`).

| Checkpoint     | valid_iters | Runtime-Pytorch (ms) | Runtime-TRT (ms) | Peak Memory (MB) |
|---------------|-------------|-------------|-----------------|-----------------|
| `23-36-37`    | 8           | 49.4        | 23.4            | 653             |
| `23-36-37`    | 4           | 41.1        | 18.4            | 653             |
| `20-26-39`    | 8           | 43.6        | 19.4            | 651             |
| `20-26-39`    | 4           | 37.5        | 16.4            | 651             |
| `20-30-48`    | 8           | 38.4        | 16.6            | 646             |
| `20-30-48`    | 4           | 29.3        | 14.0            | 646             |

Runtime profiled on GPU 3090, image size 640x480.

# Run demo (static images)

```bash
python scripts/run_demo.py \
    --model_dir weights/20-30-48/model_best_bp2_serialize.pth \
    --left_file demo_data/left.png \
    --right_file demo_data/right.png \
    --intrinsic_file demo_data/K.txt \
    --out_dir output/ \
    --remove_invisible 1 \
    --denoise_cloud 1 \
    --scale 1 \
    --get_pc 1 \
    --valid_iters 4 \
    --max_disp 192 \
    --zfar 100
```

| Flag                 | Meaning                                                                |
|----------------------|------------------------------------------------------------------------|
| `--model_dir`        | Path to the trained weights/model file                                 |
| `--left_file`        | Path to the left image file                                            |
| `--right_file`       | Path to the right image file                                           |
| `--intrinsic_file`   | Path to the camera intrinsic matrix and baseline file                  |
| `--out_dir`          | Output directory for saving results                                    |
| `--remove_invisible` | Whether to ignore non-overlapping region's depth (0: no, 1: yes)      |
| `--denoise_cloud`    | Whether to apply denoising to the point cloud (0: no, 1: yes)          |
| `--scale`            | Image scaling factor                                                   |
| `--get_pc`           | Obtain point cloud output (0: no, 1: yes)                              |
| `--valid_iters`      | Number of refinement updates during forward pass                       |
| `--max_disp`         | Maximum disparity for volume encoding, 192 should be enough            |
| `--zfar`             | Maximum depth to include in point cloud                                |

**Tips:**
- The input left and right images should be rectified and undistorted.
- Do not swap left and right image.
- The model performs better for image width <1000. Use `--scale 0.5` for larger images.
- For faster inference, reduce `--valid_iters 4` and/or `--scale 0.5`.
- The 1st time running is slower due to CUDA kernel compilation.

# Stereo camera calibration (DECXIN-2784V1)

Print the chessboard at `calibrations/chessboard_9x6_25mm.png` with 100% scale (25mm per square). Adjust `--chessboard_cols` / `--chessboard_rows` / `--square_size` for your own board.

```bash
python scripts/calibrate_stereo.py \
    --out_dir calibrations/ \
    --cam_id 2 \
    --width 1280 \
    --height 480 \
    --chessboard_cols 11 \
    --chessboard_rows 8 \
    --square_size 0.025
```

Hold the chessboard at various angles and distances. Press **SPACE** to capture (when colored corners appear on both eyes). Collect 15-30 pairs, then press **ESC** to compute calibration.

Output files:
- `calibrations/stereo_calib.npz` — full calibration (for live scripts)
- `calibrations/K.txt` — simplified intrinsics (for static demo)

# Live stereo depth (with calibration)

```bash
python scripts/live_stereo.py \
    --model_dir weights/20-30-48/model_best_bp2_serialize.pth \
    --calib_file calibrations/stereo_calib.npz \
    --cam_id 2 \
    --cam_width 1280 \
    --cam_height 480 \
    --valid_iters 4 \
    --depth_scale 1.0
```

Shows real-time disparity visualization with metric depth at screen center. Press `q` or `ESC` to exit.

# Depth scale correction

Stereo depth scales linearly with `fx × baseline`. Calibration errors (e.g. inaccurate square size) cause a constant scale offset. Measure depth at several known distances and compute the correction factor:

```bash
python scripts/check_depth.py \
    --model_dir weights/20-30-48/model_best_bp2_serialize.pth \
    --calib_file calibrations/stereo_calib.npz \
    --depth_scale 1.0
```

If measured depth is consistently shorter than ground truth, increase `--depth_scale` (e.g. 1.22 means depth is 22% too short). All live scripts accept `--depth_scale`.

# YOLOv8 + Stereo: 3D object detection

Detect objects with YOLOv8 and estimate their 3D position from stereo depth.

## Install

```bash
pip install ultralytics
```

## Run

```bash
python scripts/live_yolo_stereo.py \
    --model_dir weights/20-30-48/model_best_bp2_serialize.pth \
    --calib_file calibrations/stereo_calib.npz \
    --yolo_weights ../yolov8/runs/detect/train-3/weights/best.pt \
    --yolo_conf 0.5 \
    --depth_scale 1.22 \
    --cam_id 2
```

| Flag              | Meaning                                                |
|-------------------|--------------------------------------------------------|
| `--yolo_weights`  | Path to YOLOv8 trained weights (.pt)                   |
| `--yolo_conf`     | Detection confidence threshold (default 0.5)           |
| `--depth_scale`   | Depth scale correction factor (default 1.0)            |
| `--cam_id`        | V4L2 camera device index (default 2 for DECXIN)        |
| `--cam_width`     | Combined stereo frame width (default 1280)             |
| `--cam_height`    | Frame height (default 480)                             |
| `--valid_iters`   | GRU refinement iterations (4 for speed, 8 for quality) |
| `--scale`         | Image scale factor (<1 for faster inference)           |

Each detected object shows: confidence, X (right), Y (down), Z (forward) in meters.

Camera coordinate system: origin at left lens optical center, X right, Y down, Z forward.


