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

# Detection + Stereo: 3D object localization

Unified perception pipeline: detector → stereo depth → 3D position. Three detector backends available, switchable via `--detector_type`.

## Install

```bash
# YOLO / YOLO-World backend (conda env: ffs)
pip install ultralytics

# Grounding DINO backend (conda env: ffs1 — also needs ultralytics for YOLO-World)
pip install transformers
# Download model:
HF_ENDPOINT=https://hf-mirror.com hf download IDEA-Research/grounding-dino-tiny \
    --local-dir models/grounding-dino-tiny
```

## Detector comparison (RTX 5060 Laptop, --scale 0.5)

| Detector          | detect (ms) | total FPS | Open-vocabulary | Fine-tunable |
|-------------------|:-----------:|:---------:|:---------------:|:------------:|
| YOLO-World v2     | ~4          | ~25       | Yes             | Yes          |
| YOLOv8            | ~4          | ~25       | No (train req'd)| Yes          |
| Grounding DINO    | ~250        | ~3        | Yes             | No           |

All detectors share `--text_prompt` (`.`-separated, English). YOLO-World v2 is recommended — fast like YOLOv8, open-vocabulary like DINO, and fine-tunable with 5-10 images per class.

## Run

### YOLO-World v2 (recommended)

```bash
python scripts/run_perception.py \
    --detector_type yolo_world \
    --model_dir weights/20-30-48/model_best_bp2_serialize.pth \
    --calib_file calibrations/stereo_calib.npz \
    --yolo_weights yolov8s-worldv2.pt \
    --text_prompt "a cup. a bottle. a box." \
    --depth_scale 1.22 \
    --scale 0.5
```

Weights auto-download on first run from GitHub. In China, pre-download with: `wget https://gh-proxy.com/https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-worldv2.pt`. Variants: `s` (small), `m` (medium), `l` (large).

### YOLO (trained model — fast, fixed classes)

```bash
python scripts/run_perception.py \
    --detector_type yolo \
    --model_dir weights/20-30-48/model_best_bp2_serialize.pth \
    --calib_file calibrations/stereo_calib.npz \
    --yolo_weights ../yolov8/runs/detect/train-3/weights/best.pt \
    --depth_scale 1.22 \
    --scale 0.5
```

### Grounding DINO (open-vocabulary — slow, best zero-shot accuracy)

```bash
python scripts/run_perception.py \
    --detector_type grounding_dino \
    --model_dir weights/20-30-48/model_best_bp2_serialize.pth \
    --calib_file calibrations/stereo_calib.npz \
    --gdino_model models/grounding-dino-tiny \
    --text_prompt "a cup. a bottle. a box. a person." \
    --depth_scale 1.22
```

| Flag                | Meaning                                                |
|---------------------|--------------------------------------------------------|
| `--detector_type`   | `yolo`, `yolo_world`, or `grounding_dino`              |
| `--model_dir`       | Path to FFS weights                                    |
| `--calib_file`      | Path to stereo calibration .npz                        |
| `--depth_scale`     | Depth scale correction factor (default 1.0)            |
| `--cam_id`          | V4L2 camera device index (default 2 for DECXIN)        |
| `--cam_width`       | Combined stereo frame width (default 1280)             |
| `--cam_height`      | Frame height (default 480)                             |
| `--valid_iters`     | GRU refinement iterations (4 for speed, 8 for quality) |
| `--scale`           | Image scale factor (0.5 recommended for speed)         |
|                      |                                                        |
| **Open-vocabulary options** (yolo_world & grounding_dino) |                              |
| `--text_prompt`     | Text description of objects to detect, `.` separated   |
|                      |                                                        |
| **Grounding DINO options** |                                                  |
| `--gdino_model`     | Path to Grounding DINO model                           |
| `--box_threshold`   | Box confidence threshold (default 0.3)                 |
| `--text_threshold`  | Text similarity threshold (default 0.25)               |
|                      |                                                        |
| **YOLO / YOLO-World options** |                                                   |
| `--yolo_weights`    | Path to YOLO/YOLO-World weights (.pt)                  |
| `--yolo_conf`       | Detection confidence threshold (default 0.5)           |

Each detected object shows: class label, confidence, X (right), Y (down), Z (forward) in meters.

Camera coordinate system: origin at left lens optical center, X right, Y down, Z forward.

### Legacy standalone scripts

```bash
python scripts/live_yolo_stereo.py ...        # YOLO standalone (reference)
python scripts/live_grounding_stereo.py ...   # Grounding DINO standalone (reference)
```

Prefer `scripts/run_perception.py` for new work.

# Fine-tune YOLO-World v2

Improve detection accuracy on specific objects with few-shot fine-tuning. **5-10 images per class** is often enough — the model already has strong open-vocabulary priors; fine-tuning only needs to adapt to your camera viewpoint and lighting.

Example results (cup+bottle, 15 images, 12 train / 3 val, 20 epochs):
- cup mAP50-95: 99.5%
- bottle mAP50-95: 79.0%
- Inference: 2.5ms per frame

## Workflow

### 1. Capture dataset

Press **SPACE** to save left-eye frames. Move objects to varied positions and angles.

```bash
python scripts/capture_dataset.py \
    --calib_file calibrations/stereo_calib.npz \
    --out_dir dataset/cups_bottles/
```

### 2. Annotate with labelImg

```bash
pip install labelImg
labelImg dataset/cups_bottles/ dataset/cups_bottles/
```

Draw bounding boxes for each object, assign class labels (e.g., `cup`, `bottle`). labelImg saves LabelMe JSON format — `convert_labels.py` handles the conversion.

### 3. Convert annotations to YOLO format

```bash
python scripts/convert_labels.py \
    --json_dir dataset/cups_bottles/ \
    --classes "cup,bottle"
```

This outputs `labels/*.txt` in YOLO format. Class IDs match `--classes` order (e.g., `cup,bottle` → cup=0, bottle=1).

### 4. Organize files (if needed)

If images are in the same directory as JSONs:

```bash
mkdir -p dataset/cups_bottles/images
mv dataset/cups_bottles/*.jpg dataset/cups_bottles/images/
```

Final structure:
```
dataset/cups_bottles/
├── images/
│   ├── frame_00000.jpg
│   └── ...
├── labels/
│   ├── frame_00000.txt    # 0 0.375 0.639 0.066 0.143 ...
│   └── ...
└── data.yaml              # auto-generated by finetune script
```

### 5. Fine-tune

```bash
python scripts/finetune_yolo_world.py \
    --weights yolov8s-worldv2.pt \
    --data_dir dataset/cups_bottles/ \
    --classes "cup,bottle" \
    --epochs 20 \
    --batch 2 \
    --out_dir weights/yolo_world_finetuned/
```

| Flag           | Meaning                                               |
|----------------|-------------------------------------------------------|
| `--weights`    | Base YOLO-World weights (auto-downloaded first time)  |
| `--data_dir`   | Dataset directory with `images/` and `labels/`        |
| `--classes`    | Comma-separated class names                           |
| `--epochs`     | 5-10 for tiny datasets (5-10 imgs), 20-30 for 15-50   |
| `--batch`      | Reduce to 2 if OOM (default 4)                        |
| `--lr0`        | Keep low for fine-tuning (default 0.001)              |
| `--out_dir`    | Output directory for trained model                    |

### 6. Run with fine-tuned model

```bash
python scripts/run_perception.py \
    --detector_type yolo_world \
    --model_dir weights/20-30-48/model_best_bp2_serialize.pth \
    --calib_file calibrations/stereo_calib.npz \
    --yolo_weights weights/yolo_world_finetuned/model_best.pt \
    --text_prompt "a cup. a bottle." \
    --depth_scale 1.22 \
    --scale 0.5
```


