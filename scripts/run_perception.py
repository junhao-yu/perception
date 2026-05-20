"""
Unified perception pipeline: detection + stereo depth → 3D object localization.

Usage:
  # Grounding DINO (open-vocabulary, zero-shot):
  python scripts/run_perception.py \
      --detector_type grounding_dino \
      --model_dir weights/20-30-48/model_best_bp2_serialize.pth \
      --calib_file calibrations/stereo_calib.npz \
      --gdino_model models/grounding-dino-tiny \
      --text_prompt "a box. a chair. a door. a person." \
      --depth_scale 1.22

  # YOLOv8 (trained model):
  python scripts/run_perception.py \
      --detector_type yolo \
      --model_dir weights/20-30-48/model_best_bp2_serialize.pth \
      --calib_file calibrations/stereo_calib.npz \
      --yolo_weights ../yolov8/runs/detect/train-3/weights/best.pt \
      --depth_scale 1.22

  # YOLO-World v2 (open-vocabulary, fast):
  python scripts/run_perception.py \
      --detector_type yolo_world \
      --model_dir weights/20-30-48/model_best_bp2_serialize.pth \
      --calib_file calibrations/stereo_calib.npz \
      --yolo_weights yolov8s-worldv2.pt \
      --text_prompt "a cup. a bottle. a box." \
      --depth_scale 1.22
"""

import argparse, os, sys, time
import numpy as np
import cv2

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')

import torch
import logging
from core.utils.utils import InputPadder
from Utils import AMP_DTYPE, set_logging_format, set_seed

from detectors import YOLODetector, GroundingDINODetector, YOLOWorldDetector
from perception import PerceptionPipeline


def load_calibration(npz_path):
    data = np.load(npz_path)
    K_l, D_l = data['K_left'], data['D_left']
    K_r, D_r = data['K_right'], data['D_right']
    R1, R2 = data['R1'], data['R2']
    P1, P2 = data['P1'], data['P2']
    w, h = tuple(data['image_size'])
    baseline = float(data['baseline_m'].item())

    map_lx, map_ly = cv2.initUndistortRectifyMap(K_l, D_l, R1, P1, (w, h), cv2.CV_32FC1)
    map_rx, map_ry = cv2.initUndistortRectifyMap(K_r, D_r, R2, P2, (w, h), cv2.CV_32FC1)
    P1_KK = P1[:3, :3].copy()
    return {'K': P1_KK, 'baseline': baseline, 'size': (w, h)}, map_lx, map_ly, map_rx, map_ry


COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
    (0, 128, 255), (128, 0, 255),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detector_type', type=str, required=True,
                        choices=['yolo', 'grounding_dino', 'yolo_world'])
    # FFS
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--calib_file', type=str, default=None)
    parser.add_argument('--valid_iters', default=4, type=int)
    parser.add_argument('--max_disp', default=192, type=int)
    parser.add_argument('--depth_scale', default=1.0, type=float)
    # Camera
    parser.add_argument('--cam_id', default=2, type=int)
    parser.add_argument('--cam_width', default=1280, type=int)
    parser.add_argument('--cam_height', default=480, type=int)
    parser.add_argument('--scale', default=1.0, type=float)
    # YOLO / YOLO-World
    parser.add_argument('--yolo_weights', type=str, default=None)
    parser.add_argument('--yolo_conf', default=0.5, type=float)
    # Grounding DINO
    parser.add_argument('--gdino_model', type=str, default='models/grounding-dino-tiny')
    parser.add_argument('--text_prompt', type=str, default='a box. a chair. a door. a person.')
    parser.add_argument('--box_threshold', type=float, default=0.3)
    parser.add_argument('--text_threshold', type=float, default=0.25)
    # Display
    parser.add_argument('--display', type=int, default=1)
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)

    # --- Load calibration ---
    if args.calib_file and os.path.exists(args.calib_file):
        calib, map_lx, map_ly, map_rx, map_ry = load_calibration(args.calib_file)
        logging.info(f"Calibration: baseline={calib['baseline']:.3f}m, {calib['size']}")
    else:
        logging.error("Calibration file required.")
        return

    # --- Load FFS model ---
    logging.info(f"Loading FFS: {args.model_dir}")
    ffs_model = torch.load(args.model_dir, map_location='cpu', weights_only=False)
    ffs_model.args.valid_iters = args.valid_iters
    ffs_model.args.max_disp = args.max_disp
    ffs_model.cuda().eval()

    # --- Load detector ---
    if args.detector_type == 'yolo':
        logging.info(f"Loading YOLO: {args.yolo_weights}")
        detector = YOLODetector(args.yolo_weights, conf=args.yolo_conf)
    elif args.detector_type == 'yolo_world':
        logging.info(f"Loading YOLO-World v2: {args.yolo_weights}")
        logging.info(f"Detection prompt: {args.text_prompt}")
        detector = YOLOWorldDetector(args.yolo_weights, args.text_prompt, conf=args.yolo_conf)
    else:
        logging.info(f"Loading Grounding DINO: {args.gdino_model}")
        logging.info(f"Detection prompt: {args.text_prompt}")
        detector = GroundingDINODetector(
            args.gdino_model, args.text_prompt,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )

    # --- Pipeline ---
    pipeline = PerceptionPipeline(detector, ffs_model, calib, args.depth_scale,
                                   image_scale=args.scale)

    # --- Camera ---
    cap = cv2.VideoCapture(args.cam_id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    assert cap.isOpened(), f"Cannot open /dev/video{args.cam_id}"
    for _ in range(10):
        cap.read()

    # --- Warmup ---
    logging.info("Warming up...")
    ret, frame = cap.read()
    mid = frame.shape[1] // 2
    img_l = frame[:, :mid, :3]
    img_r = frame[:, mid:, :3]
    img_l = cv2.remap(img_l, map_lx, map_ly, cv2.INTER_LINEAR)
    img_r = cv2.remap(img_r, map_rx, map_ry, cv2.INTER_LINEAR)
    if args.scale != 1.0:
        img_l = cv2.resize(img_l, fx=args.scale, fy=args.scale, dsize=None)
        img_r = cv2.resize(img_r, dsize=(img_l.shape[1], img_l.shape[0]))

    # Warmup FFS
    t_l = torch.as_tensor(img_l).cuda().float()[None].permute(0, 3, 1, 2)
    t_r = torch.as_tensor(img_r).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(t_l.shape, divis_by=32, force_square=False)
    t_l, t_r = padder.pad(t_l, t_r)
    with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
        _ = ffs_model.forward(t_l, t_r, iters=args.valid_iters, test_mode=True, optimize_build_volume='pytorch1')
    logging.info("FFS warmup done.")

    # Warmup detector
    detector.warmup(img_l)
    logging.info(f"Detector warmup done. Starting live perception...")

    # --- Main loop ---
    fps_window = []
    timing = {'cap_read': [], 'remap': [], 'resize': [], 'pipeline': [], 'display': []}
    frame_count = 0

    print("\n" + "=" * 60)
    print(f"Detector: {args.detector_type}  |  Stereo: FFS  |  depth_scale={args.depth_scale}")
    if args.scale != 1.0:
        print(f"Image scale: {args.scale}")
    if args.detector_type in ('grounding_dino', 'yolo_world'):
        print(f"Prompt: {args.text_prompt}")
    print("Press 'q' or ESC to exit.")
    print("=" * 60 + "\n")

    while True:
        t_start = time.perf_counter()

        t0 = time.perf_counter()
        ret, frame = cap.read()
        t_cap = time.perf_counter() - t0
        if not ret:
            continue

        # --- Preprocess ---
        t0 = time.perf_counter()
        mid = frame.shape[1] // 2
        img_l = frame[:, :mid, :3]
        img_r = frame[:, mid:, :3]
        img_l = cv2.remap(img_l, map_lx, map_ly, cv2.INTER_LINEAR)
        img_r = cv2.remap(img_r, map_rx, map_ry, cv2.INTER_LINEAR)
        t_remap = time.perf_counter() - t0

        t0 = time.perf_counter()
        if args.scale != 1.0:
            img_l = cv2.resize(img_l, fx=args.scale, fy=args.scale, dsize=None)
            img_r = cv2.resize(img_r, dsize=(img_l.shape[1], img_l.shape[0]))
        t_resize = time.perf_counter() - t0

        # --- Pipeline: detect + depth + fuse ---
        t0 = time.perf_counter()
        disp_vis, depth_m, detections_3d = pipeline.process_frame(img_l, img_r)
        t_pipeline = time.perf_counter() - t0

        # --- Display ---
        t0 = time.perf_counter()
        display = np.concatenate([img_l, img_r, disp_vis], axis=1)

        for det, pos_3d in detections_3d:
            x1, y1, x2, y2 = det.bbox
            color = COLORS[hash(det.label) % len(COLORS)]
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            txt = f"{det.label} {det.score:.2f} | X:{pos_3d[0]:.2f} Y:{pos_3d[1]:.2f} Z:{pos_3d[2]:.2f}m"
            cv2.putText(display, txt, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)

        dt = time.perf_counter() - t_start
        fps_window.append(1.0 / dt if dt > 0 else 0)
        if len(fps_window) > 30:
            fps_window.pop(0)
        avg_fps = np.mean(fps_window)

        cv2.putText(display, f"FPS: {avg_fps:.1f}  |  detector: {args.detector_type}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if args.display:
            ds = 1280 / display.shape[1]
            display_small = cv2.resize(display, (int(display.shape[1] * ds), int(display.shape[0] * ds)))
            cv2.imshow('Perception Pipeline', display_small)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
        t_display = time.perf_counter() - t0

        # --- Accumulate timing ---
        timing['cap_read'].append(t_cap * 1000)
        timing['remap'].append(t_remap * 1000)
        timing['resize'].append(t_resize * 1000)
        timing['pipeline'].append(t_pipeline * 1000)
        timing['display'].append(t_display * 1000)
        frame_count += 1

        if frame_count % 30 == 0:
            print(f"\n--- Frame {frame_count}: avg timing (ms) over last 30 frames ---")
            for step in ['cap_read', 'remap', 'resize', 'pipeline', 'display']:
                vals = timing[step]
                if vals:
                    print(f"  {step:10s}: {np.mean(vals):6.1f}  (max: {np.max(vals):6.1f})")
            total_ms = sum(np.mean(timing[k]) for k in timing if timing[k])
            print(f"  {'TOTAL':10s}: {total_ms:6.1f}  (~{1000/total_ms:.0f} FPS max)")
            pt = pipeline.last_timing
            print(f"  ── pipeline internals ──")
            for k, v in pt.items():
                print(f"    {k:8s}: {v:6.1f}")
            for k in timing:
                timing[k].clear()

    cap.release()
    cv2.destroyAllWindows()
    logging.info(f"Finished.")


if __name__ == '__main__':
    main()
