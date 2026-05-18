"""
Live YOLOv8 + Fast-FoundationStereo integration for 3D object detection.

- YOLOv8 detects cardboard boxes on the left image
- Fast-FoundationStereo computes disparity → depth from stereo
- Each detection gets a 3D position (X, Y, Z in camera frame) from depth at bbox center

Usage:
  python scripts/live_yolo_stereo.py \
      --model_dir weights/20-30-48/model_best_bp2_serialize.pth \
      --calib_file calibrations/stereo_calib.npz \
      --yolo_weights ../yolov8/runs/detect/train-3/weights/best.pt \
      --cam_id 2
"""

import argparse, os, sys, time
import numpy as np
import cv2

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')

import torch
import logging
from core.utils.utils import InputPadder
from Utils import AMP_DTYPE, set_logging_format, set_seed, vis_disparity
from ultralytics import YOLO


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
    return map_lx, map_ly, map_rx, map_ry, P1[:3, :3].copy(), baseline, (w, h)


def depth_at_bbox(depth_map, bbox, margin_ratio=0.15):
    """Median depth in the center region of bbox (avoids edge noise)."""
    x1, y1, x2, y2 = bbox
    h, w = depth_map.shape
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None

    mw = int((x2 - x1) * margin_ratio)
    mh = int((y2 - y1) * margin_ratio)
    roi = depth_map[y1 + mh:y2 - mh, x1 + mw:x2 - mw]
    valid = roi[(roi > 0) & (roi < 100)]
    return float(np.median(valid)) if len(valid) > 10 else None


def pixel_to_camera(px, py, depth, K):
    """Convert pixel coordinate + depth to camera-frame 3D point."""
    x = (px - K[0, 2]) * depth / K[0, 0]
    y = (py - K[1, 2]) * depth / K[1, 1]
    return np.array([x, y, depth])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--yolo_weights', type=str, required=True)
    parser.add_argument('--calib_file', type=str, default=None)
    parser.add_argument('--cam_id', default=2, type=int)
    parser.add_argument('--cam_width', default=1280, type=int)
    parser.add_argument('--cam_height', default=480, type=int)
    parser.add_argument('--scale', default=1.0, type=float)
    parser.add_argument('--valid_iters', default=4, type=int)
    parser.add_argument('--max_disp', default=192, type=int)
    parser.add_argument('--yolo_conf', default=0.5, type=float)
    parser.add_argument('--depth_scale', default=1.0, type=float,
                        help='Scale correction factor for depth (1.0 = no correction, >1 = deeper)')
    parser.add_argument('--display', type=int, default=1)
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)

    # --- Load calibration ---
    if args.calib_file and os.path.exists(args.calib_file):
        map_lx, map_ly, map_rx, map_ry, K_rect, baseline, (calib_w, calib_h) = load_calibration(args.calib_file)
        logging.info(f"Calibration loaded: baseline={baseline:.3f}m, {calib_w}x{calib_h}")
        K = K_rect
    else:
        logging.warning("No calibration — depth will be relative, not metric.")
        K = None
        baseline = None

    # --- Load FFS model ---
    logging.info(f"Loading FFS from {args.model_dir}")
    model = torch.load(args.model_dir, map_location='cpu', weights_only=False)
    model.args.valid_iters = args.valid_iters
    model.args.max_disp = args.max_disp
    model.cuda().eval()

    # --- Load YOLO ---
    logging.info(f"Loading YOLO from {args.yolo_weights}")
    yolo = YOLO(args.yolo_weights)
    yolo.to('cuda')

    # --- Open camera ---
    cap = cv2.VideoCapture(args.cam_id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    assert cap.isOpened(), f"Cannot open /dev/video{args.cam_id}"

    for _ in range(10):
        cap.read()

    # --- Warmup FFS ---
    logging.info("Warming up FFS (compiling CUDA kernels)...")
    ret, frame = cap.read()
    mid = frame.shape[1] // 2
    img_l = frame[:, :mid, :3]
    img_r = frame[:, mid:, :3]
    img_l = cv2.remap(img_l, map_lx, map_ly, cv2.INTER_LINEAR)
    img_r = cv2.remap(img_r, map_rx, map_ry, cv2.INTER_LINEAR)
    if args.scale != 1.0:
        img_l = cv2.resize(img_l, fx=args.scale, fy=args.scale, dsize=None)
        img_r = cv2.resize(img_r, dsize=(img_l.shape[1], img_l.shape[0]))
    t_l = torch.as_tensor(img_l).cuda().float()[None].permute(0, 3, 1, 2)
    t_r = torch.as_tensor(img_r).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(t_l.shape, divis_by=32, force_square=False)
    t_l, t_r = padder.pad(t_l, t_r)
    with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
        _ = model.forward(t_l, t_r, iters=args.valid_iters, test_mode=True, optimize_build_volume='pytorch1')

    logging.info("Warmup complete. Starting live detection...")

    # --- Main loop ---
    frame_idx = 0
    fps_window = []
    print("\n" + "=" * 60)
    print("Live YOLO + Fast-FoundationStereo running.")
    print("Press 'q' or ESC to exit.")
    print("=" * 60 + "\n")

    while True:
        t_start = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            continue

        mid = frame.shape[1] // 2
        img_l = frame[:, :mid, :3]
        img_r = frame[:, mid:, :3]

        img_l = cv2.remap(img_l, map_lx, map_ly, cv2.INTER_LINEAR)
        img_r = cv2.remap(img_r, map_rx, map_ry, cv2.INTER_LINEAR)

        H_ori, W_ori = img_l.shape[:2]
        if args.scale != 1.0:
            img_l = cv2.resize(img_l, fx=args.scale, fy=args.scale, dsize=None)
            img_r = cv2.resize(img_r, dsize=(img_l.shape[1], img_l.shape[0]))

        # --- YOLO detection on left image ---
        yolo_results = yolo(img_l, conf=args.yolo_conf, verbose=False)

        # --- FFS stereo depth ---
        t_l = torch.as_tensor(img_l).cuda().float()[None].permute(0, 3, 1, 2)
        t_r = torch.as_tensor(img_r).cuda().float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(t_l.shape, divis_by=32, force_square=False)
        t_l, t_r = padder.pad(t_l, t_r)

        with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
            disp = model.forward(t_l, t_r, iters=args.valid_iters, test_mode=True, optimize_build_volume='pytorch1')
        disp = padder.unpad(disp.float())
        disp_np = disp.data.cpu().numpy().reshape(H_ori, W_ori).clip(0, None)

        # Depth map
        depth_m = None
        if K is not None and baseline is not None:
            depth_m = K[0, 0] * baseline / disp_np.clip(0.1, None) * args.depth_scale

        # --- Visualization ---
        disp_vis = vis_disparity(disp_np, min_val=None, max_val=None, cmap=None, color_map=cv2.COLORMAP_TURBO)
        display = np.concatenate([img_l, img_r, disp_vis], axis=1)

        # Draw YOLO detections on the left image
        boxes = yolo_results[0].boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = float(box.conf[0])

                # Compute 3D position
                cx_bbox = (x1 + x2) / 2
                cy_bbox = (y1 + y2) / 2
                pos_3d = None
                if depth_m is not None:
                    d = depth_at_bbox(depth_m, (x1, y1, x2, y2))
                    if d is not None:
                        pos_3d = pixel_to_camera(cx_bbox, cy_bbox, d, K)

                # Draw bbox
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Label
                if pos_3d is not None:
                    label = f"Box {conf:.2f} | X:{pos_3d[0]:.2f} Y:{pos_3d[1]:.2f} Z:{pos_3d[2]:.2f}m"
                else:
                    label = f"Box {conf:.2f}"
                cv2.putText(display, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        # FPS
        dt = time.perf_counter() - t_start
        fps_window.append(1.0 / dt if dt > 0 else 0)
        if len(fps_window) > 30:
            fps_window.pop(0)
        avg_fps = np.mean(fps_window)

        cv2.putText(display, f"FPS: {avg_fps:.1f}  |  YOLO conf={args.yolo_conf}  |  iters={args.valid_iters}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        if args.display:
            ds = 1280 / display.shape[1]
            display_small = cv2.resize(display, (int(display.shape[1] * ds), int(display.shape[0] * ds)))
            cv2.imshow('YOLO + Fast-FoundationStereo', display_small)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    logging.info(f"Finished. {frame_idx} frames.")


if __name__ == '__main__':
    main()
