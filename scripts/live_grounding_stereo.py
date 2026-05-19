"""
Live Grounding DINO + Fast-FoundationStereo for open-vocabulary 3D object detection.

- Grounding DINO detects arbitrary objects on the left image (zero-shot, no training needed)
- Fast-FoundationStereo computes disparity → depth from stereo
- Each detection gets a 3D position (X, Y, Z in camera frame) from depth at bbox center

Usage:
  python scripts/live_grounding_stereo.py \
      --model_dir weights/20-30-48/model_best_bp2_serialize.pth \
      --calib_file calibrations/stereo_calib.npz \
      --gdino_model IDEA-Research/grounding-dino-tiny \
      --text_prompt "a box. a chair. a door. a person." \
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
from Utils import AMP_DTYPE, set_logging_format, set_seed, vis_disparity
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
    (0, 128, 255), (128, 0, 255),
]


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
    return np.array([(px - K[0, 2]) * depth / K[0, 0],
                     (py - K[1, 2]) * depth / K[1, 1],
                     depth])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to FFS .pth weights')
    parser.add_argument('--gdino_model', type=str, default='models/grounding-dino-tiny',
                        help='HuggingFace model ID or local path for Grounding DINO')
    parser.add_argument('--text_prompt', type=str, default='a box. a chair. a door. a person. an obstacle.',
                        help='Objects to detect, separated by ". "')
    parser.add_argument('--box_threshold', type=float, default=0.3,
                        help='Detection confidence threshold')
    parser.add_argument('--text_threshold', type=float, default=0.25,
                        help='Text matching threshold')
    parser.add_argument('--calib_file', type=str, default=None)
    parser.add_argument('--cam_id', default=2, type=int)
    parser.add_argument('--cam_width', default=1280, type=int)
    parser.add_argument('--cam_height', default=480, type=int)
    parser.add_argument('--scale', default=1.0, type=float)
    parser.add_argument('--valid_iters', default=4, type=int)
    parser.add_argument('--max_disp', default=192, type=int)
    parser.add_argument('--depth_scale', default=1.0, type=float,
                        help='Scale correction factor for depth')
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
    ffs_model = torch.load(args.model_dir, map_location='cpu', weights_only=False)
    ffs_model.args.valid_iters = args.valid_iters
    ffs_model.args.max_disp = args.max_disp
    ffs_model.cuda().eval()

    # --- Load Grounding DINO ---
    logging.info(f"Loading Grounding DINO: {args.gdino_model}")
    gdino_processor = AutoProcessor.from_pretrained(args.gdino_model)
    gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(args.gdino_model).to('cuda').eval()
    logging.info(f"Detection prompt: {args.text_prompt}")

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
        _ = ffs_model.forward(t_l, t_r, iters=args.valid_iters, test_mode=True, optimize_build_volume='pytorch1')

    # --- Warmup Grounding DINO ---
    logging.info("Warming up Grounding DINO...")
    pil_img = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
    gd_inputs = gdino_processor(images=pil_img, text=args.text_prompt, return_tensors="pt").to('cuda')
    if gd_inputs.get('pixel_values') is not None:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _ = gdino_model(**gd_inputs)

    logging.info("Warmup complete. Starting live open-vocabulary detection...")

    # --- Main loop ---
    frame_idx = 0
    fps_window = []
    print("\n" + "=" * 60)
    print(f"Grounding DINO + Fast-FoundationStereo  |  Detecting: {args.text_prompt}")
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

        # --- Grounding DINO on left image ---
        pil_img = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
        gd_inputs = gdino_processor(images=pil_img, text=args.text_prompt, return_tensors="pt").to('cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            gd_outputs = gdino_model(**gd_inputs)

        gd_results = gdino_processor.post_process_grounded_object_detection(
            gd_outputs,
            gd_inputs.input_ids,
            threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            target_sizes=[(H_ori, W_ori)]
        )[0]

        # --- FFS stereo depth ---
        t_l = torch.as_tensor(img_l).cuda().float()[None].permute(0, 3, 1, 2)
        t_r = torch.as_tensor(img_r).cuda().float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(t_l.shape, divis_by=32, force_square=False)
        t_l, t_r = padder.pad(t_l, t_r)
        with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
            disp = ffs_model.forward(t_l, t_r, iters=args.valid_iters, test_mode=True, optimize_build_volume='pytorch1')
        disp = padder.unpad(disp.float())
        disp_np = disp.data.cpu().numpy().reshape(H_ori, W_ori).clip(0, None)

        depth_m = None
        if K is not None and baseline is not None:
            depth_m = K[0, 0] * baseline / disp_np.clip(0.1, None) * args.depth_scale

        # --- Visualization ---
        disp_vis = vis_disparity(disp_np, min_val=None, max_val=None, cmap=None, color_map=cv2.COLORMAP_TURBO)
        display = np.concatenate([img_l, img_r, disp_vis], axis=1)

        # Draw Grounding DINO detections
        for i in range(len(gd_results['boxes'])):
            x1, y1, x2, y2 = [int(v) for v in gd_results['boxes'][i].tolist()]
            score = float(gd_results['scores'][i])
            label = gd_results['labels'][i]
            color = COLORS[hash(label) % len(COLORS)]

            cx_bbox = (x1 + x2) / 2
            cy_bbox = (y1 + y2) / 2
            pos_3d = None
            if depth_m is not None:
                d = depth_at_bbox(depth_m, (x1, y1, x2, y2))
                if d is not None:
                    pos_3d = pixel_to_camera(cx_bbox, cy_bbox, d, K)

            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

            if pos_3d is not None:
                txt = f"{label} {score:.2f} | X:{pos_3d[0]:.2f} Y:{pos_3d[1]:.2f} Z:{pos_3d[2]:.2f}m"
            else:
                txt = f"{label} {score:.2f}"
            cv2.putText(display, txt, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)

        # FPS
        dt = time.perf_counter() - t_start
        fps_window.append(1.0 / dt if dt > 0 else 0)
        if len(fps_window) > 30:
            fps_window.pop(0)
        avg_fps = np.mean(fps_window)

        cv2.putText(display, f"FPS: {avg_fps:.1f}  |  thresh={args.box_threshold}  |  {args.gdino_model.split('/')[-1]}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if args.display:
            ds = 1280 / display.shape[1]
            display_small = cv2.resize(display, (int(display.shape[1] * ds), int(display.shape[0] * ds)))
            cv2.imshow('Grounding DINO + Fast-FoundationStereo', display_small)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    logging.info(f"Finished. {frame_idx} frames.")


if __name__ == '__main__':
    main()
