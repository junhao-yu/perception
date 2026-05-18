"""
Live stereo depth estimation using DECXIN-2784V1 (or other UVC stereo cameras).

The DECXIN-2784V1 presents as a single side-by-side frame via UVC.
Use --calib_file if you have calibrated the camera with calibrate_stereo.py,
otherwise raw (unrectified) frames are used directly.

Usage:
  # With calibration (recommended):
  python scripts/live_stereo.py --model_dir weights/23-36-37/model_best_bp2_serialize.pth \
      --calib_file calibrations/stereo_calib.npz --cam_id 2

  # Without calibration (if camera is already rectified):
  python scripts/live_stereo.py --model_dir weights/23-36-37/model_best_bp2_serialize.pth \
      --intrinsic_file demo_data/K.txt --cam_id 2 --no_rectify
"""

import argparse
import os
import sys
import time
import numpy as np
import cv2

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')

import torch
import logging

from core.utils.utils import InputPadder
from Utils import AMP_DTYPE, set_logging_format, set_seed, vis_disparity, depth2xyzmap, toOpen3dCloud


def load_calibration(npz_path):
    """Load stereo calibration and precompute rectification maps."""
    data = np.load(npz_path)
    K_l = data['K_left']
    D_l = data['D_left']
    K_r = data['K_right']
    D_r = data['D_right']
    R1 = data['R1']
    R2 = data['R2']
    P1 = data['P1']
    P2 = data['P2']
    image_size = tuple(data['image_size'])
    baseline = float(data['baseline_m'].item())
    w, h = image_size

    map_lx, map_ly = cv2.initUndistortRectifyMap(K_l, D_l, R1, P1, (w, h), cv2.CV_32FC1)
    map_rx, map_ry = cv2.initUndistortRectifyMap(K_r, D_r, R2, P2, (w, h), cv2.CV_32FC1)

    # Use P1 (rectified) as the intrinsic for depth computation
    K_rect = P1[:3, :3].copy()
    return map_lx, map_ly, map_rx, map_ry, K_rect, baseline, image_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to .pth weights file')
    parser.add_argument('--cam_id', default=2, type=int,
                        help='V4L2 device index for DECXIN camera')
    parser.add_argument('--cam_width', default=1280, type=int,
                        help='Combined frame width')
    parser.add_argument('--cam_height', default=480, type=int,
                        help='Frame height')
    parser.add_argument('--calib_file', default=None, type=str,
                        help='Path to stereo_calib.npz from calibrate_stereo.py')
    parser.add_argument('--intrinsic_file', default=None, type=str,
                        help='Path to K.txt (used only in --no_rectify mode)')
    parser.add_argument('--no_rectify', action='store_true',
                        help='Skip rectification (use if camera is already rectified)')
    parser.add_argument('--out_dir', default=None, type=str,
                        help='Directory to save output; if set, saves each frame')
    parser.add_argument('--scale', default=1.0, type=float,
                        help='Image scaling factor, <1 for faster inference')
    parser.add_argument('--valid_iters', default=4, type=int,
                        help='GRU refinement iterations (4 for speed, 8 for quality)')
    parser.add_argument('--max_disp', default=192, type=int)
    parser.add_argument('--hiera', default=0, type=int)
    parser.add_argument('--get_pc', type=int, default=0,
                        help='Save point cloud for each frame')
    parser.add_argument('--zfar', type=float, default=100)
    parser.add_argument('--remove_invisible', type=int, default=1)
    parser.add_argument('--depth_scale', default=1.0, type=float,
                        help='Scale correction factor for depth (1.0 = no correction, >1 = deeper)')
    parser.add_argument('--display', type=int, default=1,
                        help='Show live disparity visualization')
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    # --- Load calibration ---
    rectify = not args.no_rectify
    if rectify and args.calib_file and os.path.exists(args.calib_file):
        map_lx, map_ly, map_rx, map_ry, K_rect, baseline, (calib_w, calib_h) = load_calibration(args.calib_file)
        logging.info(f"Loaded calibration: baseline={baseline:.4f}m, size={calib_w}x{calib_h}")
    elif rectify and (not args.calib_file or not os.path.exists(args.calib_file)):
        logging.warning("No calibration file found. Switching to --no_rectify mode.")
        rectify = False

    # --- Load model ---
    logging.info(f"Loading model from {args.model_dir}")
    model = torch.load(args.model_dir, map_location='cpu', weights_only=False)
    model.args.valid_iters = args.valid_iters
    model.args.max_disp = args.max_disp
    model.cuda().eval()
    logging.info("Model loaded.")

    # --- Open camera ---
    cap = cv2.VideoCapture(args.cam_id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        logging.error(f"Cannot open /dev/video{args.cam_id}")
        return

    # Skip initial frames for auto-exposure
    for _ in range(10):
        cap.read()

    logging.info("Camera ready. Warming up model (first pass compiles CUDA kernels)...")

    # --- Warmup ---
    ret, frame = cap.read()
    if not ret:
        logging.error("Failed to capture warmup frame")
        return

    mid = frame.shape[1] // 2
    img_l = frame[:, :mid, :3]
    img_r = frame[:, mid:, :3]

    if rectify:
        img_l = cv2.remap(img_l, map_lx, map_ly, cv2.INTER_LINEAR)
        img_r = cv2.remap(img_r, map_rx, map_ry, cv2.INTER_LINEAR)

    img_l = cv2.resize(img_l, fx=args.scale, fy=args.scale, dsize=None)
    img_r = cv2.resize(img_r, dsize=(img_l.shape[1], img_l.shape[0]))

    t_l = torch.as_tensor(img_l).cuda().float()[None].permute(0, 3, 1, 2)
    t_r = torch.as_tensor(img_r).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(t_l.shape, divis_by=32, force_square=False)
    t_l, t_r = padder.pad(t_l, t_r)

    with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
        _ = model.forward(t_l, t_r, iters=args.valid_iters, test_mode=True, optimize_build_volume='pytorch1')

    logging.info("Warmup complete. Starting live inference...")

    # --- Load intrinsics for point cloud (from calibration or K.txt) ---
    K_for_depth = None
    baseline_for_depth = None
    if rectify:
        K_for_depth = K_rect.copy()
        baseline_for_depth = baseline
    elif args.intrinsic_file and os.path.exists(args.intrinsic_file):
        with open(args.intrinsic_file, 'r') as f:
            lines = f.readlines()
            K_for_depth = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3, 3)
            baseline_for_depth = float(lines[1])

    # --- Main loop ---
    frame_idx = 0
    fps_window = []
    print("\n" + "=" * 50)
    print("Live stereo depth estimation running.")
    print("Press 'q' or ESC to exit.")
    if rectify:
        print(f"Rectification: ON  | Baseline: {baseline_for_depth:.3f}m")
    else:
        print("Rectification: OFF (raw frames)")
    print(f"Scale: {args.scale}  |  valid_iters: {args.valid_iters}  |  max_disp: {args.max_disp}")
    print("=" * 50 + "\n")

    while True:
        t_start = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            logging.warning("Failed to capture frame")
            continue

        mid = frame.shape[1] // 2
        img_l = frame[:, :mid, :3]
        img_r = frame[:, mid:, :3]

        if rectify:
            img_l = cv2.remap(img_l, map_lx, map_ly, cv2.INTER_LINEAR)
            img_r = cv2.remap(img_r, map_rx, map_ry, cv2.INTER_LINEAR)

        H_ori, W_ori = img_l.shape[:2]

        if args.scale != 1.0:
            img_l = cv2.resize(img_l, fx=args.scale, fy=args.scale, dsize=None)
            img_r = cv2.resize(img_r, dsize=(img_l.shape[1], img_l.shape[0]))

        img_l_disp = img_l.copy()
        img_r_disp = img_r.copy()

        t_l = torch.as_tensor(img_l).cuda().float()[None].permute(0, 3, 1, 2)
        t_r = torch.as_tensor(img_r).cuda().float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(t_l.shape, divis_by=32, force_square=False)
        t_l, t_r = padder.pad(t_l, t_r)

        with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
            if not args.hiera:
                disp = model.forward(t_l, t_r, iters=args.valid_iters, test_mode=True, optimize_build_volume='pytorch1')
            else:
                disp = model.run_hierachical(t_l, t_r, iters=args.valid_iters, test_mode=True, small_ratio=0.5)

        disp = padder.unpad(disp.float())
        disp_np = disp.data.cpu().numpy().reshape(H_ori, W_ori).clip(0, None)

        # --- Compute depth if calibration available ---
        depth_m = None
        if K_for_depth is not None and baseline_for_depth is not None:
            depth_m = K_for_depth[0, 0] * baseline_for_depth / (disp_np.clip(0.1, None)) * args.depth_scale

        # --- Visualization ---
        vis = vis_disparity(disp_np, min_val=None, max_val=None, cmap=None, color_map=cv2.COLORMAP_TURBO)
        vis = np.concatenate([img_l_disp, img_r_disp, vis], axis=1)

        # FPS
        dt = time.perf_counter() - t_start
        fps_window.append(1.0 / dt if dt > 0 else 0)
        if len(fps_window) > 30:
            fps_window.pop(0)
        avg_fps = np.mean(fps_window)

        # Status line 1: FPS + model config
        cv2.putText(vis, f"FPS: {avg_fps:.1f}  |  iters={args.valid_iters}  scale={args.scale}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Status line 2: Depth info (center + near/far)
        if depth_m is not None:
            cy, cx = H_ori // 2, W_ori // 2
            center_d = depth_m[cy, cx]
            valid_d = depth_m[(depth_m > 0) & (depth_m < args.zfar)]
            near = valid_d.min() if len(valid_d) > 0 else 0
            far = valid_d.max() if len(valid_d) > 0 else 0
            cv2.putText(vis, f"Center: {center_d:.2f}m | Near: {near:.2f}m | Far: {far:.2f}m",
                        (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)

        if args.display:
            ds = 960 / vis.shape[1]
            vis_small = cv2.resize(vis, (int(vis.shape[1] * ds), int(vis.shape[0] * ds)))
            cv2.imshow('Fast-FoundationStereo Live', vis_small)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

        # --- Save output if requested ---
        if args.out_dir:
            cv2.imwrite(f'{args.out_dir}/{frame_idx:06d}_left.png', img_l_disp)
            cv2.imwrite(f'{args.out_dir}/{frame_idx:06d}_disp.png', vis)

        # --- Point cloud ---
        if args.get_pc and args.out_dir and K_for_depth is not None:
            if args.remove_invisible:
                yy, xx = np.meshgrid(np.arange(disp_np.shape[0]), np.arange(disp_np.shape[1]), indexing='ij')
                us_right = xx - disp_np
                invalid = us_right < 0
                disp_np_pc = disp_np.copy()
                disp_np_pc[invalid] = np.inf
            else:
                disp_np_pc = disp_np

            K_scaled = K_for_depth.copy()
            K_scaled[:2] *= args.scale
            depth = K_scaled[0, 0] * baseline_for_depth / disp_np_pc
            np.save(f'{args.out_dir}/{frame_idx:06d}_depth.npy', depth)
            xyz_map = depth2xyzmap(depth, K_scaled)
            pcd = toOpen3dCloud(xyz_map.reshape(-1, 3), img_l_disp.reshape(-1, 3))
            keep_mask = (np.asarray(pcd.points)[:, 2] > 0) & (np.asarray(pcd.points)[:, 2] <= args.zfar)
            keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
            pcd = pcd.select_by_index(keep_ids)
            import open3d as o3d
            o3d.io.write_point_cloud(f'{args.out_dir}/{frame_idx:06d}_cloud.ply', pcd)

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    logging.info(f"Finished. {frame_idx} frames processed. Avg FPS: {avg_fps:.1f}")


if __name__ == '__main__':
    main()
