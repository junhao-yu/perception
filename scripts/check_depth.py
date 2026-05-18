"""
Quick depth accuracy checker: shows depth at image center continuously.
Place an object at a known distance, read the displayed depth, compare.

Usage:
  python scripts/check_depth.py \
      --model_dir weights/20-30-48/model_best_bp2_serialize.pth \
      --calib_file calibrations/stereo_calib.npz \
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
    return map_lx, map_ly, map_rx, map_ry, P1[:3, :3].copy(), baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--calib_file', type=str, required=True)
    parser.add_argument('--cam_id', default=2, type=int)
    parser.add_argument('--scale', default=1.0, type=float)
    parser.add_argument('--valid_iters', default=4, type=int)
    parser.add_argument('--max_disp', default=192, type=int)
    parser.add_argument('--depth_scale', default=1.0, type=float,
                        help='Scale correction factor for depth')
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)

    map_lx, map_ly, map_rx, map_ry, K, baseline = load_calibration(args.calib_file)
    logging.info(f"Calibration loaded: baseline={baseline:.3f}m, fx={K[0,0]:.1f}")

    model = torch.load(args.model_dir, map_location='cpu', weights_only=False)
    model.args.valid_iters = args.valid_iters
    model.args.max_disp = args.max_disp
    model.cuda().eval()

    cap = cv2.VideoCapture(args.cam_id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    for _ in range(10):
        cap.read()

    # Warmup
    ret, frame = cap.read()
    mid = frame.shape[1] // 2
    img_l = frame[:, :mid, :3]
    img_r = frame[:, mid:, :3]
    img_l = cv2.remap(img_l, map_lx, map_ly, cv2.INTER_LINEAR)
    img_r = cv2.remap(img_r, map_rx, map_ry, cv2.INTER_LINEAR)
    t_l = torch.as_tensor(img_l).cuda().float()[None].permute(0, 3, 1, 2)
    t_r = torch.as_tensor(img_r).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(t_l.shape, divis_by=32, force_square=False)
    t_l, t_r = padder.pad(t_l, t_r)
    with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
        _ = model.forward(t_l, t_r, iters=args.valid_iters, test_mode=True, optimize_build_volume='pytorch1')

    logging.info("Warmup done. Point camera at objects at known distances.")
    print("\n" + "=" * 50)
    print("Depth Checker: center depth shown in meters")
    print("Press 'q' to quit")
    print("=" * 50 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        mid = frame.shape[1] // 2
        img_l = frame[:, :mid, :3]
        img_r = frame[:, mid:, :3]
        img_l = cv2.remap(img_l, map_lx, map_ly, cv2.INTER_LINEAR)
        img_r = cv2.remap(img_r, map_rx, map_ry, cv2.INTER_LINEAR)
        H, W = img_l.shape[:2]

        t_l = torch.as_tensor(img_l).cuda().float()[None].permute(0, 3, 1, 2)
        t_r = torch.as_tensor(img_r).cuda().float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(t_l.shape, divis_by=32, force_square=False)
        t_l, t_r = padder.pad(t_l, t_r)
        with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
            disp = model.forward(t_l, t_r, iters=args.valid_iters, test_mode=True, optimize_build_volume='pytorch1')
        disp_np = padder.unpad(disp.float()).data.cpu().numpy().reshape(H, W).clip(0, None)
        depth = K[0, 0] * baseline / disp_np.clip(0.1, None) * args.depth_scale

        # Show depth at center + crosshair
        cy, cx = H // 2, W // 2
        center_d = depth[cy, cx]
        disp_vis = vis_disparity(disp_np, min_val=None, max_val=None, cmap=None, color_map=cv2.COLORMAP_TURBO)
        display = np.concatenate([img_l, disp_vis], axis=1)
        cv2.drawMarker(display, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
        cv2.putText(display, f"Center: {center_d:.3f}m", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, f"fx={K[0,0]:.1f}  baseline={baseline:.4f}m",
                    (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        ds = 960 / display.shape[1]
        cv2.imshow('Depth Checker (red crosshair = measured point)', cv2.resize(display, (int(display.shape[1]*ds), int(display.shape[0]*ds))))
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
