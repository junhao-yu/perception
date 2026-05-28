"""
Stereo camera calibration tool for DECXIN-2784V1 (and other UVC stereo cameras)
and GEAC IMX185 via camera bridge.

Captures chessboard images from both eyes simultaneously and computes:
  - Individual camera intrinsics (K_left, K_right)
  - Distortion coefficients (D_left, D_right)
  - Stereo extrinsics (R, T)
  - Rectification transforms (R1, R2, P1, P2, Q)

Usage (old USB camera):
  python scripts/calibrate_stereo.py --cam_id 2 --width 1280 --height 480

Usage (GEAC IMX185 via camera bridge, run camera_bridge.py on host first):
  python scripts/calibrate_stereo.py --cam_mode shm
"""

import argparse
import os
import sys
import numpy as np
import cv2

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)
from shm_camera import SharedMemoryStereoReader


CHESSBOARD = (9, 6)  # inner corners (cols, rows)
SQUARE_SIZE = 0.025  # meters — measure your chessboard square size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default='calibrations', type=str)
    parser.add_argument('--cam_mode', type=str, default='v4l2', choices=['v4l2', 'shm'],
                        help='v4l2 = single side-by-side UVC camera; shm = stereo pair via camera_bridge.py')
    parser.add_argument('--cam_id', default=2, type=int, help='V4L2 device index (--cam_mode v4l2 only)')
    parser.add_argument('--width', default=1280, type=int, help='Combined frame width (--cam_mode v4l2 only)')
    parser.add_argument('--height', default=480, type=int, help='Frame height (--cam_mode v4l2 only)')
    parser.add_argument('--chessboard_cols', default=9, type=int)
    parser.add_argument('--chessboard_rows', default=6, type=int)
    parser.add_argument('--square_size', default=0.025, type=float, help='Checkerboard square size in meters')
    parser.add_argument('--max_frames', default=50, type=int, help='Max calibration frame pairs to collect')
    args = parser.parse_args()

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    pattern = (args.chessboard_cols, args.chessboard_rows)

    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * args.square_size

    objpoints = []
    imgpoints_l = []
    imgpoints_r = []

    os.makedirs(args.out_dir, exist_ok=True)

    if args.cam_mode == 'shm':
        print("Opening shared memory stereo camera...")
        cap = None
        shm_reader = SharedMemoryStereoReader()
        cam_w, cam_h = shm_reader.width, shm_reader.height
        print(f"Camera: {cam_w}x{cam_h} (via camera bridge)")
    else:
        cap = cv2.VideoCapture(args.cam_id, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            print(f"ERROR: Cannot open /dev/video{args.cam_id}")
            return
        shm_reader = None

    def read_frames():
        if shm_reader is not None:
            return shm_reader.read()
        else:
            ret, frame = cap.read()
            if not ret:
                return None, None
            mid = frame.shape[1] // 2
            return frame[:, :mid], frame[:, mid:]

    collected = 0
    print(f"Collecting calibration frames. Press SPACE to capture, ESC to finish.")
    print(f"Min frames needed: ~15. Captured: 0")

    while collected < args.max_frames:
        img_l, img_r = read_frames()
        if img_l is None:
            continue

        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        ret_l, corners_l = cv2.findChessboardCorners(gray_l, pattern, None)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, pattern, None)

        display = np.concatenate([img_l, img_r], axis=1)
        if ret_l and ret_r:
            cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
            cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
            mid = img_l.shape[1]
            cv2.drawChessboardCorners(display[:, :mid], pattern, corners_l, ret_l)
            cv2.drawChessboardCorners(display[:, mid:], pattern, corners_r, ret_r)
            status = "READY - press SPACE to capture"
        else:
            status = f"Left: {'OK' if ret_l else 'not found'} | Right: {'OK' if ret_r else 'not found'}"

        cv2.putText(display, f"Captured: {collected}  {status}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        s = 960 / display.shape[1]
        display_small = cv2.resize(display, (int(display.shape[1] * s), int(display.shape[0] * s)))
        cv2.imshow('Stereo Calibration', display_small)
        key = cv2.waitKey(10) & 0xFF

        if key == 32 and ret_l and ret_r:  # SPACE
            objpoints.append(objp)
            imgpoints_l.append(corners_l)
            imgpoints_r.append(corners_r)
            collected += 1
            print(f"Captured frame pair {collected}/{args.max_frames}")
        elif key == 27:  # ESC
            break

    if cap is not None:
        cap.release()
    if shm_reader is not None:
        shm_reader.close()
    cv2.destroyAllWindows()

    if collected < 10:
        print(f"ERROR: Need at least 10 frame pairs, got {collected}")
        return

    print(f"\nCollected {collected} frame pairs. Running calibration...")

    h, w = gray_l.shape

    # Calibrate each camera individually
    ret_l, K_l, D_l, _, _ = cv2.calibrateCamera(objpoints, imgpoints_l, (w, h), None, None)
    ret_r, K_r, D_r, _, _ = cv2.calibrateCamera(objpoints, imgpoints_r, (w, h), None, None)

    # Stereo calibration
    stereo_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    ret_s, K_l, D_l, K_r, D_r, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r,
        K_l, D_l, K_r, D_r, (w, h),
        criteria=stereo_criteria, flags=cv2.CALIB_FIX_INTRINSIC
    )

    # Rectification
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K_l, D_l, K_r, D_r, (w, h), R, T, alpha=0)

    # Save calibration
    calib = {
        'K_left': K_l, 'D_left': D_l,
        'K_right': K_r, 'D_right': D_r,
        'R': R, 'T': T,
        'R1': R1, 'R2': R2,
        'P1': P1, 'P2': P2,
        'Q': Q,
        'image_size': (w, h),
        'baseline_m': abs(T[0]),
    }

    npz_path = os.path.join(args.out_dir, 'stereo_calib.npz')
    np.savez(npz_path, **calib)
    print(f"Calibration saved to {npz_path}")

    # Also save the intrinsics file in Fast-FoundationStereo format (K_left + baseline)
    K_file = os.path.join(args.out_dir, 'K.txt')
    with open(K_file, 'w') as f:
        f.write(' '.join(map(str, P1[:3].flatten())) + '\n')
        f.write(f'{abs(float(T[0]))}\n')
    print(f"Intrinsics (rectified) saved to {K_file}")

    print(f"\n=== Calibration Results ===")
    print(f"RMS reprojection error: {ret_s:.4f}")
    print(f"Baseline: {abs(float(T[0])):.4f} m")
    print(f"K_left (original):\n{K_l}")
    print(f"P1 (rectified):\n{P1}")

    # Show rectification preview
    print("\nShowing rectification preview. Press any key to exit.")
    map_lx, map_ly = cv2.initUndistortRectifyMap(K_l, D_l, R1, P1, (w, h), cv2.CV_32FC1)
    map_rx, map_ry = cv2.initUndistortRectifyMap(K_r, D_r, R2, P2, (w, h), cv2.CV_32FC1)

    # Reopen camera for rectification preview
    if args.cam_mode == 'shm':
        preview_reader = SharedMemoryStereoReader()
        print("Press any key in the preview window to exit.")
        img_l, img_r = preview_reader.read()
    else:
        preview_reader = None
        cap2 = cv2.VideoCapture(args.cam_id, cv2.CAP_V4L2)
        cap2.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        for _ in range(5):
            cap2.read()
        print("Press any key in the preview window to exit.")
        ret, frame = cap2.read()
        if ret:
            mid_px = frame.shape[1] // 2
            img_l = frame[:, :mid_px]
            img_r = frame[:, mid_px:]
        else:
            img_l = img_r = None

    if img_l is not None:

        rect_l = cv2.remap(img_l, map_lx, map_ly, cv2.INTER_LINEAR)
        rect_r = cv2.remap(img_r, map_rx, map_ry, cv2.INTER_LINEAR)

        # Draw horizontal epipolar lines
        for y in range(0, h, 40):
            cv2.line(rect_l, (0, y), (w, y), (0, 255, 0), 1)
            cv2.line(rect_r, (0, y), (w, y), (0, 255, 0), 1)

        preview = np.concatenate([rect_l, rect_r], axis=1)
        s = 960 / preview.shape[1]
        preview = cv2.resize(preview, (int(preview.shape[1] * s), int(preview.shape[0] * s)))
        cv2.imshow('Rectification Check (lines should align)', preview)
        cv2.waitKey(0)

    if preview_reader is not None:
        preview_reader.close()
    if 'cap2' in dir():
        cap2.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
