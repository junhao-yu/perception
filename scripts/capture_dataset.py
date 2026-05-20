"""
Capture frames from the stereo camera (left eye) for dataset annotation.

Usage:
  python scripts/capture_dataset.py \
      --calib_file calibrations/stereo_calib.npz \
      --out_dir dataset/cups_bottles/ \
      --cam_id 2
"""

import argparse, os, sys, time
import numpy as np
import cv2

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')


def load_calibration(npz_path):
    data = np.load(npz_path)
    K_l, D_l = data['K_left'], data['D_left']
    K_r, D_r = data['K_right'], data['D_right']
    R1, R2 = data['R1'], data['R2']
    P1, P2 = data['P1'], data['P2']
    w, h = tuple(data['image_size'])
    map_lx, map_ly = cv2.initUndistortRectifyMap(K_l, D_l, R1, P1, (w, h), cv2.CV_32FC1)
    map_rx, map_ry = cv2.initUndistortRectifyMap(K_r, D_r, R2, P2, (w, h), cv2.CV_32FC1)
    return map_lx, map_ly, map_rx, map_ry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--calib_file', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default='dataset/cups_bottles/')
    parser.add_argument('--cam_id', default=2, type=int)
    parser.add_argument('--cam_width', default=1280, type=int)
    parser.add_argument('--cam_height', default=480, type=int)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    map_lx, map_ly, map_rx, map_ry = load_calibration(args.calib_file)

    cap = cv2.VideoCapture(args.cam_id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    for _ in range(10):
        cap.read()

    idx = 0
    existing = [f for f in os.listdir(args.out_dir) if f.endswith('.jpg')]
    if existing:
        nums = []
        for f in existing:
            try:
                nums.append(int(''.join(c for c in f if c.isdigit())))
            except:
                pass
        idx = max(nums) + 1 if nums else len(existing)

    print(f"Capturing to: {args.out_dir}")
    print(f"Starting index: {idx}")
    print("SPACE = capture  |  ESC/q = quit")
    print("=" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        mid = frame.shape[1] // 2
        img_l = frame[:, :mid, :3]
        img_l = cv2.remap(img_l, map_lx, map_ly, cv2.INTER_LINEAR)
        display = img_l.copy()
        cv2.putText(display, f"Count: {idx}  [SPACE=capture ESC=quit]",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Dataset Capture', display)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            path = os.path.join(args.out_dir, f'frame_{idx:05d}.jpg')
            cv2.imwrite(path, img_l)
            print(f"  Saved: {path}")
            idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nCaptured {idx} frames to {args.out_dir}")


if __name__ == '__main__':
    main()
