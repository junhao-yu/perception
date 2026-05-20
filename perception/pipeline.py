import time
import numpy as np
import cv2
import torch

from detectors import Detection, BaseDetector
from core.utils.utils import InputPadder
from Utils import AMP_DTYPE, vis_disparity


COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
    (0, 128, 255), (128, 0, 255),
]


class PerceptionPipeline:
    """Detection + stereo depth → 3D object positions."""

    def __init__(self, detector: BaseDetector, ffs_model,
                 calib: dict, depth_scale: float = 1.0,
                 image_scale: float = 1.0):
        self.detector = detector
        self.ffs_model = ffs_model
        self.calib = calib
        self.depth_scale = depth_scale
        self.image_scale = image_scale
        self.last_timing = {}

    def process_frame(self, img_l: np.ndarray, img_r: np.ndarray):
        """Run detection and stereo on a rectified stereo pair.

        Returns:
            disp_vis:     disparity colormap (H, W, 3)
            depth_m:      depth map in meters (H, W)
            detections_3d: list of (detection, pos_3d) tuples
        """
        H, W = img_l.shape[:2]
        t_total = time.perf_counter()

        # --- Detection on left image ---
        t0 = time.perf_counter()
        detections = self.detector.detect(img_l)
        t_detect = time.perf_counter() - t0

        # --- FFS stereo depth ---
        t0 = time.perf_counter()
        t_l = torch.as_tensor(img_l).cuda().float()[None].permute(0, 3, 1, 2)
        t_r = torch.as_tensor(img_r).cuda().float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(t_l.shape, divis_by=32, force_square=False)
        t_l, t_r = padder.pad(t_l, t_r)

        with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
            disp = self.ffs_model.forward(t_l, t_r, iters=self.ffs_model.args.valid_iters,
                                          test_mode=True, optimize_build_volume='pytorch1')
        disp = padder.unpad(disp.float())
        disp_np = disp.data.cpu().numpy().reshape(H, W).clip(0, None)
        t_ffs = time.perf_counter() - t0

        # Depth — scale K to match image resolution
        t0_d = time.perf_counter()
        K = self.calib['K'].copy()
        s = self.image_scale
        if s != 1.0:
            K[0, 0] *= s  # fx
            K[1, 1] *= s  # fy
            K[0, 2] *= s  # cx
            K[1, 2] *= s  # cy
        baseline = self.calib['baseline']
        depth_m = K[0, 0] * baseline / disp_np.clip(0.1, None) * self.depth_scale

        # Visualize disparity
        disp_vis = vis_disparity(disp_np, min_val=None, max_val=None, cmap=None,
                                 color_map=cv2.COLORMAP_TURBO)
        t_depth = time.perf_counter() - t0_d

        # --- Fuse: bbox → depth → 3D ---
        t0_fuse = time.perf_counter()
        detections_3d = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            d = self._depth_at_bbox(depth_m, (x1, y1, x2, y2))
            if d is not None:
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                pos_3d = self._pixel_to_camera(cx, cy, d, K)
                detections_3d.append((det, pos_3d))
        t_fuse = time.perf_counter() - t0_fuse

        self.last_timing = {
            'detect': t_detect * 1000,
            'ffs': t_ffs * 1000,
            'depth_vis': t_depth * 1000,
            'fuse': t_fuse * 1000,
            'total': (time.perf_counter() - t_total) * 1000,
        }

        return disp_vis, depth_m, detections_3d

    @staticmethod
    def _depth_at_bbox(depth_map, bbox, margin_ratio=0.15):
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

    @staticmethod
    def _pixel_to_camera(px, py, depth, K):
        return np.array([(px - K[0, 2]) * depth / K[0, 0],
                         (py - K[1, 2]) * depth / K[1, 1],
                         depth])
