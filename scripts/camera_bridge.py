#!/usr/bin/env python3
"""
Camera bridge: captures stereo frames from GEAC IMX185 cameras via GStreamer,
optionally applies VPI-accelerated stereo rectification, and writes to /dev/shm.

GPU-side scaling via nvvidconv (VIC hardware) avoids expensive CPU resize.
VPI remap (VIC/CUDA) offloads stereo rectification from CPU.

Usage:
    # Without rectification (raw frames):
    python scripts/camera_bridge.py --left_sensor 0 --right_sensor 1 --scale 0.5

    # With VPI rectification (rectified frames written to SHM):
    python scripts/camera_bridge.py --left_sensor 0 --right_sensor 1 --scale 0.5 \\
        --calib_file calibrations/stereo_calib.npz
"""

import argparse
import logging
import mmap
import os
import struct
import time
import numpy as np

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

Gst.init(None)

SHM_META_PATH = "/dev/shm/stereo_meta"
SHM_LEFT_PATH  = "/dev/shm/stereo_left"
SHM_RIGHT_PATH = "/dev/shm/stereo_right"

META_FORMAT = "=QQIIIIB7x"
META_SIZE   = struct.calcsize(META_FORMAT)

FLAG_LEFT_READY    = 0x01
FLAG_RIGHT_READY   = 0x02
FLAG_STEREO_READY  = 0x04
FLAG_RECTIFIED     = 0x08


def _load_calib_maps(calib_path):
    """Load OpenCV rectification maps from stereo_calib.npz."""
    data = np.load(calib_path)
    K_l, D_l = data['K_left'], data['D_left']
    K_r, D_r = data['K_right'], data['D_right']
    R1, R2  = data['R1'], data['R2']
    P1, P2  = data['P1'], data['P2']
    calib_w, calib_h = tuple(data['image_size'])

    import cv2
    map_lx, map_ly = cv2.initUndistortRectifyMap(K_l, D_l, R1, P1, (calib_w, calib_h), cv2.CV_32FC1)
    map_rx, map_ry = cv2.initUndistortRectifyMap(K_r, D_r, R2, P2, (calib_w, calib_h), cv2.CV_32FC1)
    return (map_lx, map_ly), (map_rx, map_ry), (calib_w, calib_h)


def _make_vpi_warp_map(map_x, map_y):
    """Convert OpenCV remap to VPI WarpMap (CUDA backend)."""
    import vpi
    h, w = map_x.shape
    grid = vpi.WarpGrid((w, h))
    wmap = vpi.WarpMap(grid)
    warp_data = np.asarray(wmap)
    warp_data[:h, :w, 0] = map_x
    warp_data[:h, :w, 1] = map_y
    return wmap


class StereoCameraBridge:
    def __init__(self, left_sensor=0, right_sensor=1,
                 capture_w=1920, capture_h=1200, fps=30, scale=1.0,
                 calib_file=None):
        self.input_w  = capture_w
        self.input_h  = capture_h
        self.output_w = max(1, int(capture_w * scale))
        self.output_h = max(1, int(capture_h * scale))
        self.fps      = fps
        self.frame_id = 0
        self.running  = False

        # --- VPI rectification ---
        self.do_remap = calib_file is not None and os.path.exists(calib_file)
        if self.do_remap:
            import vpi
            (map_lx, map_ly), (map_rx, map_ry), (calib_w, calib_h) = _load_calib_maps(calib_file)
            if (calib_w, calib_h) != (self.output_w, self.output_h):
                logging.warning(
                    f"Calibration size {calib_w}x{calib_h} != bridge output {self.output_w}x{self.output_h}. "
                    f"Adjust --scale to match calibration."
                )
            self.warp_l = _make_vpi_warp_map(map_lx, map_ly)
            self.warp_r = _make_vpi_warp_map(map_rx, map_ry)
            self._remap_flags = FLAG_RECTIFIED
            logging.info(f"VPI rectification enabled (CUDA backend)")
        else:
            self._remap_flags = 0

        self._init_shm()
        self.left_pipe  = self._build_pipeline(left_sensor,  "left")
        self.right_pipe = self._build_pipeline(right_sensor, "right")
        self.left_new   = False
        self.right_new  = False
        self.loop = GLib.MainLoop()

    # ==================================================================
    # Shared memory
    # ==================================================================
    def _init_shm(self):
        frame_size = self.output_w * self.output_h * 3
        init_meta = struct.pack(META_FORMAT, 0, 0, self.output_w, self.output_h,
                                frame_size, frame_size, 0)
        for path, data, size in [(SHM_META_PATH,  init_meta,              META_SIZE),
                                  (SHM_LEFT_PATH,  b'\x00' * frame_size, frame_size),
                                  (SHM_RIGHT_PATH, b'\x00' * frame_size, frame_size)]:
            try:
                os.unlink(path)
            except OSError:
                pass
            with open(path, "wb") as f:
                f.write(data)

        self.meta_fd  = os.open(SHM_META_PATH, os.O_RDWR)
        self.meta_mm  = mmap.mmap(self.meta_fd, META_SIZE, mmap.MAP_SHARED)
        self.left_fd  = os.open(SHM_LEFT_PATH, os.O_RDWR)
        self.left_mm  = mmap.mmap(self.left_fd, frame_size, mmap.MAP_SHARED)
        self.right_fd = os.open(SHM_RIGHT_PATH, os.O_RDWR)
        self.right_mm = mmap.mmap(self.right_fd, frame_size, mmap.MAP_SHARED)

    # ==================================================================
    # GStreamer pipeline
    # ==================================================================
    def _build_pipeline(self, sensor_id, name):
        pipeline_str = (
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM),width={self.input_w},height={self.input_h},"
            f"framerate={self.fps}/1 ! "
            f"nvvidconv ! video/x-raw,format=BGRx,"
            f"width={self.output_w},height={self.output_h} ! "
            f"appsink name=sink emit-signals=true sync=false max-buffers=2 drop=true"
        )
        pipe = Gst.parse_launch(pipeline_str)
        sink = pipe.get_by_name("sink")
        sink.connect("new-sample", self._on_new_sample, name)
        return pipe

    # ==================================================================
    # Frame callback
    # ==================================================================
    def _on_new_sample(self, sink, cam_name):
        sample = sink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.ERROR

        buf = sample.get_buffer()
        caps = sample.get_caps()
        struct_info = caps.get_structure(0)
        w = struct_info.get_value("width")
        h = struct_info.get_value("height")

        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR

        raw = np.frombuffer(map_info.data, dtype=np.uint8).reshape(h, w, 4)
        bgr = raw[:, :, :3]  # BGRx -> BGR

        # --- VPI rectification ---
        if self.do_remap:
            bgr = self._vpi_remap(bgr, cam_name)

        mm = self.left_mm if cam_name == "left" else self.right_mm
        mm.seek(0)
        mm.write(bgr.tobytes())

        if cam_name == "left":
            self.left_new = True
        else:
            self.right_new = True

        buf.unmap(map_info)

        # Once both frames arrive, publish metadata
        if self.left_new and self.right_new:
            self.frame_id += 1
            ts_ns = int(time.time() * 1e9)
            frame_size = self.output_w * self.output_h * 3
            flags = FLAG_LEFT_READY | FLAG_RIGHT_READY | FLAG_STEREO_READY | self._remap_flags
            meta = struct.pack(META_FORMAT,
                               self.frame_id, ts_ns,
                               self.output_w, self.output_h,
                               frame_size, frame_size,
                               flags)
            self.meta_mm.seek(0)
            self.meta_mm.write(meta)
            self.left_new  = False
            self.right_new = False

        return Gst.FlowReturn.OK

    def _vpi_remap(self, bgr_np, cam_name):
        """Run VPI remap on CUDA backend and return numpy BGR array."""
        import vpi
        warp = self.warp_l if cam_name == "left" else self.warp_r
        with vpi.Backend.CUDA:
            vin = vpi.asimage(bgr_np)
            vout = vin.remap(warp, interp=vpi.Interp.LINEAR)
            with vout.rlock_cpu() as data:
                result = np.asarray(data).copy()
        return result[:self.output_h, :self.output_w, :3]

    # ==================================================================
    # Lifecycle
    # ==================================================================
    def start(self):
        info = f"{self.input_w}x{self.input_h} -> {self.output_w}x{self.output_h} @ {self.fps}fps"
        if self.do_remap:
            info += " (VPI rectified)"
        logging.info(f"Starting camera bridge: {info}")
        self.left_pipe.set_state(Gst.State.PLAYING)
        self.right_pipe.set_state(Gst.State.PLAYING)
        self.running = True
        logging.info("Camera bridge running. Press Ctrl+C to stop.")
        try:
            self.loop.run()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        self.running = False
        for p in [self.left_pipe, self.right_pipe]:
            p.set_state(Gst.State.NULL)
        self.loop.quit()
        for m in [self.meta_mm, self.left_mm, self.right_mm]:
            m.close()
        for f in [self.meta_fd, self.left_fd, self.right_fd]:
            os.close(f)
        logging.info("Camera bridge stopped.")


# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="Stereo Camera Bridge (GStreamer -> shared memory)")
    parser.add_argument("--left_sensor",  type=int,   default=0)
    parser.add_argument("--right_sensor", type=int,   default=1)
    parser.add_argument("--width",        type=int,   default=1920)
    parser.add_argument("--height",       type=int,   default=1200)
    parser.add_argument("--fps",          type=int,   default=30)
    parser.add_argument("--scale",        type=float, default=1.0)
    parser.add_argument("--calib_file",   type=str,   default=None,
                        help="stereo_calib.npz path; if set, applies VPI rectification")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    bridge = StereoCameraBridge(
        left_sensor=args.left_sensor,
        right_sensor=args.right_sensor,
        capture_w=args.width,
        capture_h=args.height,
        fps=args.fps,
        scale=args.scale,
        calib_file=args.calib_file,
    )
    bridge.start()


if __name__ == "__main__":
    main()
