"""
Shared memory stereo camera reader.

Used inside Docker to read frames captured by camera_bridge.py (running on host).
"""

import ctypes
import mmap
import os
import struct
import time
import numpy as np

SHM_META_PATH = "/dev/shm/stereo_meta"
SHM_LEFT_PATH = "/dev/shm/stereo_left"
SHM_RIGHT_PATH = "/dev/shm/stereo_right"

META_FORMAT = "=QQIIIIB7x"
META_SIZE = struct.calcsize(META_FORMAT)

FLAG_LEFT_READY = 0x01
FLAG_RIGHT_READY = 0x02
FLAG_STEREO_READY = 0x04


class SharedMemoryStereoReader:
    """Reads stereo frames from shared memory populated by camera_bridge.py."""

    def __init__(self):
        self.meta_fd = None
        self.left_fd = None
        self.right_fd = None
        self.meta_mm = None
        self.left_mm = None
        self.right_mm = None
        self.last_frame_id = -1
        self.width = 0
        self.height = 0
        self._open()

    def _open(self):
        # Wait for shared memory to be created
        timeout = 10.0
        start = time.time()
        while not os.path.exists(SHM_META_PATH):
            if time.time() - start > timeout:
                raise RuntimeError(
                    f"Shared memory {SHM_META_PATH} not found. "
                    f"Is camera_bridge.py running on the host?"
                )
            time.sleep(0.1)

        self.meta_fd = os.open(SHM_META_PATH, os.O_RDONLY)
        self.meta_mm = mmap.mmap(self.meta_fd, META_SIZE, mmap.MAP_SHARED, mmap.PROT_READ)
        self.left_fd = os.open(SHM_LEFT_PATH, os.O_RDONLY)
        self.right_fd = os.open(SHM_RIGHT_PATH, os.O_RDONLY)

        # Read header to determine frame size
        meta = self._read_meta()
        self.width = meta["width"]
        self.height = meta["height"]

        # Now mmap the frame buffers with correct size
        frame_size = self.width * self.height * 3
        self.left_mm = mmap.mmap(self.left_fd, frame_size, mmap.MAP_SHARED, mmap.PROT_READ)
        self.right_mm = mmap.mmap(self.right_fd, frame_size, mmap.MAP_SHARED, mmap.PROT_READ)

    def _read_meta(self):
        self.meta_mm.seek(0)
        data = self.meta_mm.read(META_SIZE)
        vals = struct.unpack(META_FORMAT, data)
        return {
            "frame_id": vals[0],
            "timestamp_ns": vals[1],
            "width": vals[2],
            "height": vals[3],
            "left_size": vals[4],
            "right_size": vals[5],
            "flags": vals[6],
        }

    def read(self, timeout=5.0):
        """Wait for a new stereo frame pair and return (left_bgr, right_bgr) as numpy arrays."""
        start = time.time()
        while True:
            meta = self._read_meta()
            if (meta["flags"] & FLAG_STEREO_READY
                    and meta["frame_id"] > self.last_frame_id
                    and meta["width"] > 0 and meta["height"] > 0):
                break
            if time.time() - start > timeout:
                return None, None
            time.sleep(0.002)  # 2ms polling interval

        self.last_frame_id = meta["frame_id"]
        frame_size = meta["left_size"]

        self.left_mm.seek(0)
        left_data = self.left_mm.read(frame_size)
        self.right_mm.seek(0)
        right_data = self.right_mm.read(frame_size)

        left = np.frombuffer(left_data, dtype=np.uint8).reshape(meta["height"], meta["width"], 3).copy()
        right = np.frombuffer(right_data, dtype=np.uint8).reshape(meta["height"], meta["width"], 3).copy()

        return left, right

    def close(self):
        for mm in [self.meta_mm, self.left_mm, self.right_mm]:
            if mm:
                mm.close()
        for fd in [self.meta_fd, self.left_fd, self.right_fd]:
            if fd is not None:
                os.close(fd)


def create_stereo_reader():
    """Factory function: returns None if shared memory is not available."""
    try:
        return SharedMemoryStereoReader()
    except (RuntimeError, FileNotFoundError, OSError) as e:
        import logging
        logging.warning(f"Shared memory camera not available: {e}")
        return None
