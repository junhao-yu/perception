#!/bin/bash
#
# Run Fast-FoundationStereo container on Jetson T4000
#
# Prerequisites (run once on host before first use):
#   1. Initialize GEAC camera:  cd /home/nvidia/camera_demo_geac && sudo bash camera_init.sh
#   2. Start camera bridge:     python scripts/camera_bridge.py --scale 0.5 &
#
# Usage:
#   bash docker/run_jetson.sh [command to run in container]
#
# Examples:
#   bash docker/run_jetson.sh  (opens interactive shell)
#   bash docker/run_jetson.sh python scripts/run_perception.py --cam_mode shm ...

set -e

IMAGE_NAME="ffs-jetson"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

docker rm -f ffs-jetson 2>/dev/null || true

xhost +local: 2>/dev/null || true

if [ $# -eq 0 ]; then
    set -- bash
fi

docker run --rm -it \
    --runtime nvidia \
    --network host \
    --ipc host \
    -e DISPLAY="${DISPLAY}" \
    -e NVIDIA_DISABLE_REQUIRE=1 \
    -e TORCH_HOME=/workspace/cache/torch \
    -e YOLO_CONFIG_DIR=/workspace/cache/ultralytics \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /tmp:/tmp \
    -v /dev/shm:/dev/shm \
    -v "${PROJECT_DIR}/weights:/workspace/weights" \
    -v "${PROJECT_DIR}/models:/workspace/models" \
    -v "${PROJECT_DIR}/calibrations:/workspace/calibrations" \
    -v "${PROJECT_DIR}/demo_data:/workspace/demo_data" \
    -v "${PROJECT_DIR}/output:/workspace/output" \
    -v "${PROJECT_DIR}/scripts:/workspace/scripts" \
    -v "${PROJECT_DIR}/cache:/workspace/cache" \
    -v /dev:/dev \
    "$IMAGE_NAME" \
    "$@"

xhost -local: 2>/dev/null || true
