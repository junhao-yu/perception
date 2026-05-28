#!/bin/bash
#
# One-time camera setup script for GEAC IMX185 stereo cameras.
# Run this ONCE on the host after each boot.
#
# This script:
#   1. Initializes the GEAC camera hardware (max96712 + IMX185 sensors)
#   2. Restarts nvargus-daemon with GEAC-specific configuration
#   3. Starts camera_bridge.py with VPI rectification to feed Docker via /dev/shm
#
# Usage:
#   bash scripts/setup_camera.sh [options]
#
# Options:
#   --left L        Left camera sensor-id (default: 0)
#   --right R       Right camera sensor-id (default: 1)
#   --scale S       Output scale factor (default: 0.5 → 960x600)
#   --calib FILE    Calibration file for VPI rectification
#   --raw           Skip VPI rectification (raw frames, no --calib_file)
#

set -e

GEAC_DIR="/home/nvidia/camera_demo_geac"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

LEFT_SENSOR=0
RIGHT_SENSOR=1
WIDTH=1920
HEIGHT=1200
FPS=30
SCALE=0.5
CALIB_FILE="$PROJECT_DIR/calibrations/stereo_calib.npz"
DO_REMAP=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --left)   LEFT_SENSOR="$2"; shift 2 ;;
        --right)  RIGHT_SENSOR="$2"; shift 2 ;;
        --width)  WIDTH="$2"; shift 2 ;;
        --height) HEIGHT="$2"; shift 2 ;;
        --fps)    FPS="$2"; shift 2 ;;
        --scale)  SCALE="$2"; shift 2 ;;
        --calib)  CALIB_FILE="$2"; shift 2 ;;
        --raw)    DO_REMAP=0; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "============================================"
echo " GEAC Stereo Camera Setup"
echo "============================================"
echo " Left sensor:  $LEFT_SENSOR"
echo " Right sensor: $RIGHT_SENSOR"
echo " Resolution:   ${WIDTH}x${HEIGHT} @ ${FPS}fps"
echo " Output scale: ${SCALE}"
if [ "$DO_REMAP" -eq 1 ] && [ -f "$CALIB_FILE" ]; then
    echo " VPI rectification: $CALIB_FILE"
else
    echo " VPI rectification: OFF (raw frames)"
fi
echo ""

# ==================================================================
# Step 1: Initialize GEAC camera hardware
# ==================================================================
echo "[Step 1/4] Initializing GEAC camera hardware..."
if [ -d "$GEAC_DIR" ]; then
    cd "$GEAC_DIR"
    sudo bash rb_camera.sh init
    echo "  GEAC camera init done."
else
    echo "  WARNING: GEAC dir not found at $GEAC_DIR — skipping hardware init."
fi

# ==================================================================
# Step 2: Restart nvargus-daemon
# ==================================================================
echo "[Step 2/4] Restarting nvargus-daemon..."
pkill -f camera_bridge.py 2>/dev/null || true
sudo pkill nvargus-daemon 2>/dev/null || true
sleep 1
sudo rm -f /tmp/argus_socket

echo "  Starting nvargus-daemon with GEAC config..."
sudo NVCAMERA_NITO_PATH=CONFIG enableCamInfiniteTimeout=1 \
    nohup nvargus-daemon > /tmp/nvargus.log 2>&1 &
sleep 3

# Verify daemon is running
if pgrep -f nvargus-daemon > /dev/null; then
    echo "  nvargus-daemon PID: $(pgrep nvargus-daemon)"
else
    echo "  ERROR: nvargus-daemon failed to start. Check /tmp/nvargus.log"
    exit 1
fi

# ==================================================================
# Step 3: Clean shared memory
# ==================================================================
echo "[Step 3/4] Cleaning shared memory..."
rm -f /dev/shm/stereo_meta /dev/shm/stereo_left /dev/shm/stereo_right 2>/dev/null || true

# ==================================================================
# Step 4: Start camera bridge
# ==================================================================
echo "[Step 4/4] Starting camera bridge..."
cd "$SCRIPT_DIR"

BRIDGE_ARGS=(
    --left_sensor "$LEFT_SENSOR"
    --right_sensor "$RIGHT_SENSOR"
    --width "$WIDTH"
    --height "$HEIGHT"
    --fps "$FPS"
    --scale "$SCALE"
)

if [ "$DO_REMAP" -eq 1 ] && [ -f "$CALIB_FILE" ]; then
    BRIDGE_ARGS+=(--calib_file "$CALIB_FILE")
fi

nohup python3 camera_bridge.py "${BRIDGE_ARGS[@]}" \
    > /tmp/camera_bridge.log 2>&1 &

BRIDGE_PID=$!
echo "  Camera bridge started (PID: $BRIDGE_PID)"
echo "  Log: /tmp/camera_bridge.log"

# Wait for shared memory to appear
echo -n "  Waiting for camera frames..."
for i in $(seq 1 60); do
    if [ -f /dev/shm/stereo_meta ]; then
        SIZE=$(stat -c%s /dev/shm/stereo_meta 2>/dev/null || echo 0)
        if [ "$SIZE" -eq 40 ]; then
            echo " ready!"
            break
        fi
    fi
    sleep 0.5
    echo -n "."
done

if [ ! -f /dev/shm/stereo_meta ]; then
    echo ""
    echo "  ERROR: Camera bridge failed to start. Check /tmp/camera_bridge.log"
    echo "  Last 10 lines of log:"
    tail -10 /tmp/camera_bridge.log
    exit 1
fi

echo ""
echo "============================================"
echo " Camera setup complete!"
echo ""
echo " Run perception pipeline in Docker:"
echo ""
echo "  bash docker/run_jetson.sh bash -c 'pip install -q \"numpy<2\" && python scripts/run_perception.py \\'"
echo "      --cam_mode shm \\'"
echo "      --detector_type yolo_world \\'"
echo "      --model_dir weights/20-30-48/model_best_bp2_serialize.pth \\'"
echo "      --calib_file calibrations/stereo_calib.npz \\'"
echo "      --yolo_weights weights/yolo_world_finetuned/model_best.pt \\'"
echo "      --text_prompt \"a cup. a bottle.\" \\'"
echo "      --depth_scale 1.0 \\'"
echo "      --scale 0.5 --save_dir output/screenshots/ \\'"
echo "      --display 0'"
echo ""
echo " To stop:  pkill -f camera_bridge.py"
echo "============================================"
