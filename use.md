总结
你的新相机是 GEAC GMSL 双目相机（max96712 解串器 + IMX185 传感器），通过 NVIDIA Argus 相机栈访问。旧代码用的是 USB UVC 相机（单个 /dev/video2，输出左右拼接的 side-by-side 画面），新相机是两个独立的 CSI 相机，需要分开捕获。

核心问题
Docker 容器里没有 L4T 相机库（libnvargus.so 等），无法直接调用 nvarguscamerasrc
旧代码用的 cv2.VideoCapture V4L2 方式打不开这些相机
解决方案：相机桥接

宿主机 (有完整 L4T 驱动)               Docker 容器 (有 PyTorch/CUDA)
┌──────────────────────┐              ┌──────────────────────┐
│ camera_bridge.py     │   /dev/shm   │ run_perception.py   │
│ GStreamer 捕获左右相机 │ ←────────── │ 从共享内存读取帧      │
│ sensor-id 0 和 1     │   共享内存    │ 不需要相机驱动        │
└──────────────────────┘              └──────────────────────┘
新增文件（4个）
文件	作用
scripts/camera_bridge.py	在宿主机运行，通过 GStreamer 捕获双目图像写入 /dev/shm/
scripts/shm_camera.py	共享内存读取模块，在 Docker 里被 import
scripts/setup_camera.sh	一键初始化相机并启动桥接
docker/run_jetson.sh	更新了 Docker 启动脚本，挂载 /dev/shm 和 /tmp
修改文件（3个）
文件	改动
scripts/run_perception.py	新增 --cam_mode shm 支持双目分离捕获，旧 --cam_mode v4l2 模式保留
scripts/calibrate_stereo.py	同样支持双模式，方便用新相机标定
readme.md	新增 GEAC IMX185 相机配置说明
使用步骤
1. 初始化相机硬件（每次开机一次，宿主机）：
cd /home/nvidia/camera_demo_geac
sudo bash camera_init.sh

2. 启动相机桥接（宿主机）：
不使用VPI校正：
pkill -f camera_bridge.py
rm -f /dev/shm/stereo_*
sleep 1
python scripts/camera_bridge.py --scale 0.5 &

使用VPI校正：
cd /home/nvidia/perception/Fast-FoundationStereo
pkill -f camera_bridge.py
rm -f /dev/shm/stereo_*
python scripts/camera_bridge.py --scale 0.5 --calib_file calibrations/stereo_calib.npz &
# 或者一键: bash scripts/setup_camera.sh

3. 在 Docker 中运行感知管线：
bash docker/run_jetson.sh bash -c 'pip install -q "numpy<2" && python scripts/run_perception.py \
    --cam_mode shm \
    --detector_type yolo_world \
    --model_dir weights/20-30-48/model_best_bp2_serialize.pth \
    --calib_file calibrations/stereo_calib.npz \
    --yolo_weights weights/yolo_world_finetuned/model_best.pt \
    --text_prompt "a cup. a bottle." \
    --depth_scale 1.0 \
    --scale 0.5 --save_dir output/screenshots/'

无显示器：
bash docker/run_jetson.sh bash -c 'pip install -q "numpy<2" && python scripts/run_perception.py \
    --cam_mode shm \
    --detector_type yolo_world \
    --model_dir weights/20-30-48/model_best_bp2_serialize.pth \
    --calib_file calibrations/stereo_calib.npz \
    --yolo_weights weights/yolo_world_finetuned/model_best.pt \
    --text_prompt "a cup. a bottle." \
    --depth_scale 1.0 \
    --scale 0.5 --save_dir output/screenshots/ \
    --display 0'

bash docker/run_jetson.sh bash -c 'pip install -q "numpy<2" && python scripts/run_perception.py \
    --cam_mode shm \
    --detector_type yolo_world \
    --model_dir weights/20-30-48/model_best_bp2_serialize.pth \
    --calib_file calibrations/stereo_calib.npz \
    --yolo_weights weights/yolo_world_finetuned/model_best.pt \
    --text_prompt "a cup. a bottle." \
    --depth_scale 1.0 \
    --scale 0.5 --save_dir output/screenshots/'


4. 标定新相机（需要重新标定，分辨率和基线都变了）：
python scripts/calibrate_stereo.py \
    --cam_mode shm \
    --out_dir calibrations/ \
    --chessboard_cols 11 \
    --chessboard_rows 8 \
    --square_size 0.025

注意事项
必须先运行 GEAC 驱动的 camera_init.sh，否则相机虽然出现在 V4L2 列表中但无法出流
新相机分辨率是 1920×1200（旧的是 640×480），需要重新标定
旧的 --cam_mode v4l2 模式完全保留，换回老相机时继续能用




Captured frame pair 37/50
Captured frame pair 38/50
Captured frame pair 39/50
Captured frame pair 40/50
Captured frame pair 41/50
Captured frame pair 42/50
Captured frame pair 43/50

Collected 43 frame pairs. Running calibration...
Calibration saved to calibrations/stereo_calib.npz
/home/nvidia/perception/Fast-FoundationStereo/scripts/calibrate_stereo.py:178: DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)
  f.write(f'{abs(float(T[0]))}\n')
Intrinsics (rectified) saved to calibrations/K.txt

=== Calibration Results ===
RMS reprojection error: 0.2636
/home/nvidia/perception/Fast-FoundationStereo/scripts/calibrate_stereo.py:183: DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)
  print(f"Baseline: {abs(float(T[0])):.4f} m")
Baseline: 0.0553 m
K_left (original):
[[483.29875779   0.         475.26769815]
 [  0.         483.3042332  298.85188396]
 [  0.           0.           1.        ]]
P1 (rectified):
[[485.33602976   0.         476.01800919   0.        ]
 [  0.         485.33602976 300.26163101   0.        ]
 [  0.           0.           1.           0.        ]]

Showing rectification preview. Press any key to exit.
Press any key in the preview window to exit.



