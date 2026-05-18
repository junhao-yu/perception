# Fast-FoundationStereo 运行笔记

## 环境

- Conda 环境: `ffs` (Python 3.12)
- GPU: CUDA 可用
- 相机: DECXIN-2784V1 (USB ID `1bcf:2d4f`)，设备路径 `/dev/video2`

## 运行 Demo（静态图片）

```bash
conda activate ffs
cd /home/yu/perception/Fast-FoundationStereo

python scripts/run_demo.py \
    --model_dir weights/20-30-48/model_best_bp2_serialize.pth \
    --left_file demo_data/left.png \
    --right_file demo_data/right.png \
    --intrinsic_file demo_data/K.txt \
    --out_dir output_demo_new/ \
    --remove_invisible 1 \
    --denoise_cloud 1 \
    --scale 1 \
    --get_pc 1 \
    --valid_iters 4 \
    --max_disp 192 \
    --zfar 100
```

## 运行实时双目（DECXIN 相机）

### 无标定模式（快速验证）

```bash
python scripts/live_stereo.py \
    --model_dir weights/20-30-48/model_best_bp2_serialize.pth \
    --cam_id 2 \
    --cam_width 1280 \
    --cam_height 480 \
    --no_rectify \
    --valid_iters 4 \
    --scale 1.0 \
    --get_pc 0
```

窗口显示后按 `q` 或 `ESC` 退出。

### 有标定模式（精确深度 + 点云）

先标定：

```bash
python scripts/calibrate_stereo.py \
    --out_dir calibrations/ \
    --cam_id 2 \
    --width 1280 \
    --height 480 \
    --square_size 0.025
```

需要 9x6 棋盘格，按空格采集（至少 15 对），ESC 结束。

再运行：

```bash
python scripts/live_stereo.py \
    --model_dir weights/20-30-48/model_best_bp2_serialize.pth \
    --calib_file calibrations/stereo_calib.npz \
    --cam_id 2 \
    --cam_width 1280 \
    --cam_height 480 \
    --valid_iters 4 \
    --get_pc 1 \
    --out_dir output_live/
```

## 遇到的问题及解决

### 1. output/ 目录权限不足 (Permission denied)

**现象**：运行 demo 时报 `PermissionError: [Errno 13] Permission denied: '.../output/left.png'`

**原因**：之前 Docker 运行时 (`root` 用户) 在 `output/` 下创建了图片，当前用户 `yu` 无写入权限。

**解决**：改用新的输出目录，如 `output_demo_new/`。如需清理旧目录需要 `sudo rm -rf output/`。

### 2. 相机设备枚举

**现象**：`/dev/video1` 和 `/dev/video3` 无法打开（`Inappropriate ioctl for device`）

**原因**：DECXIN 相机注册了 4 个 V4L2 设备，其中 `/dev/video0` 是内置摄像头，`/dev/video2` 是 DECXIN 双目相机（side-by-side 模式），`/dev/video1` 和 `/dev/video3` 是元数据接口，不提供视频流。

**解决**：使用 `--cam_id 2` 指定 `/dev/video2`。

### 3. OpenCV Qt 字体警告

**现象**：`QFontDatabase: Cannot find font directory ... cv2/qt/fonts`

**原因**：opencv-contrib-python 的 Qt 后端缺少字体文件。

**影响**：仅影响窗口标题栏字体渲染，不影响功能，可忽略。也可安装 `fonts-dejavu-core` 消除警告。
