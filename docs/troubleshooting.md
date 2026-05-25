# 问题总结：Stereo 感知定位系统

## 1. 双目深度系统性偏短（比例误差）

**现象**：所有距离测量值比真值短 18-22%。0.5m 测出 0.41m，0.9m 测出 0.75m。

**根因**：标定板方格尺寸不精确，导致 fx×baseline 乘积偏小约 22%。深度公式 `Z = fx × baseline / disparity`，fx×baseline 偏小 → 深度偏短。

**解决**：引入 `--depth_scale` 乘法修正系数。测量多组已知距离，计算 ratio = 真值/测量值的均值作为修正因子（本例为 1.22）。

```
Z_corrected = fx × baseline / disparity × depth_scale
```

---

## 2. 图像缩放后深度翻倍

**现象**：加 `--scale 0.5` 后，深度值翻倍（0.5m → 1m）。

**根因**：图像缩小到 0.5 倍时，视差也减半，但 K 矩阵（内参）未同步缩放。`Z = fx(原始) × baseline / (disparity × 0.5)` = 2× 真实深度。

**解决**：`PerceptionPipeline` 接收 `image_scale` 参数，在计算深度前对 K 矩阵做同步缩放：

```
fx *= scale, fy *= scale, cx *= scale, cy *= scale
```

缩放后的 fx 与缩放后的 disparity 匹配，深度结果 => scale 无关。

---

## 3. 帧率远低于预期（相机驱动瓶颈）

**现象**：YOLO detect ~4ms，FFS ~21ms，理论 ~40 FPS，实际仅 8 FPS。

**根因**：`cap.read()` 阻塞等待相机帧，耗时 ~130ms。V4L2 默认帧率 15 FPS，该相机实际运行在 ~7.5 FPS（设定值的一半）。

**定位**：添加了每步计时（cap_read / remap / resize / pipeline / display），每 30 帧打印耗时分解，发现 `cap_read` 占 ~100ms。

**解决**：`cap.set(cv2.CAP_PROP_FPS, 60)` → 相机跑满硬件上限 30 FPS，`cap_read` 降到 ~33ms。YOLO 模式帧率从 8 → ~25 FPS。

| CAP_PROP_FPS 设定 | 实际帧率 | cap_read 耗时 |
|:---:|:---:|:---:|
| 未设置（默认 15） | ~7.5 | ~130ms |
| 30 | 15.6 | ~64ms |
| 60 | 30.4 | ~33ms |
| 120 | 30.0 | ~33ms（硬件上限） |

---

## 4. Grounding DINO 帧率极低

**现象**：DINO 方案仅 3 FPS。

**根因**：Grounding DINO-tiny 在 RTX 5060 Laptop 上单帧推理 ~250ms，占 pipeline 总耗时的 91%。FFS 优化（降分辨率、TensorRT、并行）对此无效，瓶颈是检测器本身。

**解决**：换用 YOLO-World v2。同为开放词汇检测，但推理仅 ~4ms（快 60 倍），且支持少样本微调。

| 检测器 | 推理耗时 | 总 FPS | 开放词汇 | 可微调 |
|--------|:-----:|:-----:|:----:|:----:|
| Grounding DINO | ~250ms | ~3 | Yes | No |
| YOLO-World v2 | ~4ms | ~25 | Yes | Yes |
| YOLOv8 | ~4ms | ~25 | No | Yes |

---

## 5. 远距离深度误差远超近距离（非比例误差）

**现象**：修正 depth_scale 后，0.5m 处误差 ~7%，2.5m 处误差 ~22%。单一乘法系数无法同时修正远近端。

**根因**：FFS 视差估计存在 ~3.3 像素的恒定偏低偏差。视差偏差对深度的影响是非线性的：

```
ΔZ ≈ Z² / (fx × baseline) × Δd
```

同样 3px 偏差：近距离视差 ~60px → 影响 ~5%；远距离视差 ~9px → 影响 ~30%。

**解决**：引入两参数修正模型 — `depth_scale`（乘法）+ `disp_offset`（加法，修正恒定视差偏差）：

```
depth = fx × baseline / (disparity + disp_offset) × depth_scale
```

通过最小二乘拟合测量数据得到最优参数。修正后全量程误差从 6-22% 降至 0.4-7%。

| 参数 | 含义 | 本例取值 |
|------|------|:---:|
| depth_scale | 修正 fx×baseline 系统性误差 | 1.1513 |
| disp_offset | 修正 FFS 恒定视差偏差（像素） | 0.8668 |

---

## 6. YOLO-World 少样本微调

**目标**：提升特定物体（水杯、瓶子）在自采场景下的检测精度。

**方案**：
- 采集 15 张双目左目图像（不同距离、角度）
- labelImg 标注 → `convert_labels.py` 转 YOLO 格式
- 微调 YOLO-World v2：冻结 backbone 前 10 层，lr=0.001，epochs=20，batch=2

**结果**：mAP50 98.1%，cup 99.5%，bottle 79.0%。推理速度不变（~2.5ms）。

**关键参数**：少样本必须低学习率 + 冻 backbone + 强数据增强，防止过拟合。

---

## 最终运行命令

```bash
python scripts/run_perception.py \
    --detector_type yolo_world \
    --model_dir weights/20-30-48/model_best_bp2_serialize.pth \
    --calib_file calibrations/stereo_calib.npz \
    --yolo_weights weights/yolo_world_finetuned/model_best.pt \
    --text_prompt "a cup. a bottle." \
    --depth_scale 1.1513 \
    --disp_offset 0.8668 \
    --scale 0.5
```

---

## 系统架构

```
Fast-FoundationStereo/
├── detectors/                   # 检测器模块
│   ├── base.py                  # Detection dataclass + BaseDetector ABC
│   ├── yolo_detector.py         # YOLOv8
│   ├── yolo_world_detector.py   # YOLO-World v2 (推荐)
│   └── grounding_dino.py        # Grounding DINO
├── perception/
│   └── pipeline.py              # detect → depth → 3D 融合
├── scripts/
│   ├── run_perception.py        # 统一入口（--detector_type 切换）
│   ├── capture_dataset.py       # 数据集采集
│   ├── convert_labels.py        # LabelMe JSON → YOLO 格式
│   ├── finetune_yolo_world.py   # YOLO-World 微调
│   ├── calibrate_stereo.py      # 双目标定
│   └── check_depth.py           # 深度精度验证
└── calibrations/
    └── stereo_calib.npz         # 标定结果
```
