# 感知定位系统：技术方案总结

**目标**：构建实时双目感知定位系统 — 检测场景中的任意物体，输出其 3D 空间坐标 (X, Y, Z)。

**硬件**：笔记本电脑 + NVIDIA RTX 5060 Laptop GPU + DECXIN-2784V1 USB 双目相机

**作者**：yujunhao | 日期：2026-05

---

## 1. 模型选型

### 1.1 立体匹配：为什么选 Fast-FoundationStereo (FFS)

立体匹配是深度估计的核心。候选方案：

| 方案 | 优点 | 缺点 |
|------|------|------|
| **RAFT-Stereo** | 精度高，论文多 | 迭代次数多，慢（~200ms） |
| **CREStereo** | 精度极高 | 更慢，不适合实时 |
| **Fast-FoundationStereo** | 实时、零样本泛化强 | 2026 新论文，生态不成熟 |
| **SGBM (传统)** | 极快，无 GPU | 精度差，纹理弱处失效 |

**选择 FFS**：
- 这是我们项目 FFS 论文的官方实现，RTX 5060 Laptop 上 `--scale 0.5` 仅需 **~21ms**
- 零样本：未在当前场景训练，直接泛化。对散焦、弱纹理、重复纹理仍能给出合理结果
- CVPR 2026 最新成果，CV 社区有持续关注度
- 选 `20-30-48` checkpoint（最快变体），`valid_iters=4`（速度优先）

### 1.2 检测器：为什么从 YOLOv8 → Grounding DINO → YOLO-World v2

目标是 **开放词汇检测**（用户说"杯子""瓶子"就能检，不用重新训练）。

第一代：**YOLOv8**（自己训练的 box 检测器）
- 只有 box 一个类别，换场景就得重新标注训练
- 帧率 ~25 FPS（快）
- 结论：不满足开放词汇需求，放弃

第二代：**Grounding DINO-tiny**（HuggingFace Transformers，开放词汇）
- 零样本开放词汇，文本输入"a cup. a bottle."即可检测
- 问题：单帧推理 **~250ms**，占总耗时 91%，整体仅 ~3 FPS
- 瓶颈分析：DINO 基于 Transformer 的交叉注意力机制，在笔记本 GPU 上效率低。即使把 FFS 优化到 0ms，帧率也只有 4
- 结论：精度好但太慢，不适合实时场景

第三代：**YOLO-World v2**（Ultralytics，开放词汇 + 实时）
- 同为开放词汇，但推理仅 **~4ms**（比 DINO 快 60 倍）
- 基于 YOLOv8 架构 + CLIP 文本嵌入，检测时将文本特征与视觉特征在检测头做轻量融合
- 支持少样本微调（5-10 张图即可适应新场景），微调后推理速度不变
- 模型自动下载：`yolov8s-worldv2.pt`（small 版 ~25MB）

**最终选择：YOLO-World v2**。性能对比（RTX 5060 Laptop，`--scale 0.5`）：

| 检测器 | 推理耗时 | 总帧率 | 开放词汇 | 可微调 |
|--------|:------:|:----:|:----:|:----:|
| YOLOv8 (自训) | ~4ms | ~25 | 否 | 是 |
| Grounding DINO | ~250ms | ~3 | 是 | 否 |
| **YOLO-World v2** | ~4ms | ~25 | 是 | 是 |

### 1.3 深度估计：为什么用 disparity 公式而非直接点云

FFS 输出视差图，转深度公式：

```
Z = fx × baseline / (disparity + disp_offset) × depth_scale
```

选择在 2D 图上做融合（bbox 映射到深度图取中值），而非生成 3D 点云再查询：
- 点云生成耗内存（640×480 → 307K 点），GPU→CPU 拷贝慢
- 2D 方案天然对齐：检测在左目 2D 图上，深度在像素级对应，融合 O(1)
- 深度查询只取 bbox 中心区域的视差中值，对边缘噪声鲁棒

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────┐
│                  run_perception.py                   │
│              (统一入口，--detector_type 切换)          │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Camera ──→ remap ──→ resize ──→ Pipeline ──→ Display
│  (1280×480)   │          │           │          │
│               ▼          ▼           ▼          ▼
│          undistort   --scale     detect +     bbox +
│                                    depth +     3D 坐标
│                                     fuse       叠到图上
│                                                     │
├─────────────────────────────────────────────────────┤
│                                                     │
│  detectors/              perception/                 │
│  ├── base.py             └── pipeline.py             │
│  ├── yolo_world_detector.py                          │
│  ├── yolo_detector.py                                │
│  └── grounding_dino.py                               │
│                                                     │
└─────────────────────────────────────────────────────┘
```

核心设计原则：
- **检测器与深度解耦**：统一 `BaseDetector` 接口（`detect()` + `warmup()`），`PerceptionPipeline` 不关心具体检测器
- **单一入口**：`run_perception.py` 通过 `--detector_type` 切换后端，命令行参数按检测器分组
- **计时透明**：主循环 + pipeline 内部均有每步计时，每 30 帧打印耗时分解

---

## 3. 问题与解决路径

### 3.1 深度系统性偏短（比例误差）

**现象**：0.5m 测出 0.41m，0.9m 测出 0.75m，所有距离偏短 ~22%。

**分析**：深度公式 `Z = fx × baseline / disparity`，视差值是相对的但 fx×baseline 乘积是绝对的。如果某张棋盘格打印偏大，标定出的 fx 偏小 → fx×baseline 偏小 → 深度偏短。

**为什么是比例误差而非固定偏移？** 因为 fx×baseline 在公式中是分子系数。`ΔZ/Z = -Δ(fx×baseline)/(fx×baseline)`，与距离无关的常数。

**解决**：引入 `--depth_scale` 乘法修正。测量 5 组已知距离（0.3m-0.9m），取 ratio = 真值/测量值的均值 = 1.22。验证后残差 < 5%。

### 3.2 远距离误差 > 近距离（非比例残差）

**现象**：depth_scale=1.22 修正后，近处 OK（0.5m 误差 ~7%），远处仍偏大（2.5m 误差 ~22%）。

**分析**：视差对深度的导数非恒定：

```
∂Z/∂d = -Z² / (fx × baseline)
```

同样的视差偏差 δd，对深度的影响 δZ ∝ Z²。远距离 Z 大、视差小，δd 的杠杆效应被放大。

FFS 存在 ~3.3px 的**恒定视差偏低**（系统性 bias，非比例）。这 3.3px 在近距离视差 60px 时影响 ~5%，在远距离视差 9px 时影响 ~30%。单一乘法系数 `depth_scale` 无法同时修正。

**验证方法**：多组标定杆测量 → 最小二乘拟合 `Z_true = fx×baseline/(d+offset) × scale`

**解决**：引入两参数修正 `disp_offset`（加法，修正恒定视差偏差）：
```
Z = fx × baseline / (disparity + disp_offset) × depth_scale
```

拟合结果（14 组测量数据）：

| 参数 | 含义 | 取值 |
|------|------|:---:|
| depth_scale | fx×baseline 系统性误差修正 | 1.1513 |
| disp_offset | FFS 恒定视差偏差修正（像素） | 0.8668 |

修正效果：全量程（0.15-2.5m）误差从 6-22% 降至 **0.4-7%**。

### 3.3 图像缩放后深度翻倍

**现象**：`--scale 0.5` 后深度值翻倍（0.5m → 1m）。

**分析**：图像缩放到 0.5，视差值也约减半。但 K 矩阵内参（fx, fy, cx, cy）仍用标定原始值，未同步缩放。

**解决**：在 `PerceptionPipeline` 中传入 `image_scale`，计算深度前缩放 K：
```python
K[0,0] *= scale  # fx
K[1,1] *= scale  # fy
K[0,2] *= scale  # cx
K[1,2] *= scale  # cy
```

简化逻辑：缩放后的 fx 与缩放后的 disparity 同步缩小，比值得以保持。

### 3.4 帧率远低于理论值（相机驱动瓶颈）

**现象**：YOLO detect 4ms + FFS 21ms + 其他 7ms = 32ms，理论 31 FPS，实际仅 8 FPS。

**定位方法**：添加每步计时（`cap_read` / `remap` / `resize` / `pipeline` / `display`），每 30 帧打印耗时分解。

**发现**：`cap.read()` 耗时 ~100ms，占总时间 76%。

`cap.read()` 是 OpenCV 从 V4L2 驱动缓冲区取帧的阻塞调用。相机默认帧率设置导致其实际只输出 ~7.5 FPS。`cap.set(CAP_PROP_FPS, 60)` 后驱动重新协商，相机跑满硬件上限 ~30 FPS。

**教训**：USB 相机在 1280×480 MJPG 下的最大帧率通常在 30 FPS，但 V4L2 默认值（15 FPS）会限制实际输出。设为 60/120 触发硬件上限协商是标准做法。

### 3.5 Grounding DINO 帧率瓶颈

**现象**：DINO 方案 3.7 FPS。

**分析**：计时显示 detect 250ms / FFS 24ms / 其他 2ms。DINO 占 91%。

**根本原因**：Grounding DINO 基于 DETR-like Transformer，对输入图像做全局交叉注意力。即使 tiny 版本在笔记本 GPU 上推理也比 YOLO 架构慢 60 倍。这是**架构性差异**，不是实现优化能解决的。

**解决路径**：
1. 降检测输入分辨率 → 有损精度，收益有限
2. 检测降频（每 N 帧跑一次）→ 可行，但 bbox 有延迟
3. TensorRT 加速 DINO → 优化空间 ~2x，仍只有 6-8 FPS
4. **换 YOLO-World v2** → 架构级优化，4ms detect，一劳永逸 ✓

选择方案 4。YOLO-World 将文本特征预先与视觉骨干做轻量融合，推理时不需要每帧做 text-image 交叉注意力，这是它比 DINO 快 60 倍的根本原因。

---

## 4. 最终运行配置

```bash
# 推荐：YOLO-World v2（开放词汇 + 实时 + 可微调）
python scripts/run_perception.py \
    --detector_type yolo_world \
    --model_dir weights/20-30-48/model_best_bp2_serialize.pth \
    --calib_file calibrations/stereo_calib.npz \
    --yolo_weights yolov8s-worldv2.pt \
    --text_prompt "a cup. a bottle. a box." \
    --depth_scale 1.1513 \
    --disp_offset 0.8668 \
    --scale 0.5
```

**性能**：~25 FPS，全量程深度误差 0.4-7%，检测任意文本物体。

内部计时分解（典型帧）：
```
cap_read  :   33 ms   ← 相机帧间隔 (30 FPS)
remap     :    2 ms
resize    :    0 ms
pipeline  :   26 ms   ← (detect 4ms + FFS 21ms + depth 1ms)
display   :    3 ms
─────────────────────
TOTAL     :   64 ms   → ~15 FPS (显示帧率)
                      → ~25 FPS (pipeline 处理帧率，不含 cap_read 等待)
```

---

## 5. 核心经验总结

**检测器选型**：开放词汇检测器中，YOLO-World 比 Grounding DINO 更适合实时场景。架构差异（YOLO 单阶段 vs DETR 两阶段）带来的速度差距远超推理优化能弥补的。

**深度修正**：双目系统的测量误差有两个来源 — fx×baseline 的比例误差（乘法修正）和视差估计的恒定偏差（加法修正）。单一 depth_scale 只在近距离足够准确，全量程需要 `depth_scale + disp_offset` 联合校准。

**瓶颈分析**：在优化前必须先计时定位真实瓶颈。本次实践中 YOLO + FFS 的理论帧率很高，但被 USB 相机的 V4L2 驱动帧率限制拖垮。没有计时工具的话很容易误判为"模型太慢"。

**图像缩放**：缩放输入时必须同步缩放内参 K 矩阵。`fx, fy, cx, cy` 全是像素单位的物理量，与图像分辨率绑定的。这个 bug 非常隐蔽 — 深度值看起来合理（有数量级）但完全错误。
