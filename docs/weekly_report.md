# 周报

完成双目视觉感知定位管线的搭建与优化工作

1、搭建 Fast-FoundationStereo 双目深度估计环境，完成 DECXIN-2784V1 USB 双目相机的标定与深度验证，实现实时 metric depth 输出

2、集成三种检测器（YOLOv8 / Grounding DINO / YOLO-World v2）到统一管线，实现 `--detector_type` 一键切换；设计 detector 抽象接口与 PerceptionPipeline 融合模块，完成项目模块化重构

3、定位并解决深度精度问题：发现单一 depth_scale 乘法修正无法覆盖全量程（近距离 7% 误差、远距离 22% 误差），分析根因是 FFS 视差估计存在恒定像素偏差，引入 depth_scale + disp_offset 两参数修正模型，全量程误差降至 7% 以内

4、定位帧率瓶颈：通过添加逐步骤耗时分析，发现 cap.read() 阻塞耗时 ~130ms 为最大瓶颈，设置相机 FPS 参数后帧率从 8 提升至 25；同时将 Grounding DINO（~250ms/帧）替换为 YOLO-World v2（~4ms/帧），满足实时性要求

5、基于 labelImg + convert_labels 工具链完成水杯/瓶子数据集的采集与标注（15 张），对 YOLO-World v2 进行少样本微调，冻结 backbone + 低学习率策略，mAP50 达到 98.1%，推理速度无明显下降
