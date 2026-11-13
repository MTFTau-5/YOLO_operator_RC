# 算子优化指南

优先完成数值对齐，再按目标设备的端到端 profile 优化。实际部署说明见 [项目 README](../README.md)，模型形状与导出见 [模型契约](MODEL_CONTRACTS.md)。

| 平台 | 推理 | 本库优化位置 |
|---|---|---|
| K3 | SpaceMIT ORT / ncnn CPU | 宿主 RVV letterbox；A100 Decode/NMS，配置见 [说明](IME_PORTING.md) |
| x86 | ncnn CPU / Vulkan | AVX2 后处理和融合预处理 |
| NVIDIA | TensorRT | CUDA 预处理、Decode、NMS 或 TRT 插件 |

## 优化顺序

1. 确认输出布局、概率/logits、网格和 anchors，先对齐 FP32 golden。
2. profile 采集/预处理/推理/提取与拷贝/Decode/NMS/跟踪，找到主要耗时。
3. 预分配并复用 buffer、anchor 表和推理上下文，减少逐帧分配。
4. 用 logit 阈值快筛，只有存活点做 exp/DFL；NMS 按 score 截 top-k 后计算同类 IoU。
5. 尝试矩形输入、FP16、流水线，再考虑量化或自定义图内算子；每步验精度。

`logit_threshold = log(conf/(1-conf))` 用于尚未 sigmoid 的 logits；fused 概率直接比较 conf。v5 还需 objectness 与类别概率乘积的最终判断。

## 模型侧

- Focus 的空间重排、SPPF 改图必须验证等价性并重新测整链路，改图不保证更快。
- 类别裁剪须同时修改输出通道、模型权重和部署配置。
- raw / post-DFL / fused 的切点决定输出体积与部署计算量；动态尺寸能力受图中固定 reshape/grid 限制。
- INT8 需要代表性校准集与精度评估，敏感层可保留高精度。

## 执行侧

CUDA 输入已在显存时尽量完成图内 Decode/NMS，只回传最终框；小候选集是否用 GPU NMS按实测决定。相机零拷贝需额外验证 buffer 生命周期、格式与同步。

CPU 使用目标 ISA 编译，避免线程池过量和过窄绑核；纯 CPU 推理与 Vulkan 提交的绑核配置不能直接混用。K3 暂保留 OpenBLAS 单线程和 EP 优先初始化；这些只是越界触发条件的规避。

验收使用现有 golden 和真实模型工具，性能记录需注明设备、输入尺寸、精度、线程数及是否包含拷贝。目标机实测见 [应用说明](../app/README.md)。
