# YOLO 算子加速开发指南

> 目标硬件：**NVIDIA GPU** 与 **AMD Ryzen 7 7840HS**（Zen4 CPU + Radeon 780M 核显 + XDNA NPU）
> 优化对象：YOLO 系列检测模型（Focus / SPPF / Decode / NMS 为典型结构）
> 本文档用于后期开发对照，按「平台 → 加速领域 → 实现要点 → 验证」组织。

---

## 目录

- [0. 总览](#0-总览)
- [Part I：NVIDIA GPU 方向](#part-invidia-gpu-方向)
- [Part II：Ryzen 7 7840HS 方向](#part-iiryzen-7-7840hs-方向)
- [附录 A：外置优化总清单（两平台通用）](#附录-a外置优化总清单两平台通用)
- [附录 B：开发路线图（TODO Checklist）](#附录-b开发路线图todo-checklist)
- [附录 C：参考仓库与资料](#附录-c参考仓库与资料)

---

## 0. 总览

### 0.1 两平台路线对比

| 维度 | NVIDIA GPU | Ryzen 7 7840HS |
|---|---|---|
| 计算单元 | CUDA 核心 + Tensor Core | Zen4 CPU（8C16T，AVX2/双泵 AVX-512）+ 780M 核显（RDNA3, gfx1103, Vulkan）+ XDNA NPU（仅 Windows） |
| 主力推理栈 | **TensorRT**（ONNX → engine） | **ncnn + Vulkan**（主线）；OpenVINO/ORT CPU（保底）；Ryzen AI NPU（支线） |
| 图级融合 | Builder（Myelin）自动完成 | pnnx 转换时自动完成 |
| 需要手写的算子 | Decode 插件（或用现成插件）、预处理 kernel | Decode（AVX2 必写，Vulkan 可选）、混合式 NMS（可选） |
| 精度模式 | FP32 / TF32 / FP16 / INT8（校准） | FP32 / FP16 / INT8（ncnn2int8 校准）；NPU 仅 INT8/BF16 |
| 后处理策略 | Decode+NMS 全部 GPU 化，只回传框 | 混合：IoU 重活上 GPU，贪心选择留 CPU |
| 典型瓶颈 | 预处理 CPU↔GPU 拷贝、小 batch 利用率 | 大分辨率首层 conv、串行小算子、CPU↔GPU 带宽 |

### 0.2 平台无关的模型侧改图（两平台共同前置）

这些改动在导出前完成，两个平台都受益：

1. **Focus 重写**：切片 + concat 是纯显存搬运，必须消除。两种等价方案：
   - **保权重方案**：space-to-depth 重排输入 + 卷积权重按通道序 `[c, dy, dx]` 重排；
   - **重训/新模型方案**：直接换成 6×6/s2 卷积（YOLOv5 v6.0 起官方做法）。
2. **SPPF 并行化**：串行 3 次 5×5 MaxPool 数学上等价于并行 `mp5 / mp9 / mp13 + concat`（stride=1 时 5×5 池化串联 = 感受野线性叠加）。消除串行依赖，对核显/NPU 收益最大。
3. **类别数裁剪**：COCO 80 类 → 实际 N 类，head 输出通道 `255 → 3×(N+5)`。10 类场景 head 计算量降约 75%，decode 遍历量同比缩小。直接切权重通道，无需重训。
4. **导出约定**：
   - 导出 ONNX（opset ≥ 12），先过 onnxsim 化简；
   - **切掉 Detect 层的 decode 部分**，三个 head 的原始卷积输出作为模型输出，decode/NMS 在部署侧实现（图干净、量化友好）；
   - 注意记录三个输出 blob 与 stride（8/16/32）的对应关系。

### 0.3 关键数学技巧（通用）

**logit 域阈值**：sigmoid 单调递增，因此阈值比较可以在 logit 域完成：

```cpp
const float logit_thr = logf(conf_thr / (1.f - conf_thr));  // 预计算一次
// obj 分支: if (obj_logit < logit_thr) continue;  —— 全部 anchor 免 exp
// 类别 argmax 同样在 logit 上比较，仅存活框真正算 sigmoid
```

效果：decode 提速约 3~5 倍，两平台通用。

---

# Part I：NVIDIA GPU 方向

## 1. 技术栈选型

| 层级 | 选型 | 说明 |
|---|---|---|
| 图优化 + 推理 | **TensorRT**（ONNX → .engine） | 自动完成 Conv+BN+激活融合、kernel 自动调优 |
| Decode / NMS | 官方插件 `EfficientNMS_TRT` / `batchedNMSPlugin`，或自写 CUDA 插件 | 参考 tensorrtx 的 `yololayer.cu` |
| 预处理 | 自写 CUDA kernel（letterbox + normalize + HWC→CHW 单 kernel） | 避免 CPU 侧预处理 + H2D 拷贝 |
| 部署形态 | 进程内 TRT runtime，或多模型上 Triton Inference Server | 视业务规模 |
| 量化 | FP16（一行 flag）/ INT8（熵校准，需 500~1000 张校准集） | head 敏感层可回退 FP16 |

**核心认知：NVIDIA 上 90% 的融合工作 TensorRT builder 自动完成**（`trtexec --verbose` 可看到 Conv+BN+SiLU 被合成一个 kernel），开发精力集中在 **Decode/NMS 插件、预处理 GPU 化、INT8 校准** 三件事上。

## 2. 算子加速领域清单

### 2.1 图级融合（自动，无需开发）

| 优化项 | 机制 |
|---|---|
| Conv+BN | builder 常量折叠，BN 参数折进卷积权重：`W′ = γ/√(σ²+ε)·W`，`b′ = β + γ(b−μ)/√(σ²+ε)` |
| Conv+Bias+SiLU | Myelin kernel 融合，SiLU 进 epilogue |
| Focus | 模型侧改图后不存在；未改图时 slice+concat 会原样保留（慢），**务必先改图** |
| SPPF | 并行改写后三个 MaxPool 可被调度并行执行 |

### 2.2 Decode CUDA 插件（必做项之一）

- 一个线程处理一个 anchor；grid/anchor/stride 表预计算后放 constant memory 或 device buffer；
- 结合 logit 域阈值（见 0.3），warp 内 ballot 压缩存活索引，atomicAdd 全局计数器写紧凑输出；
- 输出直接接 NMS 插件输入（boxes + scores 两个 tensor）。
- 参考实现：tensorrtx `yololayer.cu`；`IPluginV2DynamicExt` 接口（支持动态 shape）。

### 2.3 NMS 插件

| 方案 | 说明 |
|---|---|
| `EfficientNMS_TRT` | 官方插件，支持 class-aware/class-agnostic、score 阈值、top-K，**首选** |
| `batchedNMSPlugin` | DetectionOutput 风格，位掩码迭代算法 |
| 自写位掩码 kernel | 参考 mmcv `nms` kernel：每 64 框一个 `uint64` 掩码，按 score 降序逐块抑制，O(N²/64) 访存 |

算法要点（位掩码迭代式）：按 score 排序后，第 i 个保留框生成「与后续所有框 IoU > 阈值」的位掩码，后续框的存活状态 = 所有已保留框掩码的按位或取反。轮数 = 保留框数（几十轮），每轮全并行。

### 2.4 预处理 GPU 化

- 单 kernel 完成：解码后 BGR/RGB → letterbox resize（双线性）→ `/255` 归一化 → HWC→CHW 重排；
- 视频场景配合 NVDEC：解码输出直接留在显存（CUDA-Vulkan/NVDEC 互操作），零 H2D 拷贝。

### 2.5 量化

```cpp
config->setFlag(nvinfer1::BuilderFlag::kFP16);            // FP16：一行
config->setFlag(nvinfer1::BuilderFlag::kINT8);            // INT8：需校准器
config->setInt8Calibrator(myEntropyCalibrator);           // 熵校准，500~1000 张代表性图
```

- INT8 掉点明显时：用 `setPrecision` / layer precision constraints 把三个 head 输出层留在 FP16；
- Tensor Core 要求通道对齐，INT8 下 tensor 布局按 32 通道对齐收益最大（builder 自动处理）。

### 2.6 运行时优化

- **CUDA stream**：预处理 kernel、推理、后处理同 stream 串行；多路视频用多 stream；
- **CUDA Graph**（TRT ≥ 8.x + 固定 shape）：消除 kernel launch 开销，小模型收益 10%+；
- `enqueueV3` + 显式绑定输入输出地址，避免每帧分配；
- context 复用，warmup 3~5 次再测速。

## 3. 实现流水线

```
best.pt
  │  ① 模型侧改图（Focus/SPPF/类别裁剪，见 0.2）
  ▼
export.py --include onnx --simplify          # 切 head 导出
  │  ② onnx-graphsurgeon（可选）：替换/注册 Decode+NMS 节点
  ▼
trtexec --onnx=best.onnx --saveEngine=best.engine --fp16
  │  ③ 或 C++ builder API + 自定义插件注册
  ▼
推理进程：CUDA 预处理 kernel → context->enqueueV3 → Decode/NMS 插件输出 → 只回传最终框
```

## 4. Builder 配置骨架

```cpp
auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
    builder->createNetworkV2(1U << int(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
auto parser  = std::unique_ptr<nvonnxparser::IParser>(
    nvonnxparser::createParser(*network, gLogger));
parser->parseFromFile("best.onnx", (int)nvinfer1::ILogger::Severity::kWARNING);

auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 30);
config->setFlag(nvinfer1::BuilderFlag::kFP16);
// 动态 shape 才需要 optimization profile；固定 shape 可省
auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(
    builder->buildEngineWithConfig(*network, *config));
// 序列化保存，部署时反序列化（构建只跑一次）
```

## 5. 验证与调优

1. **数值对齐**：同一输入，PyTorch head 原始输出 vs TRT 输出，FP32 max abs err < 1e-3，FP16 < 1e-2；
2. **分层耗时**：`trtexec --profilingVerbosity=detailed` 导出每层耗时；Nsight Systems 看流水线空隙；
3. **精度验证**：量化后跑 COCO val 子集（500 张）mAP 对比；
4. 顺序：**fp32 正确 → fp16 → int8 → 插件/Graph 优化**，每步先对齐数值再谈速度。

## 6. 常见坑

| 坑 | 对策 |
|---|---|
| Focus 未改图，slice+concat 每帧大量显存往返 | 导出前改图（0.2） |
| INT8 校准集不具代表性，mAP 暴跌 | 校准图覆盖真实分布；head 留 FP16 |
| 每帧 cudaMalloc | 预分配 + 地址复用 |
| Decode 用 Python/CPU 做 | 必须插件化，feature map 不下传 |
| 动态 shape 未配 profile 直接崩 | 固定 shape 或显式 optimization profile |

---

# Part II：Ryzen 7 7840HS 方向

## 7. 硬件盘点与路线选型

7840HS（Phoenix APU）内含三个计算单元，路线按性价比排序：

| 计算单元 | 栈 | 定位 | 平台限制 |
|---|---|---|---|
| **780M 核显（RDNA3, gfx1103）** | **ncnn + Vulkan（主线）** | conv 全部走核显，fp16 直通 | Linux / Windows 均可，最省心 |
| Zen4 CPU（8C16T, AVX2/双泵 AVX-512） | OpenVINO / ONNX Runtime / ncnn CPU | 保底基准、后处理 | 无 |
| XDNA NPU（Phoenix） | Ryzen AI SW + Quark + ORT Vitis AI EP | 低功耗支线 | **仅 Windows、仅量化模型；Phoenix 属老平台，新版软件栈支持不稳定** |

备选：Linux 下 ONNX Runtime **MIGraphX EP** 也可跑 780M（gfx1103 社区实测可用；ORT 1.23 起 ROCm EP 已移除，MIGraphX 是官方迁移路径），但自定义算子不如 ncnn 灵活，仅作备选。

## 8. 主线：ncnn + Vulkan（780M）

### 8.1 转换流水线

```bash
# PyTorch → ONNX（已按 0.2 改图、切 head）
python export.py --weights best.pt --include onnx --simplify
# ONNX → ncnn（pnnx 路线）
pnnx best.onnx inputshape=[1,3,640,640]
# 产物：best.pnnx.param / best.pnnx.bin
# fp16 存储压缩（体积减半）
ncnn2fp16 best.pnnx.param best.pnnx.bin
```

pnnx 自动完成：**Conv+BN 折叠、Conv+SiLU 合并、Residual 相加合并**——0.2 的改图让它更彻底。

### 8.2 量化（INT8）

```bash
# ① 生成校准表（500~1000 张代表性图片）
ncnn2table best.param best.bin images/ imagelist.txt best.table \
    mean=[0,0,0] norm=[0.00392,0.00392,0.00392] shape=[640,640,3] pixel=BGR thread=8
# ② per-channel INT8 量化
ncnn2int8 best.param best.bin best-int8.param best-int8.bin best.table
```

掉点明显时：编辑 param，把 SPPF 输出层与三个 head 留在 fp16（约 5% 算力换回大部分精度）。

### 8.3 推理配置

```cpp
ncnn::Net net;
net.opt.use_vulkan_compute   = true;   // 走 780M
net.opt.use_fp16_packed      = true;
net.opt.use_fp16_storage     = true;
net.opt.use_fp16_arithmetic  = true;
net.opt.use_shader_pack8     = true;   // RDNA 上 pack8 有收益
net.opt.use_packing_layout   = true;
net.opt.lightmode            = true;
net.opt.num_threads          = 8;      // Zen4 物理核数，别开满 16 逻辑线程
net.load_param("best.param");
net.load_model("best.bin");
```

### 8.4 自定义算子 ①：Decode（唯一必写）

ncnn 官方推荐的自定义层开发节奏（四步迭代，每步可独立验证）：

1. fp32 / elempack=1 把数学写对；
2. 加 packing 支持；
3. 加 fp16/bf16 分支；
4. 最后开 `support_vulkan` 写 Vulkan 路径。

**(a) CPU AVX2 版**

要点：
- 预计算三个尺度的 `(gy, gx, stride, anchor_w, anchor_h)` 表，SoA 连续存储，只算一次；
- logit 域阈值快筛（0.3），99% 的 anchor 直接跳过；
- SoA 布局遍历，`-O3 -march=znver4 -ffast-math`，OpenMP 并行；
- sigmoid 用向量化近似（ncnn 源码 `sigmoid_ps` 可直接借）。

```cpp
void decode(const ncnn::Mat& pred, std::vector<Box>& out,
            const AnchorTable& tab, float logit_thr)
{
    const float* p_obj = pred.row(4);
    #pragma omp parallel for
    for (int i = 0; i < pred.w; i++) {
        if (p_obj[i] < logit_thr) continue;              // logit 域快筛
        // 类别 argmax 也在 logit 域；通过后才 sigmoid
        float obj = sigmoid_fast(p_obj[i]);
        float x = (sigmoid_fast(pred.row(0)[i]) * 2.f - 0.5f + tab.gx[i]) * tab.stride[i];
        float y = (sigmoid_fast(pred.row(1)[i]) * 2.f - 0.5f + tab.gy[i]) * tab.stride[i];
        float w = powf(sigmoid_fast(pred.row(2)[i]) * 2.f, 2) * tab.aw[i];
        float h = powf(sigmoid_fast(pred.row(3)[i]) * 2.f, 2) * tab.ah[i];
        // ... 阈值判断后 push
    }
}
```

**(b) Vulkan compute 版**

- 一个 invocation 处理一个 anchor；anchor 表放 SSBO；
- 早退 + `atomicAdd` 压缩写回；三个尺度合并进一个 dispatch（stride 表区分），避免多次小 dispatch。

```glsl
#version 450
layout(local_size_x = 256) in;
layout(binding = 0) readonly  buffer Pred  { float pred[]; };   // N*85
layout(binding = 1) readonly  buffer Anc   { vec4 anc[]; };     // gx,gy,stride,aw / ah
layout(binding = 2) writeonly buffer Boxes { vec4 boxes[]; };
layout(binding = 3) buffer Meta { uint count; float scores[]; };
layout(push_constant) uniform PC { uint N; float logit_thr; } pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.N) return;
    if (pred[i*85 + 4] < pc.logit_thr) return;          // 早退
    // argmax class（logit 域）→ 阈值 → decode → 压缩写回
    uint slot = atomicAdd(count, 1);
    boxes[slot] = decode_one(i);
    scores[slot] = score;
}
```

### 8.5 自定义算子 ②：混合式 NMS

**判断前提**：decode 阈值过滤后候选框通常几十~几百个，CPU SIMD NMS 仅 0.1~0.3 ms——框 ≤300 时不必 GPU 化。值得算子化的场景：大分辨率 / 低阈值 / 密集小目标（候选框上千）。

**推荐形态：重活（O(N²) IoU 矩阵）上 GPU，贪心选择留 CPU 位掩码**，避免多轮 dispatch 的同步开销：

```glsl
#version 450
// iou_matrix.comp —— 一个 dispatch 算完 IoU 上三角矩阵
layout(local_size_x = 16, local_size_y = 16) in;
layout(binding = 0) readonly  buffer Boxes { vec4 b[]; };     // x1,y1,x2,y2
layout(binding = 1) readonly  buffer Meta  { vec4 m[]; };     // score,label,area,pad
layout(binding = 2) writeonly buffer IoU   { float iou[]; };  // N*N
layout(push_constant) uniform PC { uint N; } pc;

void main() {
    uint i = gl_GlobalInvocationID.x, j = gl_GlobalInvocationID.y;
    if (i >= pc.N || j >= pc.N || j <= i) return;
    if (int(m[i].y) != int(m[j].y)) { iou[i*pc.N+j] = 0.0; return; }  // class-aware
    vec4 a = b[i], c = b[j];
    float iw = max(min(a.z, c.z) - max(a.x, c.x), 0.0);
    float ih = max(min(a.w, c.w) - max(a.y, c.y), 0.0);
    float inter = iw * ih;
    iou[i*pc.N+j] = inter / (m[i].z + m[j].z - inter);
}
```

CPU 侧：矩阵转位掩码 `uint64_t mask[N][ceil(N/64)]`，贪心遍历（按 score 降序，逐保留框按位或抑制）——N=300 时 11KB，几十微秒。

CPU SIMD 贪心 NMS 要点（默认版，必写）：
- decode 输出 `partial_sort` 截 top-200/300，不全排；
- SoA 存 `x1,y1,x2,y2,area,score,label`，IoU 内层 8-wide SIMD；
- class-aware 时同类框连续排列，整块跳过异类；
- 早停：剩余 score 过低即 break。

### 8.6 预处理融合 shader + 零拷贝

- 相机管线：YUV→RGB、letterbox resize、归一化、HWC→CHW 合并为**一个 compute shader**（ncnn 默认 `from_pixels_resize` 在 CPU，是四次内存往返）；
- Linux 下相机 buffer 经 dmabuf 导入 Vulkan external memory，零拷贝；
- **feature map 绝不下传**：decode(+NMS) 全部 GPU 侧完成，只回传几十字节的框。

## 9. 支线：NPU（Ryzen AI，仅 Windows）这里由于基本用的是LINUX所以并没有实现

- 流程固定：**Quark 量化 → ONNX Runtime + Vitis AI EP**；
- 量化策略按 AMD 官方顺序：XINT8（对称 INT8 + 2 的幂 scale，NPU 原生格式）→ 精度不够依次加 CLE → AdaRound → AdaQuant → 再不行换 A8W8/A16W8/BF16；
- AMD 官方实测 YOLO 系列 A8W8 基本无精度损失，NPU 推理比 CPU 快 7 倍以上；
- 不支持的算子（decode、reshape 等）自动切 CPU 子图执行——**NPU 路线零自研算子**；
- Phoenix（7840HS）注意事项：
  - 编译 INT8 时 `target` 必须设 `X1`，配 phoenix 的 `4x4.xclbin`；
  - 新版 Ryzen AI 软件栈在 Phoenix 上有已知运行时故障（NPU 可检测但 Vitis AI 执行路径报错），建议锁定验证过的旧版本组合；
  - Linux 下 NPU 基本不可用（AMD Linux 文档仅覆盖更新的 Strix/Krackan）。

## 10. 保底：纯 CPU 路径

- **OpenVINO**：ONNX 直接转 IR，图融合全自动，Zen4 上开箱即用，适合作为正确性基准；
- **ncnn CPU 模式**：与主线同一份 param/bin，`use_vulkan_compute=false`，packing + AVX2 自动；
- 编译：`-O3 -march=znver4 -ffast-math`，线程绑物理核。

## 11. 系统层调优（十分钟，白捡的性能）

```bash
# CPU 绑物理核 0-7（避开 SMT 逻辑核）
taskset -c 0-7 ./your_app
# CPU 频率策略
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
# 780M GPU 频率拉满
echo high | sudo tee /sys/class/drm/card*/device/power_dpm_force_performance_level
```

其他：`Extractor` 复用、VkAllocator 池化避免每帧显存分配；采集线程与推理线程双缓冲流水线（同一 `Net` 可建多个 `Extractor`，注意 Vulkan 队列竞争，两条流水线足够）。

## 12. 验证与 profiling

1. **数值对齐**：PyTorch dump 三个 head 原始输出 → 同输入喂 ncnn → fp32 max abs err < 1e-3，fp16 < 1e-2；
2. **精度验证**：量化后 COCO val 子集（500 张）mAP 对比掉点；
3. **分层耗时**：ncnn 编译加 `-DNCNN_BENCHMARK=ON` 打印每层耗时（经验：780M 瓶颈通常在前两层大分辨率 conv 和 nearest 上采样，而不是 head）；
4. 迭代顺序：fp32 正确 → fp16 → INT8 → 自定义层 GPU 化。**每步先对齐精度再压速度**。

经验量级（以实测为准）：YOLOv5s@640，780M fp16 Vulkan 约十几~三十毫秒；Zen4 CPU int8 约三十~五十毫秒。

---

# 附录 A：外置优化总清单（两平台通用）

按投入产出比排序：

| 优先级 | 优化项 | 说明 | 收益 |
|---|---|---|---|
| ★★★ | logit 域阈值 + top-K 截断 | 0.3 节；decode 免全体 exp | decode 提速 3~5× |
| ★★★ | 类别数裁剪 | 80→N 类，head 通道 255→3(N+5) | head 算力 -75%（10 类时） |
| ★★★ | 预处理单 pass 化 + 零拷贝 | 一个 kernel/shader 完成全部预处理；NVIDIA 用 NVDEC 显存直通，780M 用 dmabuf | 省掉最大的一次 CPU↔GPU 传输 |
| ★★★ | feature map 不下传 | decode+NMS 在 GPU 侧，只回传最终框 | 省 4MB+/帧 |
| ★★ | 矩形推理 | 16:9 视频按长边 640、短边对齐 stride=32 | 少算约 25% |
| ★★ | 跳帧 + 跟踪 | BYTETrack，隔 2~3 帧检测 | 等效帧率翻倍（系统级最大收益） |
| ★★ | 异步流水线 | 采集/推理双缓冲，多 stream / 多 Extractor | 隐藏预处理延迟 |
| ★ | 混合精度兜底 | 敏感层（SPPF 输出、head）留 fp16 | 量化掉点最小化 |
| ★ | 系统层 | 绑核、governor、GPU dpm、warmup、内存池 | 白捡 5~15% |

# 附录 B：开发路线图（TODO Checklist）

**阶段 0 — 模型侧（两平台共同）**
- [ ] Focus 改写（space-to-depth 保权重 或 6×6/s2 重训）
- [ ] SPPF 并行化改写 + 数值对齐
- [ ] 类别数裁剪
- [ ] 切 head 导出 ONNX + onnxsim

**阶段 1 — 基线打通**
- [ ] [NVIDIA] trtexec FP16 baseline + 数值对齐
- [ ] [7840HS] pnnx 转换 + ncnn-Vulkan FP16 baseline + 数值对齐
- [ ] [7840HS] OpenVINO CPU 基准（正确性参照）

**阶段 2 — 后处理算子化**
- [ ] logit 域阈值 decode（CPU SIMD，两平台通用验证逻辑）
- [ ] [NVIDIA] Decode CUDA 插件 / EfficientNMS_TRT 接入
- [ ] [7840HS] Decode Vulkan compute 版（如 profiling 需要）
- [ ] CPU SIMD NMS（partial_sort top-K + SoA + 早停）
- [ ] [可选] 混合式 NMS：IoU 矩阵 shader + CPU 位掩码

**阶段 3 — 量化**
- [ ] [NVIDIA] INT8 熵校准 + head FP16 回退
- [ ] [7840HS] ncnn2table + ncnn2int8 + 敏感层 FP16
- [ ] mAP 回归（COCO val 500 张）

**阶段 4 — 系统与外置**
- [ ] 预处理融合 kernel/shader + 零拷贝
- [ ] 矩形推理 / 动态 shape
- [ ] 异步流水线 + 内存池
- [ ] 系统层调优（绑核/频率）
- [ ] [视频场景] BYTETrack 跳帧

**阶段 5 — 支线（按需）**
- [ ] [7840HS·Windows] NPU：Quark XINT8 → Vitis AI EP
- [ ] [NVIDIA] CUDA Graph / Triton 服务化

# 附录 C：参考仓库与资料

| 资源 | 用途 |
|---|---|
| `ultralytics/yolov5` | 官方导出流程、`Conv.fuse()` 参考实现 |
| `wang-xinyu/tensorrtx` | TensorRT 手搭网络 + Decode CUDA 插件（`yololayer.cu`） |
| `shouxieai/tensorRT_Pro` | 手撕 CUDA 前后处理 + 融合推理 |
| `Linaom1214/TensorRT-For-YOLO-Series` | ONNX + 图手术路线 |
| `Tencent/ncnn` | 主线框架；内置 sigmoid AVX2 实现、自定义层 wiki（fp32→packing→fp16→Vulkan 四步法） |
| `NVIDIA/TensorRT`（开源插件） | `batchedNMSPlugin` 位掩码迭代 NMS 教科书实现 |
| `open-mmlab/mmcv` | 单 kernel 位掩码 NMS，适合移植 GLSL |
| AMD Quark 文档 | Ryzen AI NPU 量化（XINT8/CLE/AdaRound/AdaQuant） |
| ONNX Runtime Vitis AI EP 文档 | NPU 部署、PHX `X1` target 配置 |

---

