# YOLO_operator — YOLO 部署算子库

按《YOLO 算子加速开发指南》（[开发参考](./docs/Operator_Suggestion.md)）实现的两平台部署侧算子，覆盖 YOLO 推理链路中**框架未优化三部分**：预处理、Decode、NMS。

| 模块 | 平台 | 内容 | 对应文档章节 |
|---|---|---|---|
| `common/` | 共享 | Detection/AnchorTable 类型、标量参考实现、测试向量读取 | §0.2 §0.3 |
| `x86/` | 7840HS（Zen4，独立 C++，无第三方依赖） | AVX2 decode（logit 域快筛）、CPU SIMD NMS、AVX2 预处理单 pass | §8.4a §8.5 §8.6 |
| `nvidia/` | NVIDIA GPU | Decode CUDA kernel（warp ballot 压缩）、位掩码 NMS kernel、预处理融合 kernel、TensorRT 插件、pybind11 绑定 | §2.2 §2.3 §2.4 |
| `vulkan/` | 780M 核显（可选） | 混合式 NMS：IoU 矩阵 compute shader + CPU 位掩码贪心 | §8.5 |
| `app/` | 7840HS 整机应用 | **实时流水线 `yolo_rt`**：两线程三槽位丢旧帧、ncnn Vulkan 推理、卡尔曼跟踪、UDP 控制输出 | [app/README.md](./app/README.md)、`../PIPELINE.md` |
| `tools/` | 工具 | `gen_test_vectors.py` golden 生成、`bench_baseline.cpp` 无算子基线基准 | §5 §12 |

---

## 1. 性能：无算子基线 vs 本库算子

实测环境：RTX 3090 Ti + Ryzen 7 9700X，典型 YOLO 输入（3 尺度 head、N=25200 anchor、80 类、密集场景 1930 个候选框）。基线 = 功能相同但未做算子优化的实现（numpy 向量化写法 / 标量 C++ -O3）。

| 阶段 | 无算子基线 | x86 算子（AVX2) | CUDA 算子 | 加速比 |
|---|---|---|---|---|
| **Decode**（N=25200） | numpy 0.272 ms；标量 C++ 0.160 ms | **0.054 ms** | **0.0096 ms** | CPU 3× / GPU 17~28× |
| **NMS**（1930 候选） | numpy 20.4 ms；标量 C++ 0.917 ms | **0.012 ms** | **0.385 ms** | CPU 75×（对 numpy 1700×）/ GPU 2.4× |
| **预处理** 1280×720→640 | 标量 C++ 2.03 ms | **0.216 ms** | **0.0096 ms** | 9.4× / 211× |

复现：

```bash
./build/bench_baseline test_data     # 标量 C++ vs x86 AVX2
./build/nvidia/test_cuda test_data   # 末尾打印 CUDA 基准
```

解读：

- Decode 的 CPU 加速比 ≈3×，与文档 §0.3 对 logit 域阈值「提速 3~5×」的预估一致（免全体 exp）。
- **候选框少时 NMS 不要上 GPU**：1930 候选下 x86 SIMD（0.012 ms）仍快于 CUDA（0.385 ms，含设备端排序）。GPU NMS 的价值是「feature map 不下传」——decode 结果本就在显存，GPU 侧做完只回传几十字节的框，省每帧 4MB+ 拷贝；候选上千且数据已在 GPU 时才值得（§8.5）。
- 预处理是无算子时最大的隐藏开销（2 ms 级），融合 kernel 后基本归零。

---

## 2. 构建

```bash
pip install -r requirements.txt 

# CUDA 11.8 的 nvcc 与 g++13 系统头不兼容，需指定 CUDA 12.5：
cmake -S . -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.5/bin/nvcc
cmake --build build -j

```

Vulkan 依赖安装（可选，`vulkan/` 混合 NMS 模块需要；Ubuntu/Debian）：

```bash
sudo apt install libvulkan-dev glslang-tools vulkan-tools
# 7840HS 的 780M 核显还需 RADV 驱动（一般 Mesa 已自带）：
sudo apt install mesa-vulkan-drivers
vulkaninfo --summary 
```

装好后删掉 build 目录重新 `cmake`（配置期探测到 glslangValidator 才会启用该模块）。

选项：

| 选项 | 默认 | 说明 |
|---|---|---|
| `YOLOOP_BUILD_X86` | ON | x86 算子库 + 测试 |
| `YOLOOP_BUILD_CUDA` | ON | CUDA 算子库 + 测试 + Python 绑定 |
| `YOLOOP_BUILD_TRT_PLUGIN` | ON | TensorRT decode 插件（找不到头文件/库自动跳过） |
| `YOLOOP_BUILD_PYTHON` | ON | pybind11 模块 `yolo_ops_cuda` |
| `YOLOOP_BUILD_VULKAN` | ON | 需 glslangValidator + Vulkan SDK，找不到自动跳过 |
| `YOLOOP_X86_ARCH` | native | 目标机 7840HS 用 `-DYOLOOP_X86_ARCH=znver4` |

**本机实测状态**：以上五项全部实际启用并构建成功（Vulkan 有 glslangValidator + SDK，TensorRT 头文件取自 NVIDIA/TensorRT v11.3 + pip 版动态库）。「自动跳过」只在依赖缺失的机器上发生，属兜底行为。

产物：`build/x86/libyoloop_x86.a`、`build/nvidia/libyoloop_cuda.a`、`build/nvidia/yolo_ops_cuda*.so`、`build/nvidia/libyoloop_trt_plugins.so`、`build/vulkan/libyoloop_vulkan.a`。

**多 conda 环境注意**：pybind11 / TensorRT 是按「当前 shell 激活环境的 python」探测的——在哪个环境用 `yolo_ops_cuda`，就在哪个环境构建并先 `pip install pybind11`（老环境可能还需 `-DPYBIND11_FINDPYTHON=ON -DPython_EXECUTABLE=<env>/bin/python`，防止 pybind11 抓到 base 的 libpython 导致 .so ABI 不匹配）。TensorRT 装不进的老环境（如 py3.8）可直接指向其他环境的库：
`-DTRT_INCLUDE_DIR=<项目>/nvidia/trt/include -DTRT_NVINFER_LIB=<site-packages>/tensorrt_libs/libnvinfer.so.11`

---

## 3. 接口约定（对接 YOLO 的前提）

算子输入是**切掉 Detect 层 decode 部分后**三个 head 拼接的原始卷积输出（§0.2）：

- 内存布局：channel-major `[4 + 1 + num_classes, N]`，float32
  - 行 0..3：`tx, ty, tw, th`；行 4：objectness logit；行 5..：类别 logit
- anchor 顺序：stride 8 → 16 → 32 逐 level 排列；level 内 row-major 网格，anchor 索引变化最快
- decode 数学（YOLOv5 式）：`x=(σ(tx)*2-0.5+gx)*stride`，`w=(σ(tw)*2)²*anchor_w`，`score=σ(obj)*σ(cls_max)`
- 输入边长须被 32 整除（640、992、992×544 均可）；`num_classes` 运行时传入，天然支持类别数裁剪

---

## 4. 如何使用

### 4.1 Python（最省事，NVIDIA）

```python
import numpy as np, yolo_ops_cuda   # PYTHONPATH=build/nvidia

# pred: np.float32 [4+1+nc, N]，三个 head 原始输出拼接
dets = yolo_ops_cuda.decode(pred, num_classes=80, conf_thr=0.25,
                            input_w=992, input_h=992)   # 分辨率自适应, 默认 640x640
kept = yolo_ops_cuda.nms(dets, nms_thr=0.45)            # [K,6] x1,y1,x2,y2,score,label

img = ...  # np.uint8 [H,W,3]
blob = yolo_ops_cuda.letterbox(img, 992, 992)           # [3,992,992] float32, 可直接喂模型
```

anchor 表按 `(input_w, input_h)` 首次构建并缓存，之后零开销；边长非 32 倍数抛 `ValueError`。

### 4.2 C++ / x86（7840HS，无第三方依赖）

```cpp
#include "x86/decode_avx2.h"
#include "x86/nms_simd.h"
#include "x86/preprocess_avx2.h"
using namespace yoloop;

// 初始化一次
AnchorTable tab = build_anchor_table(default_levels(992, 992));

// 每帧
std::vector<float> blob(3 * 992 * 992);
x86::letterbox_normalize(bgr_img, img_w, img_h, blob.data(), 992, 992);   // 预处理
/* ... 把 blob 喂给 ncnn/OpenVINO/ORT 拿到 pred ... */
auto dets = x86::decode(pred.data(), tab, /*num_classes=*/80, /*conf_thr=*/0.25f);
auto kept = x86::nms(std::move(dets), /*nms_thr=*/0.45f, /*topk=*/300);   // topk<=0 不截断
```

链接：`libyoloop_x86.a` + `libyoloop_common.a` + OpenMP。目标机编译 `-DYOLOOP_X86_ARCH=znver4`，运行 `taskset -c 0-7` 绑物理核（§11）。

### 4.3 C++ / CUDA（NVIDIA，进程内集成）

```cpp
#include "nvidia/yolo_ops.h"
using namespace yoloop;

// 初始化一次：anchor 表上传显存
cuda::CudaAnchorTable tab(build_anchor_table(default_levels(992, 992)));

// 每帧（全部异步于同一 stream，feature map 不下传）
cuda::letterbox_normalize(img_dev, sw, sh, blob_dev, 992, 992, stream);    // 预处理
/* ... TensorRT enqueue 拿到 pred_dev ... */
cuda::decode(pred_dev, tab.get(), 80, 0.25f, dets_dev, count_dev, cap, stream);
cuda::nms(dets_dev, n, 0.45f, keep_dev, keep_count_dev, stream);
// 只回传 keep_count 个 Detection（24B/框）
```

注意：`decode` 的 `count_dev` 是总存活数（可能 > cap），取 `min(count, cap)`。

### 4.4 TensorRT 插件（NVIDIA，engine 内嵌）

`nvidia/trt/yolo_decode_plugin.cpp` 提供 `YoloDecodePlugin`（IPluginV2DynamicExt，TRT 11.3 已验证）：

```cpp
// 构建 engine 时注册一次
REGISTER_TENSORRT_PLUGIN 已在插件编译单元内完成;
// 用 onnx-graphsurgeon / INetworkDefinition 把 "YoloDecode" 节点挂到三个 head 拼接输出后,
// plugin 字段: num_classes, conf_thr, input_w, input_h, max_output_boxes
// 输入 [1, C, N] -> 输出 [1, max_boxes, 6] float32 + [1, 1] int32 count
```

sanity 测试：`./build/nvidia/test_trt_plugin`（创建→序列化→反序列化→enqueue→销毁全链路）。

---

## 5. 怎么跟 YOLO 部署串起来

以训练好的 `best.pt` 为例，两条路线对应文档 Part I / Part II：

### 5.1 NVIDIA 路线（TensorRT）

```
best.pt
  │ ① 模型侧改图(§0.2): Focus 消除 / SPPF 并行 / 类别数裁剪(可选)
  ▼
YOLOv5 export: python export.py --weights best.pt --include onnx --simplify
  │   —— 关键: 导出前把 Detect 层的 decode 切掉, 让三个 head 的原始卷积输出
  │      直接作为 ONNX 输出(布局整理成 [1, C, N] channel-major 拼接)
  ▼
trtexec --onnx=best.onnx --saveEngine=best.engine --fp16
  │   —— 或用 C++ builder + onnx-graphsurgeon 挂上 YoloDecode 插件节点(§4.4)
  ▼
推理进程(§4.3):
  cuda::letterbox_normalize   ← 预处理 kernel(相机帧已在显存则零拷贝)
  → context->enqueueV3      ← TensorRT backbone+neck+head(自动融合 Conv+BN+SiLU)
  → cuda::decode            ← 或 engine 内嵌插件
  → cuda::nms               ← 或 CPU x86::nms(候选 ≤300 时更快, 见 §1)
  → 只回传最终框
```

### 5.2 7840HS 路线（ncnn + Vulkan 主线）

```
best.pt → ONNX(同样切 head、过 onnxsim)
  → pnnx best.onnx inputshape=[1,3,992,992]     # pnnx 自动融合 Conv+BN+SiLU
  → ncnn2fp16 / ncnn2table + ncnn2int8(可选量化, §8.2)
  ▼
推理进程(§4.2):
  x86::letterbox_normalize → ncnn::Extractor(Vulkan fp16) → 三个输出 blob
  → 按 §3 约定拼成 [C, N] → x86::decode → x86::nms → 出框
  候选框常年上千(密集小目标/低阈值)再启用 vulkan 混合 NMS(§8.5)
```

**该运行时已实现为 `app/yolo_rt`**（`app/README.md`）：两线程 + 三槽位 + 丢旧帧
流水线（规格见仓库上级 `../PIPELINE.md`），线程A 跑 letterbox/decode/NMS/卡尔曼
（`app/tracker`，逻辑提取自 RK3588-AntiUAV-Tracker）/UDP 控制输出，线程B 提交
ncnn Vulkan 推理（`app/ncnn_infer`，配置同 §8.3）。ncnn 由根 CMake
FetchContent 自动拉取（`-DNCNN_GIT_TAG` 可换版本）。

**验证顺序**（§5/§12，每步先对齐数值再压速度）：fp32 对 golden → fp16 → int8(7840hs不推荐使用int8) → 系统层调优。

---

### 5.3 实时流水线应用 `app/yolo_rt`（v5 / v8/v11 / pose）

`app/yolo_rt` 是按 `../PIPELINE.md` 实现的整机应用：两线程 + 三槽位 + 丢旧帧，
线程A 采集/预处理/decode/NMS/卡尔曼/UDP 控制，线程B 提交 ncnn Vulkan 推理。
设计与实现细节见 **[app/README.md](./app/README.md)**，这里只讲配置与调用。

所有参数收在一个扁平 YAML 里（模板：`app/pipeline.example.yaml`），
覆盖顺序：**默认值 < `--config` 文件 < CLI**：

```bash
./build/app/yolo_rt --config app/pipeline.yaml                  # 纯配置文件
./build/app/yolo_rt --config app/pipeline.yaml --conf 0.35      # CLI 临时覆盖
./build/app/yolo_rt --config app/pipeline.yaml --dump-config    # 打印最终生效值
```

#### 配置文件参数总表

| 参数 | 含义 | 默认 |
|---|---|---|
| **模型契约** |||
| `param` / `model` | ncnn `.param` / `.bin` 路径 | 必填 |
| `in0` | 输入 blob 名 | `in0` |
| `model-type` | `v5`=切头契约 `[4+1+nc,N]`；`v8`=v8/v11 检测（Detect 头烘在图内）`[4+nc,N]`；`v8-pose`=姿态 `[4+nc+3K,N]` | `v5` |
| `heads` | 输出 blob 名列表。v5：三个，按 stride 8→16→32；v8/pose：通常单个拼接输出 | 必填 |
| `in-size` | 网络输入 `WxH`，须被 32 整除；letterbox 与 anchor 网格的唯一来源 | `640x640` |
| `num-classes` | 类别数 nc；加载时校验 head 通道数（v5: 4+1+nc；v8: 4+nc；pose: 4+nc+3K） | `2` |
| `num-kpts` | 仅 `v8-pose`：每框关键点数（任意，角点扫描场景） | `0` |
| `anchors-s8/s16/s32` | 仅 `v5`：三级 anchor（w,h 平铺）。不给 = YOLOv5s P5 默认；训练开 autoanchor 时**必须从训练配置抄** | 空=默认 |
| `gpu` | Vulkan 设备序号（`vulkaninfo --summary` 查） | `0` |
| `cpu-threads` | ncnn CPU fallback 层线程数 | `1` |
| **检测阈值** |||
| `conf` | 置信度阈值 | `0.3` |
| `nms` | NMS IoU 阈值 | `0.6` |
| `topk` | NMS 输出上限 | `300` |
| **跟踪** |||
| `gate-px` | 关联门限下限（像素） | `120` |
| `gate-scale` | 门限随预测框尺寸放大系数 | `2.0` |
| `size-ratio` | 尺寸记忆过滤：面积比在 [1/r, r] 外的候选丢弃 | `3.0` |
| `size-memory` | 尺寸先验记忆帧数 | `120` |
| `coast` | 纯 Kalman 外推帧数上限，超过判 LOST | `5` |
| `max-misses` | 门限自适应系数上限 | `15` |
| **运行模式** |||
| `serial` | `true` = 单线程串行（M1/M3 验收）；`false` = 两线程流水线 | `false` |
| `cores-a` / `cores-b` | 线程A / 线程B 绑核列表（`0-3` 或 `0,2`；核数不足自动收缩）。**默认值是给 Vulkan 模式设计的；纯 CPU 两线程必须放宽 `cores-b`（如 `4-31`），否则推理线程挤 4 核，劣化 2~3 倍** | `0-3` / `4-7` |
| `max-w` / `max-h` | 采集缓冲上限（相机分辨率上限） | `1920` / `1080` |
| `max-frames` | 处理 N 帧后自动退出（0=不限，测试用） | `0` |
| **控制输出** |||
| `udp` | 确认轨迹 UDP 目标 `host:port`（`hits>=2 && misses==0` 才发）；注释掉不发 | 空 |
| **调试** |||
| `no-model` | 跳过 ncnn，线程B 假推理（冒烟验证调度/关机） | `false` |
| `fake-gpu-ms` | 假推理每帧睡眠毫秒数 | `10` |

#### 三种模型的配置与调用

**① YOLOv5 系（切头导出，anchor-based）**

导出：`best.pt → ONNX（切 head、过 onnxsim）→ pnnx → ncnn`（§5.2）。

```yaml
# pipeline_v5.yaml
param: weights/best.param
model: weights/best.bin
model-type: v5
heads: [out8, out16, out32]        # 三个 head，stride 8→16→32
in-size: 640x640
num-classes: 2
anchors-s8:  [10,13, 16,30, 33,23] # 从训练配置抄（开了 autoanchor 尤其要抄）
anchors-s16: [30,61, 62,45, 59,119]
anchors-s32: [116,90, 156,198, 373,326]
```

**② YOLOv8 / v11 检测（Detect 头烘在图内，anchor-free）**

导出：Ultralytics 默认 `model.export(format='onnx')`（DFL+网格解码已在图内，
GPU 上跑）`→ pnnx → ncnn`。输出为单个拼接 blob `[1, 4+nc, N]`。**不需要 anchor 表。**

```yaml
# pipeline_v8.yaml
param: weights/yolov8n.param
model: weights/yolov8n.bin
model-type: v8
heads: [output0]                   # 单个输出（名字以 pnnx 转换后为准）
in-size: 640x640
num-classes: 1
```

**③ YOLOv8 / v11 pose（角点/关键点扫描，点数任意）**

输出 `[1, 4+nc+3×nkpt, N]`，关键点 (x,y,conf) 像素。关键点随框过 NMS，
UDP 控制包附带被跟踪框命中当帧的关键点（≤32 点，coast/丢失帧 `kpt_count=0`）。

```yaml
# pipeline_pose.yaml
param: weights/pose.param
model: weights/pose.bin
model-type: v8-pose
heads: [output0]
in-size: 640x640
num-classes: 1
num-kpts: 7                        # 你的关键点数
```

三种类型共用同一调用：

```bash
./build/app/yolo_rt --config app/pipeline_v5.yaml     # 换成对应 yaml
```

**防大小不匹配**：加载时试跑一帧校验输出通道数（v5 还校验 anchor 总数）——
配置与模型不一致**启动即报错**，不会带病运行。

---

## 6. 验证

```bash
python tools/gen_test_vectors.py                 # 重新生成 golden(已随仓库生成)
./build/x86/test_x86 test_data                   # x86 数值对齐 + 基准
./build/nvidia/test_cuda test_data               # CUDA 数值对齐 + 基准
PYTHONPATH=build/nvidia python nvidia/tests/test_python_bindings.py
./build/nvidia/test_trt_plugin                   # TRT 插件 sanity
# app 流水线(yolo_rt):
python3 tools/gen_tracker_golden.py              # Tracker golden(import RK3588 原版 BoxKalman)
python3 tools/gen_v8_vectors.py                  # v8/v11 检测+姿态 golden(case7/8，已随仓库生成)
./build/app/test_tracker test_data               # 卡尔曼/门控数值对齐(误差<1e-3)
./build/app/test_queue                           # 槽位不变式 + 丢旧帧语义
./build/app/yolo_rt --no-model                   # 无相机冒烟(掉线重连路径 + 干净退出)
```

验收标准（§5/§12）：decode/NMS 输出与 numpy golden 数量精确相等、坐标 max abs err < 1e-2、score < 1e-3；预处理 < 1e-3。本机实测全部 PASS：decode 坐标 max err ≤ 2.5e-4（含 992×992 / 992×544 分辨率），NMS 集合逐位一致。

---

## 7. 目标机（7840HS）注意事项

- `-DYOLOOP_X86_ARCH=znver4`，`taskset -c 0-7` 绑物理核，`scaling_governor=performance`（§11）
- vulkan 模块需 glslangValidator + Vulkan SDK；本开发机已通过 lavapipe 软件 Vulkan 实测（case2_dense 1930→1913 与 golden 逐位一致；该冒烟测试为临时程序未入库，库本身已编译验证），780M 真机性能未测
- TRT 插件基于 IPluginV2DynamicExt（TRT 11.3 起 deprecated，长期建议迁移 IPluginV3）
- NMS GPU kernel 的 walk 阶段为单 warp 串行扫描（贪心算法固有），候选 >>2048 时可再优化
- 本机基准数字来自 Ryzen 7 9700X（Zen5），仅供量级参考，以 7840HS 实测为准（§12）
- 第二台实测机 Ryzen 9 7940HX（16C/32T 双 CCD）+ RTX 4060 Laptop：全模块构建通过
  （含 CUDA 13.4 / ncnn / Vulkan / TRT 插件）。app 流水线实测：Vulkan 两线程 77 FPS，
  纯 CPU 建议 `--serial --threads 8`（56.9ms / ≈17 FPS）；**纯 CPU 两线程必须放宽
  `cores-b`（默认 `4-7` 只适用于 Vulkan 模式）**，实测数据与排查过程见
  [app/README.md](./app/README.md)「实测补充：7940HX + RTX 4060 Laptop」
