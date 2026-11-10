# K3 RVV 与 SpaceMIT EP 部署、调试和验证报告

日期：2026-11-10

## 1. 报告范围

本报告记录以下工作的实现、测试和故障排查过程：

- 为 `YOLO_operator_RC` 增加 RISC-V RVV 后端；
- 在 K3 上运行 ncnn 基线；
- 接入 SpacemiT-ONNXRuntime 和 SpaceMITExecutionProvider；
- 修复 ONNX 非等分 Split 导致的 EP Reshape 编译失败；
- 排查自定义视频工具中间歇性段错误；
- 验证最终 FP16 视频推理性能。

报告不包含任何 SSH 地址、密码或其他连接凭据。

## 2. 测试环境

### 2.1 PC 开发环境

| 项目 | 环境 |
|---|---|
| 用途 | C++ 开发、x86/RVV 编译检查、ONNX 图修改、XSlim 转换、x86 对照测试 |
| CPU 架构 | x86_64 |
| GCC | 13.3.0 |
| Clang | 18.1.3 |
| CMake | 3.28.3 |
| 模型转换环境 | `gputest`，Python 3.11、Torch 2.5.1+cu121、XSlim 2.1.2 |

PC 上的 `test_rvv` 使用标量回退验证算法，同时使用 Clang 生成 `rv64gcv` 汇编，确认存在 `vsetvli`、`vcompress.vm`、`vloxei32.v` 等 RVV 指令。PC 本身不能代替 K3 的实机性能和 EP 验证。

### 2.2 K3 实机环境

| 项目 | 环境 |
|---|---|
| 开发板 | SpacemiT K3 CoM260 IFX |
| 通用核 | 8 个 X100，RVV 1.0，VLEN=256 |
| AI 核 | 8 个 A100，RVV 1.0，VLEN=1024，带 IME 扩展 |
| 系统 | Bianbu Linux，RISC-V 64 |
| 内核 | 6.18.3-generic，PREEMPT |
| C++ 编译器 | GCC 15.2.0 |
| ncnn | FetchContent 标签 `20250503` |
| SpaceMIT ORT | `spacemit-onnxruntime 2.0.7+2` |
| Spine Runtime | `spacemit-runtime 0.6.3` |
| ONNX Runtime 库 | `libonnxruntime.so.1.24.2+spacemit.a1` |
| EP 库 | `libspacemit_ep.so.2.0.7` |
| 其他 | 系统 OpenCV、FFmpeg、OpenBLAS |

### 2.3 模型与输入

| 项目 | 内容 |
|---|---|
| 任务 | 单类别、4 关键点姿态检测 |
| 输入 | 3×640×640，RGB，CHW float32 |
| 原始模型 | `best.pt` / `best.onnx` |
| ncnn 模型 | `best.ncnn.param` / `best.ncnn.bin` |
| 最终 EP 模型 | `best.spacemit.fp16.onnx`，约 19 MB |
| 输出 | `[1,17,8400]`，N=8400，C=17 |
| ONNX 输入/输出名 | `images` / `output0` |
| ncnn 输入/输出名 | `in0` / `out0` |
| 测试视频 | `Screencast from 2026-10-13 20-00-08.webm` |
| 视频规格 | 1447×1080，500 帧用于正式基准 |

## 3. 当前程序结构

### 3.1 ncnn 基线

```text
视频/OpenCV
  → RVV letterbox
  → ncnn CPU 推理
  → RVV Decode/NMS
  → Kalman Tracker
  → 分阶段耗时统计
```

### 3.2 SpaceMIT EP 路径

```text
先创建 OrtInfer / ORT Session
  → SpaceMIT EP 完成模型加载和首次试推理
  → 再打开视频并创建解码线程
  → RVV letterbox
  → EP 执行网络主体
  → RVV Decode/NMS
  → Kalman Tracker
```

`OrtInfer` 支持：

- `SessionOptionsSpaceMITEnvInit`；
- 输入/输出名称和 head 契约校验；
- `Ort::IoBinding` 输入输出绑定；
- 单输出 channel-major 时直接绑定现有 `pred`；
- 其他布局使用一次性分配的暂存区；
- `SPACEMIT_EP_*` 环境变量配置。

## 4. 构建与运行命令

### 4.1 K3 编译和测试

```bash
cd /home/epoch/work/k3/YOLO_operator_RC
./build_k3.sh
```

作用：

- 使用 `rv64gcv` 和 `lp64d`；
- 启用 RISC-V 后端，关闭 x86、CUDA 和 Vulkan；
- 自动探测 SpaceMIT ONNX Runtime；
- 编译 `test_rvv`、`yolo_rt`、`bench_pipeline` 等；
- 运行 golden、Tracker、队列和 ncnn 布局测试。

结果：

- v5、v8/v9/v11、pose、raw DFL、dist、v10 全部通过；
- Tracker 42 帧回放通过；
- v5 anchor 重排及负例测试通过；
- 最终输出 `ALL PASS`。

本次 K3 算子基准示例：

| 算子 | 平均耗时 |
|---|---:|
| Decode，N=25200 | 741.7 us |
| Decode AfRaw，N=8400 | 297.3 us |
| Decode AfDist，N=8400 | 271.3 us |
| NMS，TopK=300 | 193.6 us |
| 1280×720→640 预处理 | 2111.6 us |

只编译、不跑测试：

```bash
./build_k3.sh --no-test
```

指定编译并行度：

```bash
JOBS=16 ./build_k3.sh
```

### 4.2 视频测试

默认命令：

```bash
./run_k3_video.sh
```

作用：

- 默认使用 `best.spacemit.fp16.onnx` 和 SpaceMIT EP；
- 先完成构建检查；
- 在 EP 加载完成后才打开视频；
- 使用 `taskset -c 0-7` 固定 X100 宿主线程；
- 默认 `OPENBLAS_NUM_THREADS=1`；
- 测试 500 帧。

最终结果：

```text
letterbox  mean=18.936 ms
infer      mean=34.785 ms
decode+nms mean= 0.019 ms
tracker    mean= 0.008 ms
端到端     mean=53.749 ms
理论吞吐上限 ≈ 18.6 FPS
```

短测试：

```bash
FRAMES=10 ./run_k3_video.sh
```

ncnn 对照基线：

```bash
BACKEND=ncnn ./run_k3_video.sh
```

原 ncnn 结果：

```text
infer(GPU) mean=268.266 ms
端到端      mean=287.372 ms
理论吞吐上限 ≈ 3.5 FPS
```

日志中的 `infer(GPU)` 是旧计时标签；`Vulkan=0` 和 `--gpu -1` 表明实际为 ncnn CPU 推理。现在标签已改为 `infer`。

### 4.3 EP 并行度测试

```bash
THREADS=1 FRAMES=100 ./run_k3_video.sh
THREADS=2 FRAMES=100 ./run_k3_video.sh
THREADS=4 FRAMES=100 ./run_k3_video.sh
THREADS=8 FRAMES=100 ./run_k3_video.sh
```

这些命令调整 `SPACEMIT_EP_INTRA_THREAD_NUM`。K3 有 8 个 A100 核，超过 8 通常没有收益。

显式指定 A100 核：

```bash
SPACEMIT_EP_INTRA_THREAD_AFFINITY='8;9;10;11;12;13;14;15' \
THREADS=8 ./run_k3_video.sh
```

### 4.4 单独验证模型加载

```bash
./build-k3/app/bench_pipeline \
  --backend spacemit \
  --model ../data/best.spacemit.fp16.onnx \
  --in0 images \
  --heads output0 \
  --model-type v8-pose \
  --num-classes 1 \
  --num-kpts 4 \
  --load-only
```

作用：只创建 Session 并执行首次试推理，不打开视频。此路径连续 50 次全部成功。

`--video-first` 用于故障复现：强制恢复“先打开视频、再初始化 EP”的旧顺序。

### 4.5 官方性能工具对照

```bash
onnxruntime_perf_test ../data/best.spacemit.fp16.onnx \
  -e spacemit -r 50 -x 8 -c 1 -S 1 -s -I
```

作用：使用官方工具测试同一 FP16 模型，不包含视频、Tracker 或自定义预处理。

结果示例：

```text
Average inference time cost total: 34.386596 ms
Number of inferences per second: 29.0748
```

官方工具连续 50 次全部成功，证明模型可以在官方单用途路径下运行。

## 5. 模型修复和转换

### 5.1 非等分 Split 修复

新增 `tools/fix_spacemit_split.py`，把非等分 Split 改为具有明确 starts/ends/axes 的 Slice：

```bash
/home/mtftau-5/miniconda3/envs/gputest/bin/python \
  tools/fix_spacemit_split.py \
  ../data/best.onnx \
  ../data/best.spacemit.onnx
```

实际转换：

```text
Split_130: split=[32,32,64] -> Slice
Split_336: split=[64,1] -> Slice
```

该步骤只修改计算图，不改变模型数学含义或权重。

### 5.2 FP16 转换

```bash
/home/mtftau-5/miniconda3/envs/gputest/bin/python \
  -m xslim \
  -i ../data/best.spacemit.onnx \
  -o ../data/best.spacemit.fp16.onnx \
  --fp16
```

结果：

- 模型从约 38 MB 降为约 19 MB；
- 官方工具推理从约 395 ms 降为约 34.4 ms；
- 自定义完整视频端到端约 54 ms。

### 5.3 动态量化结果

动态量化示例：

```bash
/home/mtftau-5/miniconda3/envs/gputest/bin/python \
  -m xslim \
  -i ../data/best.spacemit.onnx \
  -o ../data/best.spacemit.dynq.onnx \
  --dynq
```

动态量化没有带来速度提升，约 398 ms/帧，因此没有采用。静态 INT8 需要另外准备 200～1000 张真实场景校准图，并进行精度回归，不能把动态量化结果当作 INT8 已验证完成。

## 6. 故障 1：Reshape_139 编译失败

### 6.1 错误

首次接入 EP 时出现：

```text
operator compile failed at node (Reshape_139)[Reshape]
The input tensor cannot be reshaped to the requested shape.
```

### 6.2 定位

导出 EP 子图后确认：

- `Split_130` 的实际分段为 `[32,32,64]`；
- EP 却将 `onnx::MatMul_336` 推导为 `[1,4,42,400]`；
- 随后 Reshape 到 `[1,256,20,20]`，元素数量不一致。

模型本身合法，属于 EP 对非等分 Split 的形状推导错误。

### 6.3 结论

将 `Split_130` 和 `Split_336` 改为 Slice 后，模型加载成功。单纯的 `SPACEMIT_EP_DISABLE_OP_NAME_FILTER` 虽可绕过，但会让节点回退 CPU，不是推荐主方案。

## 7. 故障 2：间歇性 FP16 direct_conv2d 段错误

### 7.1 错误性质

GDB 和内核日志显示：

```text
spe_direct_conv2d<_Float16, ...>
vle16.v
badaddr 指向未映射页
cause=0xd
```

`0xd` 是 Load page fault。错误指令是 FP16 向量读取，说明卷积线程读到了未映射页，不是写入页错误。

### 7.2 对照实验

| 实验 | 结果 | 结论 |
|---|---:|---|
| 自定义程序，EP 单线程 50 次 | 46 成功、4 失败 | 不是单纯多线程竞争 |
| 官方 `onnxruntime_perf_test` 50 次 | 50/50 成功 | 官方单用途路径稳定 |
| 自定义 `--load-only` 50 次 | 50/50 成功 | 脱离视频时模型加载稳定 |
| 先 EP、后视频，单线程 50 次 | 50/50 成功 | 初始化顺序相关 |
| 先 EP、后视频，8 线程 30 次 | 30/30 成功 | 多核也可稳定 |
| 旧顺序，8 线程 30 次 | 9 成功、21 失败 | 可复现旧顺序问题 |
| 旧顺序+glibc 分配调整 | 30/30 失败 | 无规避作用 |
| 旧顺序+FFmpeg 单线程 | 15 成功、15 失败 | 非 FFmpeg 线程数本身 |
| 旧顺序+OpenBLAS 单线程 | 30/30 成功 | OpenBLAS 初始化是关键触发条件 |

### 7.3 根因结论

最终确认的触发组合是：

```text
视频/OpenCV/OpenBLAS 相关初始化
→ OpenBLAS 多线程状态与 EP 初始化并存
→ EP FP16 direct_conv2d 异常读取
```

因此不能简单归因为：

- 用户启动命令错误；
- 上一次视频数据未丢弃；
- ONNX 输入形状固定错误；
- 某个 A100 核必然损坏；
- 单纯的 EP 多线程竞争。

更准确的结论是：**自定义工具的视频与 OpenBLAS 初始化环境影响了 EP 初始化，触发了其 FP16 卷积的异常读取。** EP 内部具体为何生成坏地址，仍需厂商结合 `libspacemit_ep` 源码继续分析。

## 8. 官方工具与自定义工具的差异

官方工具：

```text
创建 ORT Session
→ 准备测试输入
→ EP 执行模型
→ 输出耗时
```

自定义工具原来还包含：

- OpenCV/FFmpeg 视频初始化；
- OpenBLAS 初始化；
- 自研 `OrtInfer` 和缓冲区管理；
- RVV 预处理、Decode/NMS；
- Tracker 和流水线。

所以官方工具没有自定义工具的联合初始化环境。它稳定不代表完整视频应用必然稳定，但为定位提供了有效对照。

## 9. 最终修复

### 9.1 初始化顺序

`bench_pipeline` 已改为：

```text
绑定宿主 CPU 核
→ 创建 ORT Session、加载 EP、完成首次试推理
→ 再打开视频并创建解码线程
```

### 9.2 OpenBLAS 配置

`run_k3_video.sh` 默认：

```bash
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
```

这只限制 OpenBLAS 自己的线程池，不改变 `SPACEMIT_EP_INTRA_THREAD_NUM=8`。

### 9.3 移除临时 workaround

找到初始化问题后，已移除自动重试，不再把重试成功当作稳定修复。保留：

- `taskset -c 0-7`；
- C++ 模型加载前绑核；
- `--load-only` 诊断；
- `--video-first` 故障复现选项；
- `YOLOOP_ORT_DEBUG_ADDR` 地址诊断。

## 10. 最终结论

1. RVV 后端及现有 golden 测试全部通过。
2. 非等分 Split 导致的 Reshape 编译失败已通过图等价改写解决。
3. FP32 和动态量化没有带来理想加速，FP16 是当前有效路径。
4. 自定义工具的段错误不是用户操作问题，而是视频/OpenBLAS 与 EP 初始化顺序共同触发。
5. 最终采用“EP 先加载、视频后启动、OpenBLAS 单线程”的方案。
6. 最终 K3 500 帧完整视频测试约 **18.6 FPS**，相比 ncnn CPU 的约 **3.5 FPS** 明显提升。
7. 推理主体约 35 ms 后，letterbox 约 19 ms 已成为另一个主要优化对象。
8. 静态 INT8 和实际业务精度/mAP 仍需单独校准和验收；FP16 的速度验证不能代替精度验证。

## 11. 后续建议

- 用真实业务图片完成静态 INT8 校准和精度回归；
- 对比 `THREADS=1/2/4/8` 的稳定吞吐，不盲目增加线程；
- 优化 letterbox 查找表复用、RVV gather 或 V2D 预处理；
- 将 `hmp_set_ai_thread`、GDB 栈和本报告中的对照结果提交给进迭时空，继续定位 EP 内部坏地址的产生原因；
- 如升级 EP，必须使用完整配套的 release 包，避免单独替换 `libspacemit_ep.so` 造成 ABI 不匹配。
