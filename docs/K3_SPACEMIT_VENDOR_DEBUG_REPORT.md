# 致进迭时空：K3 SpaceMIT EP 模型兼容性及 FP16 卷积间歇性段错误调试报告

日期：2026-11-10

## 1. 问题摘要

我们在 SpacemiT K3 CoM260 IFX 上部署一个 YOLO 姿态检测模型时遇到两个独立问题：

1. **确定性问题**：SpaceMIT EP 对非等分 `Split` 的形状推导错误，导致 `Reshape_139` 编译失败。将 `Split_130`、`Split_336` 等价改为显式范围 `Slice` 后解决。
2. **间歇性问题**：自定义视频工具运行 FP16 模型时，EP 的 `spe_direct_conv2d` 在首次试推理阶段随机出现 `Load page fault`。对照实验最终定位到：**视频/OpenCV/OpenBLAS 初始化顺序与 EP 初始化并存是必要触发条件**。完成 EP 加载后再打开视频，并限制 OpenBLAS 为单线程后，当前连续测试均通过。

本报告提供复现命令、关键输出、每轮对照实验及最终修复方式，供贵司进一步分析 EP 内部坏地址产生的具体原因。

## 2. 测试环境

### 2.1 硬件和系统

| 项目 | 内容 |
|---|---|
| 开发板 | SpacemiT K3 CoM260 IFX |
| 通用核 | 8×X100，RVV 1.0，VLEN=256 |
| AI 核 | 8×A100，RVV 1.0，VLEN=1024，带 IME 扩展 |
| 系统 | Bianbu Linux，RISC-V 64 |
| 内核 | 6.18.3-generic，PREEMPT |
| C++ 编译器 | GCC 15.2.0 |
| OpenCV/FFmpeg/OpenBLAS | 使用系统库 |

### 2.2 运行时版本

```text
spacemit-onnxruntime: 2.0.7+2
spacemit-runtime:     0.6.3
libonnxruntime:       libonnxruntime.so.1.24.2+spacemit.a1
libspacemit_ep:       libspacemit_ep.so.2.0.7
libspert:             libspert.so.0.6.3
```

APT 查询时，已安装版本和候选版本均为 `spacemit-onnxruntime 2.0.7+2`。

运行库链接目标：

```text
/usr/lib/libspacemit_ep.so.2 -> /usr/lib/libspacemit_ep.so.2.0.7
/usr/lib/libonnxruntime.so.1 -> /usr/lib/libonnxruntime.so.1.24.2+spacemit.a1
/usr/lib/libspert.so.1       -> /usr/lib/libspert.so.0.6.3
```

校验值：

```text
9f7d1dde2941e2a4b584b163a50189e0  /usr/lib/libspacemit_ep.so.2.0.7
9167f802d1c7dbc891782cb4b6f214d5  /usr/lib/libspert.so.1
```

### 2.3 模型和输入

| 项目 | 内容 |
|---|---|
| 模型任务 | 单类别、4 关键点姿态检测 |
| 输入 | `[1,3,640,640]`，RGB，CHW float32 |
| 输出 | `[1,17,8400]`，N=8400，C=17 |
| ONNX 输入/输出 | `images` / `output0` |
| 最终测试模型 | `best.spacemit.fp16.onnx`，约 19 MB |
| 对照模型 | `best.ncnn.param`、`best.ncnn.bin` |
| 测试视频 | 1447×1080 WebM，500 帧用于正式基准 |

### 2.4 模型转换环境

```text
Python 3.11
Torch 2.5.1+cu121
XSlim 2.1.2
```

## 3. 自定义工具与官方工具的结构差异

### 3.1 官方对照工具

`onnxruntime_perf_test` 的结构为：

```text
创建 ONNX Runtime Session
→ 准备测试输入
→ SpaceMIT EP 执行模型
→ 输出推理耗时
```

它不包含视频打开、FFmpeg 解码、OpenCV/OpenBLAS 初始化、RVV 后处理、Tracker 或多帧流水线。

### 3.2 自定义工具

自定义 `bench_pipeline` 的结构为：

```text
OpenCV VideoCapture / FFmpeg 视频解码
→ 自研 OrtInfer / ONNX Runtime / SpaceMIT EP
→ RVV letterbox、Decode、NMS
→ Kalman Tracker
→ 分阶段耗时统计
```

相关线程和库包括：

- OpenCV/FFmpeg 解码线程；
- OpenBLAS 工作线程；
- ONNX Runtime 与 EP 工作线程；
- 自研流水线线程。

官方工具与自定义工具使用同一个 ONNX 模型和同一套核心 EP 库。区别主要在于**外围初始化环境**，而不是模型文件不同。

## 4. 问题一：非等分 Split 导致 Reshape 编译失败

### 4.1 首次错误输出

```text
Non-zero status code returned while running SpineSubgraph_..._1 node.
Status Message: operator compile failed at node (Reshape_139)[Reshape],
error message: The input tensor cannot be reshaped to the requested shape.
```

### 4.2 EP 子图证据

原始 ONNX 中：

```text
Split_130:
  axis=2
  split=[32,32,64]
```

EP 导出的子图中：

```text
onnx::MatMul_336 shape = [1,4,42,400]
Reshape_139 target   = [1,256,20,20]
```

元素数量分别为：

```text
1×4×42×400 = 67200
1×256×20×20 = 102400
```

因此 EP 对非等分 Split 的形状推导不正确，而不是模型本身的 Reshape 数学关系错误。

仅禁用 `Split_130` 后，错误继续移动至 `Reshape_337`，说明 `Split_336` 也存在同类问题。

### 4.3 修复方法

新增 `tools/fix_spacemit_split.py`，将非等分 Split 改为带显式 starts/ends/axes 的 Slice：

```bash
python tools/fix_spacemit_split.py \
  ../data/best.onnx \
  ../data/best.spacemit.onnx
```

输出：

```text
转换 Split_130: split=[32,32,64] -> Slice
转换 Split_336: split=[64,1] -> Slice
已保存: ../data/best.spacemit.onnx
```

修改后无需禁用节点，EP 可以正常加载模型。

### 4.4 请厂商确认

建议贵司检查 EP 对以下 Split 形式的 shape inference：

```text
Split(axis=2, split=[32,32,64])
Split(axis=1, split=[64,1])
```

当前现象表明，它可能被错误地按输出数量近似均分。

## 5. FP32、动态量化和 FP16 对照

### 5.1 FP32/动态量化

FP32 和动态量化模型的官方工具结果约为：

```text
Average inference time cost total: 394.69 ms
Number of inferences per second: 2.53
```

自定义视频结果约为 398 ms，约 2.3 FPS。动态量化没有带来速度提升，因此未采用。

### 5.2 FP16 转换

```bash
python -m xslim \
  -i ../data/best.spacemit.onnx \
  -o ../data/best.spacemit.fp16.onnx \
  --fp16
```

官方工具输出：

```text
Average inference time cost total: 34.386596 ms
Number of inferences per second: 29.0748
```

这证明 FP16 是当前有效的加速路径，但与后面的间歇性崩溃属于两个独立问题。

## 6. 问题二：FP16 direct_conv2d 间歇性段错误

### 6.1 故障现象

同一模型、同一输入配置下，连续启动时成功与失败交替出现。失败发生在模型加载日志打印之前，返回码为 139，即 128+SIGSEGV(11)。

### 6.2 GDB 关键输出

GDB 显示：

```text
Thread 20 "bench_pipeline" received signal SIGSEGV, Segmentation fault.

void mlir::speir::spekernel::ime2::spe_direct_conv2d<
  _Float16, _Float16, _Float16, ...>(...)
```

调用路径为：

```text
spe_direct_conv2d
→ ComputeDirectConv2DBlockThreaded
→ SpineConv2DDispatchDirectConv::operator()
→ spert::detail::tile_trampoline
```

主线程则停在：

```text
spert::detail::sync_impl
→ SpineGraphExecutor::Predict
→ InferenceSession::Run
→ OrtApis::Run
→ yoloop::OrtInfer::Impl::load
```

这证明崩溃发生在**模型加载阶段的首次试推理**，不是后续视频帧、Decode、NMS 或 Tracker 阶段。

### 6.3 内核日志证据

`dmesg` 中同时可见：

```text
hmp_set_ai_thread: pid:...(bench_pipeline), vector has been enabled already!!!
...
bench_pipeline[...]: unhandled signal 11 code 0x1 at ... in libspacemit_ep.so.2.0.7
...
cause: 000000000000000d
badaddr: ...f8000
```

出错指令为：

```text
vle16.v v3, (a3)
```

其中 `a3` 与 `badaddr` 相同。`cause=0xd` 是 **Load page fault**，说明 EP 的 FP16 向量读访问命中了未映射页，不是 Store/AMO page fault。

`vector has been enabled already` 只是伴随信息。单独依据该提示，不能证明异常迁移或核间向量状态失步。

## 7. 对照实验全过程

### 7.1 自定义程序 EP 单线程 50 次冷启动

测试设置：

```bash
SPACEMIT_EP_INTRA_THREAD_NUM=1
SPACEMIT_EP_INTER_THREAD_NUM=1
SPACEMIT_EP_INTRA_THREAD_AFFINITY=8
```

每次运行 1 帧，连续启动 50 次。

结果：

```text
SINGLE_THREAD_RESULT ok=46 fail=4
```

结论：单线程不能彻底消除问题，因此不能仅归因于多线程竞争。

### 7.2 官方工具 50 次冷启动

```bash
onnxruntime_perf_test ../data/best.spacemit.fp16.onnx \
  -e spacemit -r 1 -x 1 -c 1 -S 1 -I
```

每次启动一个新进程，共 50 次。

结果：

```text
OFFICIAL_TOOL_RESULT ok=50 fail=0
```

结论：同一模型在官方单用途路径下稳定，说明需要重点检查自定义工具的初始化环境，而不是直接认定模型或 EP 必然不可用。

### 7.3 核对运行库

自定义工具链接：

```text
libonnxruntime.so.1 => /usr/lib/libonnxruntime.so.1
libspacemit_ep.so.2 => /usr/lib/libspacemit_ep.so.2
libspert.so.1       => /usr/lib/libspert.so.1
libspacemit_mpp.so.0
libopenblas.so.0
```

官方工具的直接依赖中只显示：

```text
libonnxruntime.so.1 => /lib/libonnxruntime.so.1
```

EP 由运行时继续加载。两边核心 ORT/EP 库版本一致，但自定义工具还引入了 OpenCV/OpenBLAS 初始化链路。

### 7.4 脱离视频，仅加载模型 50 次

新增 `--load-only`：

```text
只创建 Session 并执行首次试推理，不打开视频。
```

结果：

```text
LOAD_ONLY_RESULT ok=50 fail=0
```

结论：脱离视频初始化后，自定义 `OrtInfer` 的模型加载稳定。

### 7.5 调整初始化顺序

将自定义工具从：

```text
先打开视频 → 再加载 EP
```

改为：

```text
先加载 EP → 再打开视频
```

结果：

```text
单线程完整视频：REORDERED_VIDEO_RESULT ok=50 fail=0
8 线程完整视频：REORDERED_8_THREAD_RESULT ok=30 fail=0
```

结论：初始化顺序明确影响触发概率。

### 7.6 使用同一二进制强制恢复旧顺序

新增 `--video-first`，强制先打开视频，再加载 EP：

结果：

```text
旧顺序：CONTROL mode=video-first ok=9 fail=21
新顺序：CONTROL mode=normal      ok=30 fail=0
```

这是在同一个二进制、同一个模型和同一视频下完成的直接对照，排除了简单版本差异。

### 7.7 glibc 大块内存分配实验

```bash
GLIBC_TUNABLES=glibc.malloc.mmap_threshold=33554432 \
  ./build-k3/app/bench_pipeline ... --video-first
```

结果：

```text
VARIANT mode=glibc ok=0 fail=30
```

结论：该参数不能规避问题，不能把原因简单归为 glibc 大块 mmap 分配。

### 7.8 FFmpeg 单线程实验

```bash
OPENCV_FFMPEG_CAPTURE_OPTIONS=threads=1 \
  ./build-k3/app/bench_pipeline ... --video-first
```

结果：

```text
VARIANT mode=ffmpeg1 ok=15 fail=15
```

结论：不是 FFmpeg 解码线程数量本身导致。

### 7.9 OpenBLAS 单线程实验

```bash
OPENBLAS_NUM_THREADS=1 \
  ./build-k3/app/bench_pipeline ... --video-first
```

结果：

```text
VARIANT mode=openblas1 ok=30 fail=0
```

结论：OpenBLAS 多线程初始化/状态与 EP 初始化并存，是目前最关键的触发条件。官方工具没有自定义工具中的 OpenCV/OpenBLAS 初始化链路，因此不会以相同方式触发。

## 8. 最终修复

### 8.1 初始化顺序

自定义工具现在按以下顺序启动：

```text
绑定宿主 CPU 核
→ 创建 ORT Session
→ 加载 EP 并完成首次试推理
→ 再打开视频并创建解码线程
```

### 8.2 OpenBLAS 线程限制

启动脚本默认：

```bash
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
```

这只限制 OpenBLAS 自身线程池，不改变 EP 的 `SPACEMIT_EP_INTRA_THREAD_NUM=8`。

### 8.3 不再依赖重试

找到触发条件后，已移除自动重试，避免把“前两次崩溃、第三次成功”的容错行为误认为稳定修复。

保留以下诊断入口：

- `--load-only`：脱离视频，仅加载模型；
- `--video-first`：恢复旧初始化顺序以复现问题；
- `YOLOOP_ORT_DEBUG_ADDR`：打印首次试推理输入地址。

## 9. 最终验证结果

最终执行：

```bash
./run_k3_video.sh
```

500 帧完整视频输出：

```text
[spacemit-ort] 加载完成: .../best.spacemit.fp16.onnx,
                head=v8-pose, N=8400, C=17, EP线程=8, 直接输出绑定=1

== 串行模式逐阶段（500 帧）==
  letterbox    mean=18.936ms  P50=19.103ms  P99=22.058ms
  infer        mean=34.785ms  P50=34.656ms  P99=36.001ms
  decode+nms   mean= 0.019ms  P50= 0.020ms  P99= 0.031ms
  tracker      mean= 0.008ms  P50= 0.011ms  P99= 0.014ms
  端到端       mean=53.749ms  P50=53.812ms  P99=57.672ms
  理论吞吐上限 ≈ 18.6 FPS
```

最终脚本另连续运行 5 次，均成功。

作为对照，原 ncnn CPU 结果约为：

```text
infer      mean=268.266 ms
端到端     mean=287.372 ms
理论吞吐上限 ≈ 3.5 FPS
```

## 10. 当前结论边界

目前能够确认的是：

1. 非等分 Split 的 EP 形状推导错误是 Reshape 失败的直接原因。
2. 自定义工具的间歇性崩溃不是启动命令、视频内容或上一帧数据未丢弃导致。
3. OpenBLAS 多线程初始化状态与 EP 初始化顺序共同构成关键触发条件。
4. 调整初始化顺序并限制 OpenBLAS 后，当前测试已稳定。

目前不能据此断言：

- 某个 A100 核存在硬件故障；
- OpenBLAS 或 EP 单独存在必然错误；
- `vector has been enabled already` 就是直接根因；
- EP 内部具体哪一个 tile 指针、stride 或生命周期管理发生错误。

要定位到 `libspacemit_ep` 内部的具体坏地址生成逻辑，仍需贵司结合源码和调试符号继续分析。

## 11. 建议厂商协助事项

1. 检查非等分 Split 的 shape inference，避免按输出数量近似均分。
2. 检查 `spe_direct_conv2d` 的 tile 指针、stride、映射长度和缓冲区生命周期。
3. 评估 EP 与外部 OpenBLAS 线程池同时初始化时是否存在共享状态、线程池或内存管理冲突。
4. 提供带调试符号的 EP/Spine 包，或增加 conv 节点名、tile 地址、stride 和映射长度日志。
5. 如提供修复版本，请提供完整配套的 ONNX Runtime、EP、Spine 和运行时包，避免 ABI 不匹配。

## 12. 可复现日志获取方式

### 12.1 EP 子图和 profile

```bash
SPACEMIT_EP_DUMP_SUBGRAPHS=1 \
SPACEMIT_EP_DEBUG_PROFILE=debug_ \
FRAMES=10 ./run_k3_video.sh
```

生成：

- `SpaceMITExecutionProvider_*.onnx`：EP 子图；
- `debug_*.json`：算子执行时间。

### 12.2 中间张量

```bash
SPACEMIT_EP_DUMP_TENSORS=dump FRAMES=1 ./run_k3_video.sh
```

会生成 `.npy` 中间张量，文件较大且会降低速度，只建议短测试。

### 12.3 GDB

```bash
gdb --batch \
  -ex run \
  -ex 'thread apply all bt 20' \
  --args \
  taskset -c 0-7 ./build-k3/app/bench_pipeline \
  --backend spacemit \
  --model ../data/best.spacemit.fp16.onnx \
  --in0 images \
  --heads output0 \
  --model-type v8-pose \
  --num-classes 1 \
  --num-kpts 4 \
  --video "../data/Screencast from 2026-10-13 20-00-08.webm" \
  --frames 1 \
  --serial \
  --gpu -1 \
  --threads 8 \
  --cores-a 0-7
```

如强制使用旧初始化顺序，在命令末尾增加：

```bash
--video-first
```

### 12.4 内核日志

```bash
sudo dmesg | tail -80
```

重点保留 `badaddr`、`cause`、出错指令、EP 库名和进程 PID，便于与 GDB 对照。
