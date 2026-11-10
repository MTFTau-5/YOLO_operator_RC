# 致进迭时空：K3 SpaceMIT EP 调试报告

日期：2026-11-10

## 1. 问题概述

在 SpacemiT K3 CoM260 IFX 上部署 YOLO 姿态模型时，遇到两个独立问题：

1. 非等分 `Split` 导致 EP 对输出形状推导错误，`Reshape_139` 编译失败。
2. 自定义视频工具调用 FP16 模型时，`spe_direct_conv2d` 在首次试推理阶段间歇性出现 `Load page fault`。

以下只记录问题、调试证据和修复过程，不包含性能对比。

## 2. 必要环境信息

| 项目 | 内容 |
|---|---|
| 开发板 | SpacemiT K3 CoM260 IFX |
| 通用核 / AI 核 | 8×X100 / 8×A100 |
| 系统 | Bianbu Linux，RISC-V 64 |
| 内核 | 6.18.3-generic |
| SpaceMIT ORT | 2.0.7+2 |
| Spine Runtime | 0.6.3 |
| EP 库 | `libspacemit_ep.so.2.0.7` |
| ORT 库 | `libonnxruntime.so.1.24.2+spacemit.a1` |
| 模型输入 | `[1,3,640,640]` |
| 模型输出 | `[1,17,8400]` |
| 输入/输出名 | `images` / `output0` |

运行库校验：

```text
9f7d1dde2941e2a4b584b163a50189e0  /usr/lib/libspacemit_ep.so.2.0.7
9167f802d1c7dbc891782cb4b6f214d5  /usr/lib/libspert.so.1
```

## 3. 官方工具与自定义工具的关系

### 官方工具

`onnxruntime_perf_test` 只做：

```text
创建 Session → 准备输入 → 调用官方 EP 推理 → 输出结果
```

它不包含视频、OpenCV、OpenBLAS、RVV 后处理或 Tracker。

### 自定义工具

自定义工具在官方 ONNX Runtime 和 SpaceMIT EP 之外，还包括：

- OpenCV/FFmpeg 视频初始化；
- OpenBLAS 初始化；
- 视频解码、预处理和流水线；
- RVV Decode/NMS、Tracker。

其中 `OrtInfer` 是**本项目自研的集成适配层**，位于 `app/ort_infer.cpp`，不是官方组件。它只负责：

- 调用 `SessionOptionsSpaceMITEnvInit`；
- 校验输入输出名称和 head 契约；
- 使用 `Ort::IoBinding` 管理缓冲区；
- 将官方输出整理为自研后处理需要的布局。

真正的推理引擎仍是官方 ONNX Runtime 和 SpaceMIT EP。官方性能工具不包含这层业务适配代码。

## 4. 问题一：非等分 Split 导致 Reshape 失败

### 4.1 错误输出

```text
operator compile failed at node (Reshape_139)[Reshape]
The input tensor cannot be reshaped to the requested shape.
```

### 4.2 EP 子图证据

原始节点：

```text
Split_130:
  axis=2
  split=[32,32,64]
```

EP 导出的子图却包含：

```text
onnx::MatMul_336 = [1,4,42,400]
Reshape_139 target = [1,256,20,20]
```

两边元素数量分别为 67200 和 102400，因此无法 Reshape。说明 EP 将非等分 Split 错误地按近似均分处理，模型本身的数学关系没有问题。

禁用 `Split_130` 后，错误继续移动到 `Reshape_337`，确认 `Split_336` 也有同类问题。

### 4.3 修复

使用 `tools/fix_spacemit_split.py` 将非等分 Split 改为显式范围的 Slice：

```text
Split_130: split=[32,32,64] -> Slice
Split_336: split=[64,1] -> Slice
```

修改后无需禁用节点，模型可以正常加载。建议贵司检查 EP 对非等分 Split 的 shape inference。

## 5. 问题二：FP16 direct_conv2d 间歇性段错误

### 5.1 故障位置和日志

GDB 关键调用链：

```text
spe_direct_conv2d
→ ComputeDirectConv2DBlockThreaded
→ SpineConv2DDispatchDirectConv::operator()
→ spert::detail::tile_trampoline
```

主线程位于：

```text
SpineGraphExecutor::Predict
→ InferenceSession::Run
→ OrtApis::Run
→ yoloop::OrtInfer::Impl::load
```

因此崩溃发生在**首次模型试推理**，不是视频解码、Decode/NMS 或 Tracker 阶段。

内核日志中的出错指令为：

```text
vle16.v v3, (a3)
```

`a3` 与 `badaddr` 相同。`cause=0xd` 是 **Load page fault**，即 FP16 向量读取命中了未映射页。

`hmp_set_ai_thread: vector has been enabled already` 只是伴随信息，不能据此直接认定异常迁移或向量状态失步。

## 6. 每轮调试实验及结论

| 实验 | 结果 | 能说明什么 |
|---|---|---|
| 自定义程序，EP 单线程冷启动 50 次 | 46 成功、4 失败 | 不是单纯多线程竞争 |
| 官方工具冷启动 50 次 | 50/50 成功 | 同一模型在官方单用途路径下稳定 |
| 自定义 `--load-only` 50 次 | 50/50 成功 | 脱离视频后，自定义 OrtInfer 加载稳定 |
| 先 EP、后视频，单线程 50 次 | 50/50 成功 | 初始化顺序会影响问题触发 |
| 先 EP、后视频，8 线程 30 次 | 30/30 成功 | 多核也能在调整后稳定 |
| 使用 `--video-first` 强制旧顺序 30 次 | 9 成功、21 失败 | 旧顺序可以稳定复现问题 |
| 旧顺序+glibc mmap 参数调整 | 30/30 失败 | 不能归因于 glibc 大块 mmap |
| 旧顺序+FFmpeg 单线程 | 15 成功、15 失败 | 不是 FFmpeg 线程数本身 |
| 旧顺序+OpenBLAS 单线程 | 30/30 成功 | OpenBLAS 多线程初始化是关键触发条件 |

### 调试结论

问题不是：

- 用户启动命令错误；
- 上一次视频数据未清理；
- 某个固定输入形状错误；
- 某个 A100 核必然损坏；
- `OrtInfer` 取代了官方 EP。

目前确认的关键触发组合是：

```text
先启动视频/OpenCV/OpenBLAS 相关环境
→ OpenBLAS 多线程状态与 EP 初始化并存
→ EP FP16 direct_conv2d 异常读取未映射页
```

官方工具没有这些外围初始化步骤，所以不会以相同方式触发。

需要强调的是：**这确认了触发条件，但不等于已经证明 EP 内部具体哪一行代码产生坏地址。** 指针、stride、tile 映射或生命周期错误仍需结合 `libspacemit_ep` 源码确认。

## 7. 最终修复

### 初始化顺序

自定义工具改为：

```text
绑定宿主 CPU 核
→ 创建 ORT Session、加载 EP、完成首次试推理
→ 再打开视频并创建解码线程
```

### OpenBLAS 设置

```bash
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
```

这只限制 OpenBLAS 自己的线程池，不改变 EP 的线程数。

### 移除重试

找到触发条件后，已移除自动重试，避免把“前两次失败、第三次成功”的容错行为误认为已经修复。

保留两个诊断入口：

- `--load-only`：脱离视频，仅加载模型；
- `--video-first`：强制恢复旧初始化顺序以复现问题。

## 8. 需要厂商协助的事项

1. 检查非等分 Split 的 shape inference。
2. 检查 `spe_direct_conv2d` 的 tile 指针、stride、映射长度和缓冲区生命周期。
3. 检查 EP 与外部 OpenBLAS 线程池同时初始化时是否存在共享状态或资源管理冲突。
4. 提供带调试符号的 EP/Spine 包，或增加 conv 节点名、tile 地址、stride 和映射长度日志。
5. 如提供修复版本，请提供完整配套的 ONNX Runtime、EP、Spine 和运行时包，避免 ABI 不匹配。

## 9. 复现日志获取

EP 子图和 profile：

```bash
SPACEMIT_EP_DUMP_SUBGRAPHS=1 \
SPACEMIT_EP_DEBUG_PROFILE=debug_ \
FRAMES=10 ./run_k3_video.sh
```

中间张量：

```bash
SPACEMIT_EP_DUMP_TENSORS=dump FRAMES=1 ./run_k3_video.sh
```

GDB：

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

强制旧初始化顺序时，在命令末尾添加：

```bash
--video-first
```

内核日志：

```bash
sudo dmesg | tail -80
```
