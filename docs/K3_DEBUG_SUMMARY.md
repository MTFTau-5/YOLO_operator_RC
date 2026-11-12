# K3 调试结论摘要

## 1. 非等分 Split 导致 Reshape 失败

### 现象

```text
operator compile failed at node (Reshape_139)[Reshape]
The input tensor cannot be reshaped to the requested shape.
```

### 证据

原始 `Split_130` 的分段为 `[32,32,64]`，但 EP 子图将 `onnx::MatMul_336` 推导为 `[1,4,42,400]`，随后目标形状为 `[1,256,20,20]`，元素数量不一致。

### 结论与处理

这是 EP 对非等分 Split 的形状推导错误，不是模型数学关系错误。

使用 `tools/fix_spacemit_split.py` 将 `Split_130` 和 `Split_336` 改为显式范围的 Slice，问题消失。

## 2. FP16 direct_conv2d 间歇性段错误

### 现象

自定义工具首次试推理时随机返回 139。GDB 定位到：

```text
spe_direct_conv2d<_Float16, ...>
```

内核日志中的 `vle16.v` 与 `badaddr` 表明 EP 在 FP16 向量读取时访问了未映射页。`cause=0xd` 是 Load page fault，不是写入页错误。

### 对照实验

| 实验 | 结果 | 结论 |
|---|---|---|
| 自定义程序 EP 单线程 50 次 | 46 成功、4 失败 | 不是单纯多线程竞争 |
| 官方工具 50 次 | 50/50 成功 | 模型在官方单用途路径下稳定 |
| `--load-only` 50 次 | 50/50 成功 | 脱离视频后加载稳定 |
| 先 EP、后视频，单线程 50 次 | 50/50 成功 | 初始化顺序相关 |
| 先 EP、后视频，8 线程 30 次 | 30/30 成功 | 多核也可稳定 |
| 强制旧初始化顺序 30 次 | 9 成功、21 失败 | 旧顺序可复现问题 |
| glibc mmap 参数调整 | 30/30 失败 | 与 glibc 大块 mmap 无关 |
| FFmpeg 单线程 | 15 成功、15 失败 | 不是 FFmpeg 线程数本身 |
| OpenBLAS 单线程 | 30/30 成功 | OpenBLAS 多线程初始化是关键触发条件 |

### 最终修复

初始化顺序调整为：

```text
先创建 ORT Session、加载 EP、完成首次试推理
→ 再打开视频并创建解码线程
```

同时设置：

```bash
OPENBLAS_NUM_THREADS=1
```

自动重试已移除，不再把重试成功当作真正修复。

## 3. RVV OpenCV 与 OpenBLAS 的选择

仅替换 RVV OpenBLAS 没有解决初始化问题。将 `/opt/opencv-spacemit` 的 OpenCV 与 `/opt/openblas-spacemit` 的 OpenBLAS 同时启用后：

- 新、旧初始化顺序各 100 次冷启动均未崩溃；
- 预处理部分也有改善。

因此默认配置已改为优先使用这两个优化库，并保留初始化顺序和 OpenBLAS 单线程设置。

## 4. `OrtInfer` 的归属

`OrtInfer` 是本项目编写的业务适配层，不是官方组件。它负责：

- 调用官方 EP 初始化接口；
- 校验模型输入输出；
- 管理 Tensor 和输出布局；
- 将结果交给自研 Decode/NMS 和 Tracker。

实际推理由官方 ONNX Runtime 与 SpaceMIT EP 完成。

## 5. 仍需厂商确认的内部问题

当前已确认并消除触发条件，但 EP 内部具体为何产生坏地址仍需进迭时空结合源码确认：

- 非等分 Split 的 shape inference；
- `spe_direct_conv2d` 的 tile 指针、stride 和映射长度；
- EP 与外部 OpenBLAS 线程池同时初始化时的共享状态或资源管理。
