# K3 调试结论摘要

更新：2026-11-12。已连接设备重新复现，并补做 ELF 依赖、完整模型 opset 对照、GDB 入口参数和 TCM 保护页实验。详细证据见 [根因复核报告](K3_ROOT_CAUSE_ANALYSIS_2026-11-12.md)。

## 1. Split：旧版属性形式进入 EP 后形状错误

原始 opset 12 模型的 `Split_130` 声明 `split=[32,32,64]`。EP 导出节点仍保留该属性，却把输出形状标为 `[1,4,43,400]`、`[1,4,43,400]`、`[1,4,42,400]`。

最后一支只有 67,200 个元素，不能 reshape 为 `[1,256,20,20]` 所需的 102,400 个元素。因此 `Reshape_139` 是下游报错点，源头是 EP 的 Split 形状处理。

历史上用 `tools/fix_spacemit_split.py` 将 `Split_130` 和 `Split_336` 改为 Slice，可以规避。本次还验证：**同权重完整模型转为 opset 13、分段改为第二输入后，EP 输出形状正确且首次推理通过。** 本地 CPU 单样本输出与原模型逐元素一致。

当前 FP16 模型实际为 opset 24，仍含第二输入形式的非等分 Split。不能把结论泛化为“EP 不支持所有非等分 Split”。

## 2. FP16 卷积：已确认越过 TCM 边界读取

自然崩溃和保护页实验都定位到：

```text
libspacemit_ep.so.2.0.7
ime2::spe_direct_conv2d<...> +402 字节
库内地址 0x3fb78e：vle16.v v3, (a3)
```

本次入口参数/映射证据：

```text
TCM 范围：[0x3f73b98000, 0x3f73bf8000)，384 KiB
第一张量起点：0x3f73be0000
故障读取地址：0x3f73bf8000 = 第一张量起点 + 96 KiB = TCM 末端
```

布局与 288 KiB 权重暂存 + 96 KiB 输入暂存吻合。已确认内核读取越过有效 TCM 区域，具体是尾块处理、padding/halo、stride 还是空间规划错误，仍需厂商源码确定。

**给每个 TCM 块加不可访问保护页后，厂商 RVV OpenCV 和系统 OpenCV 各 3/3 次崩溃，OpenBLAS 均设为单线程。** 厂商配置也停在同一指令和相同相对偏移。因此换库和调整顺序没有消除这次越界。

历史 `cause=0xd` 表示 Load page fault。不要直接把它等同于“进程中该地址完全没有 VMA”：本次自然故障的 TCM 后方仍出现匿名 VMA，但实际访问失败；保护页故障则明确为 `SEGV_ACCERR`。

## 3. RVV OpenCV / OpenBLAS 为什么规避

系统 OpenCV 4.10 使用外部 BLAS/LAPACK、TBB；厂商 OpenCV 5.0 使用内置 libclapack、pthreads，并关闭 GStreamer。厂商程序在不预加载时，依赖链中没有 OpenBLAS。系统 OpenBLAS 本身也已选择 `riscv64_zvl256b` 向量内核，不能把问题简单归因于“系统库不支持 RVV”。

本次均使用旧顺序 `--video-first`，EP 8 线程：

| 配置 | 成功 / 冷启动 |
|---|---:|
| 系统 OpenCV + 系统 OpenBLAS 8 线程 | 4/10，6 次 SIGSEGV |
| 系统 OpenCV + 系统 OpenBLAS 1 线程 | 10/10 |
| 厂商 OpenCV，不加载 OpenBLAS | 10/10 |
| 厂商 OpenCV + 预加载系统 OpenBLAS 8 线程 | 10/10 |
| 厂商 OpenCV + 预加载厂商 OpenBLAS 8 线程 | 10/10 |
| 厂商 OpenCV + 预加载厂商 OpenBLAS 1 线程 | 10/10 |

结合保护页实验，最有根据的解释是：**换库、线程数与初始化顺序改变依赖及内存布局，掩盖或暴露 EP 的越界读取。** OpenBLAS 多线程是一个触发因素，不能再直接称其为内部根因；RVV OpenBLAS 也没有替换 EP 出错的卷积内核。

另发现历史“仅替换 RVV OpenBLAS”的 60 份日志全部为：

```text
libblas.so.3: undefined symbol: gotoblas
```

系统 BLAS/LAPACK 拆分库依赖 DYNAMIC_ARCH OpenBLAS 内部符号，厂商固定 x100 目标库不导出该符号。这是独立的链接兼容错误，程序没有进入推理，不能用于判断卷积崩溃是否被解决。

## 4. 当前保留的规避措施

```text
先创建 ORT Session、完成首次试推理
→ 再打开视频并创建解码线程
```

同时保留厂商 OpenCV 配置及：

```bash
OPENBLAS_NUM_THREADS=1
```

这些是规避措施，不能标为 EP 根因修复。自动重试仍不应当作修复。原有厂商配置新、旧顺序各 100 次成功的记录，可以说明当时的运行稳定性，但不能反证已由保护页揭示的越界。

本次没有修改部署代码、模型或系统库。尚未验证越界是否影响最终输出数值。

## 5. `OrtInfer` 与待厂商修复事项

`OrtInfer` 是本项目业务适配层，负责官方 EP 初始化、输入输出校验和缓冲区管理。出错的首次推理位于 `app/ort_infer.cpp` 的普通 `session->Run`，尚未进入业务 Decode/NMS/Tracker；实际卷积由官方 EP/Spine 执行。

厂商需处理：

- opset 12 Split 属性形式的形状推导兼容性。
- `spe_direct_conv2d+402` 的 TCM 访问边界，以及尾块/padding/调度与缓冲区容量之间的一致性。
- 使用保护页和数值比对验收，不能只以冷启动不崩溃作为修复标准。

日志、实验脚本和保护页探针见 [证据目录](evidence/k3-2026-11-12/README.md)。
