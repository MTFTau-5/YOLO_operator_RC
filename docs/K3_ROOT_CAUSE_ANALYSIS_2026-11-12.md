# K3 根因复核：Split 兼容错误、TCM 越界读取与 RVV 库的规避机制

复核日期：2026-11-12。检查本地模型与源码，并连接 K3 实测。所有新增设备侧程序、模型和日志位于 `/tmp/k3-rootcause-20261112`；没有修改设备的部署程序、系统库或默认运行配置。

## 1. 结论

1. **Split：EP 对旧版 `split` 属性形式的形状处理错误。** 完整 opset 12 模型的 `Split_130` 声明 `[32,32,64]`，EP 导出的子图却标注 `[43,43,42]`。转换为 opset 13 的第二输入形式后，分段恢复正确，完整模型首次推理通过。这不是所有非等分 Split 都不支持。
2. **FP16 崩溃：已实证 EP 的 `ime2::spe_direct_conv2d` 越过 TCM 张量/映射边界读取。** 本次从自然崩溃捕获到入口参数、故障地址和 TCM 映射；使用保护页后，原本稳定的厂商 RVV OpenCV 配置也在相同指令、相同相对偏移崩溃。
3. **换 RVV 库、限制 OpenBLAS 线程、调整初始化顺序都是当前规避措施。** 它们改变依赖、线程与地址空间布局，使非法读取不一定触发段错误；没有修复 EP 的越界访问。不能把“不崩溃”当作内存访问正确的证明。
4. **单独替换 OpenBLAS 还存在独立的动态链接兼容问题。** 历史 60 份相关日志是 `undefined symbol: gotoblas`，根本没有进入推理，不能计入卷积崩溃的有效 A/B 对照。

内部源码不可见，因此尚不能指认厂商具体哪一行代码出错；但卷积问题已经从“可能的线程池冲突”缩小到**TCM 打包缓冲区容量/边界与内核读取范围不一致**。厂商应检查尾块、padding/halo、调度尺寸与暂存空间规划，而不应只排查 OpenBLAS 线程竞争。

## 2. 环境与验证方式

| 项目 | 实测值 |
|---|---|
| 系统 | Bianbu 4.0.8，riscv64，Linux 6.18.3-generic |
| SpaceMIT ORT / Spine Runtime | 2.0.7+2 / 0.6.3 |
| TCM 包 | 3.0.1 |
| 系统 OpenCV | 4.10.0+dfsg-7ubuntu5 |
| 厂商 OpenCV | 5.0.0-1bb1 |
| 系统 / 厂商 OpenBLAS | 0.3.32+ds-5 / 0.3.32-1bb2 |
| EP MD5 | `9f7d1dde2941e2a4b584b163a50189e0` |
| libspert MD5 | `9167f802d1c7dbc891782cb4b6f214d5` |

设备上现有 `build-k3` 和 `build-k3-rvv` 已经都链接厂商 OpenCV，不能按目录名称把它们当作系统/厂商对照。本次重新用系统 OpenCV 头文件编译 `bench_pipeline.cpp`，复用其他现有对象和静态库，在临时目录生成 `bench-system`。

六组冷启动对照直接运行二进制，清除继承的 `LD_*`、`OPENBLAS_*`、`GOTO_*`、`OMP_*`、`SPACEMIT_EP_*`、`OPENCV_*` 环境变量，再明确设置本组参数。都使用同一个 FP16 模型、`--video-first`、EP 8 线程、宿主绑核 0-7、串行处理 1 帧。没有经由会默认预加载库的 `run_k3_video.sh`。

证据：[环境与校验](evidence/k3-2026-11-12/environment-facts.json)、[矩阵脚本](evidence/k3-2026-11-12/matrix.py)、[逐次结果](evidence/k3-2026-11-12/matrix.json)。

## 3. Split 的具体错误

原始 `data/best.onnx` 是 opset 12，符合本机 ONNX checker；标准形状推导给出：

```text
输入 [1,4,128,400]
Split_130(axis=2, split=[32,32,64])
正确输出：[1,4,32,400]、[1,4,32,400]、[1,4,64,400]
```

本次 EP 导出的 opset 12 完整模型子图仍保留正确的 `split` 属性，但输出 shape 被标成：

```text
[1,4,43,400]、[1,4,43,400]、[1,4,42,400]
```

最后一支的 67,200 个元素不能 reshape 为 `[1,256,20,20]` 的 102,400 个元素。因此 `Reshape_139` 是报错位置，错误来源在上游 Split 形状处理。这与把 128 近似均分为三份的结果一致，说明旧属性没有在该形状处理路径中正确生效；属性实际上仍在导出的节点上，不能简单说“文件里丢了属性”。

本次用 ONNX 版本转换器将同一 FP32 模型转为 opset 13，未改权重：

```text
Split(images, split_sizes=[32,32,64], axis=2)
```

新子图输出为正确的 `[32,32,64]`，完整模型加载及首次推理成功。本地 CPU ORT 对两个版本使用同一个固定种子随机输入，输出 `[1,17,8400]` 逐元素相同，最大绝对差为 0。该检查只有一个输入样本，不替代完整精度回归。

现有 `best.spacemit.fp16.onnx` 实际是 opset 24，已重新含有 `Split_130_slice0_split` 和 `Split_336_slice0_split`，分段以第二输入提供。这也说明“Slices 永远必须保留”或“EP 不支持任何非等分 Split”都过于宽泛。

证据：[两份 EP 子图节点/shape 摘录](evidence/k3-2026-11-12/subgraph-facts.json)、[opset 12 失败](evidence/k3-2026-11-12/full-opset12.log)、[opset 13 成功](evidence/k3-2026-11-12/full-opset13.log)、[单样本数值对照](evidence/k3-2026-11-12/opset-numerical-check.json)。

另做了四个仅含 Split/Slice/Reshape 的小图，均通过，但没有产生 EP 子图，不能用这些小图证明 EP 的旧版路径正确；定性依据是上述完整模型及其 EP 子图。

## 4. FP16 卷积：TCM 边界证据

### 4.1 自然崩溃

系统 OpenCV + 系统 OpenBLAS 8 线程重新复现，GDB 定位到：

```text
libspacemit_ep.so.2.0.7
ime2::spe_direct_conv2d<... spe_kernel_block_m16n32k8 ...>
函数起点：0x3fb5fc
故障指令：0x3fb78e，即函数 +402 字节（+0x192）
vle16.v v3, (a3)
```

以上为库内虚拟地址/符号偏移，运行时绝对地址随 ASLR 改变。完整调用链与历史记录一致，仍在首次 `Ort::Session::Run`，尚未进入业务 Decode/NMS/Tracker。

`entry-1.txt` 在函数第一条指令处断下，记录的关键值为：

```text
TCM 映射：[0x3f73b98000, 0x3f73bf8000)，大小 0x60000 = 384 KiB
第二张量基址：0x3f73b98000
第一张量基址：0x3f73be0000 = TCM 起点 + 0x48000
第一张量 strides：[98304,32768,16384,256]
第二张量 strides：[294912,4608,1536,512]
故障 a3：0x3f73bf8000 = 第一张量起点 + 0x18000 = TCM 末端
```

这与 **288 KiB 权重暂存 + 96 KiB 打包输入暂存，恰好占满 384 KiB TCM** 的布局吻合。故障访问从边界外第一个字节开始，不是单纯“向量尾部跨页”。寄存器显示正在按 `0x4000`/`0x8000` 的步长读取后续行/块。

这是可观测的越界读取。究竟是内核多读了一块、打包缓冲缺少必要 padding、还是调度器给了不匹配的形状/stride，需要厂商源码进一步区分。不能仅凭这些寄存器声称某一个内部变量已经找到。

另一个细节：自然故障快照中，TCM 后面还出现匿名 `rw-p` VMA，但内核仍返回 `SEGV_MAPERR`。因此严谨表述是“越过 TCM 映射并触发非法访存”；不能只凭错误码断言整个进程地址空间中那里完全没有 VMA。设备映射/页表层面的差异仍需厂商核实。

证据：[自然故障寄存器及完整映射](evidence/k3-2026-11-12/gdb-1.txt)、[函数入口参数与故障映射](evidence/k3-2026-11-12/entry-1.txt)、[内核反汇编](evidence/k3-2026-11-12/conv-prologue.asm)。

### 4.2 保护页实验：规避配置中越界仍然存在

临时 `LD_PRELOAD` 探针只拦截 `/dev/tcm` 的 0x60000 字节映射，在两侧预留 `PROT_NONE` 保护页。有效 TCM 大小、设备 fd/offset、映射权限及模型均不变。探针只用于诊断，不是修复，不进入部署配置。

| 配置 | OpenBLAS 线程环境 | 结果 |
|---|---:|---:|
| 厂商 RVV OpenCV，无额外 OpenBLAS 预加载 | 1 | 3/3 SIGSEGV |
| 系统 OpenCV | 1 | 3/3 SIGSEGV |

厂商配置的 GDB 也停在同一条 `vle16.v v3,(a3)`，而不是探针内部：

```text
TCM：[0x3f733a4000, 0x3f73404000)
第一张量：0x3f733ec000
故障地址：0x3f73404000 = 第一张量 + 0x18000 = 保护页起点
si_code = 2 (SEGV_ACCERR)
```

由此可以排除“换库已经让这次越界访问消失”的解释。无保护页时，相邻 TCM 或其他可访问区域可能承接这次非法读取，使程序看起来正常；保护页把它显式暴露出来。是否污染最终输出尚未验证，不能据此声称模型输出已经错误，也不能承诺输出一定正确。

证据：[保护页源码](evidence/k3-2026-11-12/tcm_guard.c)、[6 次结果](evidence/k3-2026-11-12/guard-results.json)、[厂商配置 GDB](evidence/k3-2026-11-12/guard-vendor-gdb.txt)、[只记录映射的厂商成功运行](evidence/k3-2026-11-12/maptrace-vendor.txt)。

## 5. RVV OpenCV / OpenBLAS 为什么能规避

### 5.1 “RVV 版”同时换了很多东西

| 项目 | 系统 OpenCV 4.10 | 厂商 OpenCV 5.0 |
|---|---|---|
| CPU baseline | 未标注 RVV | RVV，启用 RVV HAL |
| LAPACK | 外部 `liblapack.so.3`、`libblas.so.3` | 内置 libclapack |
| OpenCV 并行框架 | TBB | pthreads |
| GStreamer | 开启 | 关闭 |
| `opencv_core` 直接加载外部 BLAS/TBB | 是 | 否 |

本次厂商 `bench_pipeline` 的 `ldd` 在不预加载时**没有任何 OpenBLAS**。EP 本身也没有 OpenBLAS 直接依赖，故障栈运行的是 EP 的 IME2 卷积内核。把 OpenBLAS 换成另一版，不是在替换这条出错的卷积实现。

系统 OpenBLAS 查询结果已经是：

```text
OpenBLAS 0.3.32 NO_LAPACKE DYNAMIC_ARCH NO_AFFINITY riscv64_zvl256b MAX_THREADS=128
corename=riscv64_zvl256b, parallel=1, num_threads=8
```

厂商版则为：

```text
OpenBLAS 0.3.32 NO_AFFINITY x100 MAX_THREADS=8
corename=x100, parallel=1, num_threads=8
```

两者加载后都可创建 8 个任务，不能把厂商版误认为天然单线程，也不能把系统版称为完全没有向量优化。决定本次是否段错误的是具体运行环境和边界访问是否被暴露，不是“有无 RVV”这个标签。

证据：[系统构建信息](evidence/k3-2026-11-12/system-opencv-build.txt)、[厂商构建信息](evidence/k3-2026-11-12/vendor-opencv-build.txt)、[无 OpenBLAS 的依赖列表](evidence/k3-2026-11-12/vendor_no_blas-ldd.txt)、[ELF 依赖与符号](evidence/k3-2026-11-12/environment-facts.json)。

### 5.2 本次六组对照

| OpenCV | OpenBLAS 加载方式 | 线程数 | 成功 / 冷启动 |
|---|---|---:|---:|
| 系统 | 系统依赖自动加载 | 8 | 4/10，6 次 SIGSEGV |
| 系统 | 系统依赖自动加载 | 1 | 10/10 |
| 厂商 | 不加载 OpenBLAS | 不适用 | 10/10 |
| 厂商 | 预加载系统 OpenBLAS | 8 | 10/10 |
| 厂商 | 预加载厂商 OpenBLAS | 8 | 10/10 |
| 厂商 | 预加载厂商 OpenBLAS | 1 | 10/10 |

因此：

- OpenBLAS 线程数是系统 OpenCV 组合下已复现的触发因素，但“存在 OpenBLAS 多线程”不是充分条件。
- 在本次样本中，厂商 OpenCV 单独就足以规避段错误，不要求同时换厂商 OpenBLAS。
- 更换 OpenCV 同时改变依赖、线程后端、启动分配及地址空间布局；当前证据不足以把规避效果唯一归因于 TBB、GStreamer 或某个 RVV 函数。
- 结合保护页证据，最有根据的机制是这些外围变化掩盖了 EP 已存在的边界读取。有限次成功只描述本次试验，不能保证长期无故障。

### 5.3 只换 OpenBLAS 的历史试验实际是 ABI 失败

系统依赖链为：

```text
libopencv_core.so.410
  -> libblas.so.3 / liblapack.so.3（openblas-pthread 提供）
  -> libopenblas.so.0
```

系统 `libblas.so.3` 有未定义的内部符号 `gotoblas`、`gotoblas_corename`，由系统 DYNAMIC_ARCH OpenBLAS 提供；厂商 x100 固定目标库不导出这些符号。仅把 `libopenblas.so.0` 指向厂商版，会产生：

```text
symbol lookup error: /usr/lib/riscv64-linux-gnu/libblas.so.3: undefined symbol: gotoblas
```

本次核查 `openblas-rvv-*.log` 30 份、`recheck-rvv-*.log` 30 份，全部是此错误。这是混用系统 BLAS/LAPACK 拆分库与厂商 OpenBLAS 的链接兼容问题，和 `direct_conv2d` 的 SIGSEGV 是两回事。厂商 OpenCV 使用内置 libclapack，避开了这条外部拆分库依赖链。

## 6. 处理建议与厂商修复验收

当前部署可继续保留“先 EP 首次推理、后打开视频”及 `OPENBLAS_NUM_THREADS=1`，但应标注为规避措施。不要为了验证替代方案修改全局 BLAS alternatives 或把 OpenCV 4/5 的不同 SONAME 强行做软链接。本次未变更部署库。

厂商需要修复/核查：

1. opset 12 Split 属性形式进入 EP 后的 shape inference，确保 `[32,32,64]` 不被处理成 `[43,43,42]`。
2. 库内 `0x3fb78e` / `spe_direct_conv2d+402` 的 TCM 读取边界，尤其是本次 288 KiB + 96 KiB 恰好填满 TCM 的调用；检查 M=20、N=32、K=512 对应的分块/尾块及输入 padding。
3. TCM 规划容量、内核实际最大访问偏移、不同 worker 的映射独立性。即使相邻映射碰巧连续，也不能据此允许读过当前缓冲区末端。
4. 对需要读取 padding 的实现，保证 padding 属于已合法分配且按算子语义初始化的区域；不能只延长映射掩盖异常。

修复验收至少应包含原始故障配置、厂商配置，以及保护页配置；同时比对输出数值。只重复 100 次“不崩溃”不足以验收这个已经实证的越界问题。

保护页复现源码及示例命令见 [证据目录说明](evidence/k3-2026-11-12/README.md)。
