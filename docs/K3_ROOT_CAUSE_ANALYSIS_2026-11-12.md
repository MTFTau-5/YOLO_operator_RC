# K3 根因复核

日期：2026-11-12。设备上使用临时目录 `/tmp/k3-rootcause-20261112`，未修改部署库。原始日志、探针、EP 子图及六组实验脚本保存在 [证据压缩包](evidence/k3-2026-11-12.tar.gz)，下文文件名指压缩包内部路径。

## 环境

| 项目 | 版本 |
|---|---|
| 系统 | Bianbu 4.0.8，riscv64，Linux 6.18.3-generic |
| SpaceMIT ORT / Spine / TCM | 2.0.7+2 / 0.6.3 / 3.0.1 |
| 系统 / 厂商 OpenCV | 4.10.0+dfsg-7ubuntu5 / 5.0.0-1bb1 |
| 系统 / 厂商 OpenBLAS | 0.3.32+ds-5 / 0.3.32-1bb2 |
| EP MD5 | `9f7d1dde2941e2a4b584b163a50189e0` |
| libspert MD5 | `9167f802d1c7dbc891782cb4b6f214d5` |

## Split：旧版属性处理

原始 opset 12 模型通过 ONNX checker。`Split_130` 输入 `[1,4,128,400]`，属性 `split=[32,32,64]`；EP 子图保留该属性，却标注三路形状为 `[1,4,43,400]`、`[1,4,43,400]`、`[1,4,42,400]`。最后一支 67,200 个元素不能 reshape 为 `[1,256,20,20]` 所需的 102,400 个元素，因此下游 `Reshape_139` 报错。

同权重完整 FP32 模型通过 ONNX 版本转换器转到 opset 13，分段改为第二输入后，EP 子图三路为正确的 `[32,32,64]`，首次推理成功。本地 CPU ORT 单个固定随机输入的输出逐元素一致，最大绝对差为 0；这不替代完整精度回归。

历史 Slice 改图也可规避。当前 FP16 模型为 opset 24，转换后又含第二输入形式的非等分 Split，不能泛化为“所有非等分 Split 都不支持”。小型 Split 测试图虽通过，但没有生成 EP 子图，不能替代完整模型对照。

证据：`full-opset12.log`、`full-opset13.log`、两份 `SpaceMIT*.onnx` 图快照。快照不含完整权重，不能单独运行。

## FP16：TCM 输入暂存区越界

自然故障调用链：首次 `Ort::Session::Run → SpineGraphExecutor → ComputeDirectConv2DBlockThreaded → ime2::spe_direct_conv2d`。尚未执行项目 Decode/NMS/Tracker。

```text
内核：spe_direct_conv2d<...spe_kernel_block_m16n32k8...>
库内指令地址：0x3fb78e = 函数起点 +402 字节
指令：vle16.v v3,(a3)
TCM：[0x3f73b98000,0x3f73bf8000)，0x60000 = 384 KiB
第二张量基址：0x3f73b98000
第一张量基址：0x3f73be0000 = TCM 起点 +288 KiB
第一张量 strides：[98304,32768,16384,256]
第二张量 strides：[294912,4608,1536,512]
故障地址：0x3f73bf8000 = 第一张量起点 +96 KiB = TCM 末端
```

布局与 288 KiB 权重 +96 KiB 打包输入恰好填满 TCM 吻合。故障从边界外第一个字节开始，已确认越界读取。内核尾块多读、输入 padding/halo、stride 或调度与容量规划不一致仍需源码区分；不能据此指认具体内部变量。绝对地址随 ASLR 改变，比较应使用映射边界和相对偏移。

自然故障为 `SEGV_MAPERR`，但 TCM 后方快照还存在匿名 `rw-p` VMA，不能只凭该错误码断言整个地址空间那里没有 VMA。设备映射/页表行为仍需厂商确认。历史 `cause=0xd` 表示 Load page fault。

证据：`gdb-1.txt`、`entry-1.txt`、`conv-prologue.asm`。

### 保护页验证

临时 mmap 探针对 `/dev/tcm` 每个 0x60000 字节块两侧增加 `PROT_NONE` 页，保留有效映射大小、fd、offset 和权限。厂商与系统 OpenCV、OpenBLAS 均设单线程，各 3/3 SIGSEGV。厂商配置也在相同指令、相同输入相对偏移失败：

```text
TCM：[0x3f733a4000,0x3f73404000)
输入起点：0x3f733ec000
故障地址：0x3f73404000 = 输入起点 +96 KiB
si_code=SEGV_ACCERR
```

这证明此前规避配置中的越界未消失。无保护页时，相邻 TCM/其他区域可能承接非法读取，使程序不崩溃。尚未确认是否影响最终输出，不可保证数值安全，也不可声称已发生输出污染。

证据：`tcm_guard.c`、`guard-results.json`、`guard-vendor-gdb.txt`、`maptrace-vendor.txt`。当前仓库另提供不依赖 OpenCV/视频的 [单文件复现](../tools/reproduce_k3_tcm.cpp)。

## RVV 库为何规避

| 构建差异 | 系统 OpenCV 4.10 | 厂商 OpenCV 5.0 |
|---|---|---|
| LAPACK | 外部 BLAS/LAPACK | 内置 libclapack |
| 并行框架 | TBB | pthreads |
| GStreamer | 开启 | 关闭 |
| RVV HAL | 无 | 开启 |

厂商程序不预加载时，`ldd` 中没有 OpenBLAS；EP 的出错卷积也不是通过外部 OpenBLAS 实现。系统 OpenBLAS 实际选择 `riscv64_zvl256b`，厂商版为 `x100`，两者均可创建 8 线程。不能把根因解释成“系统库无 RVV”或“厂商库天然单线程”。

设备现有两个 build 目录都已链接厂商 OpenCV，本次另编译系统 OpenCV 对照。均直接运行二进制，清除相关继承环境变量，使用同一 FP16 模型、旧顺序 `--video-first`、EP 8 线程、绑核 0-7、串行 1 帧：

| OpenCV / OpenBLAS | 成功 / 冷启动 |
|---|---:|
| 系统 / 系统 8 线程 | 4/10（6 次 SIGSEGV） |
| 系统 / 系统 1 线程 | 10/10 |
| 厂商 / 无 OpenBLAS | 10/10 |
| 厂商 / 预加载系统 8 线程 | 10/10 |
| 厂商 / 预加载厂商 8 线程 | 10/10 |
| 厂商 / 预加载厂商 1 线程 | 10/10 |

所以 OpenBLAS 多线程是一个触发因素，不是充分条件。换 OpenCV 同时改变多个依赖、启动分配和地址布局，无法唯一归因于 TBB、GStreamer 或某个 RVV 函数。结合保护页结果，外围变化掩盖/暴露 EP 边界读取是有证据的解释；有限次成功不保证长期安全。

另核查历史 `openblas-rvv-*.log` 与 `recheck-rvv-*.log` 各 30 份，全部为 `libblas.so.3: undefined symbol: gotoblas`。系统 BLAS/LAPACK 拆分库依赖系统 DYNAMIC_ARCH OpenBLAS 的内部符号，厂商固定 x100 库不导出。它们未进入推理，不能用作卷积故障 A/B 试验。厂商 OpenCV 的内置 libclapack 避开这条依赖链。

证据：`environment-facts.json`、`matrix.py/json`、`*-opencv-build.txt`、`*-ldd.txt`。

## 处理与验收

保留 EP 优先初始化、厂商 OpenCV 和 `OPENBLAS_NUM_THREADS=1`，注明是规避。不要将重试成功或冷启动不崩溃称为修复。

厂商应修复旧 Split 属性形状处理，并核查 `spe_direct_conv2d+402` 的 M=20/N=32/K=512 调用、TCM 暂存容量、尾块/padding 和最大读取偏移。合法 padding 应在有效分配范围内且按语义初始化，不能仅延长映射掩盖异常。修复验收需原故障组合、厂商组合、保护页和输出数值对照。
