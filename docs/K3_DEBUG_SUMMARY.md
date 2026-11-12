# K3 调试摘要

复核日期：2026-11-12。完整参数与证据见 [根因报告](K3_ROOT_CAUSE_ANALYSIS_2026-11-12.md)。

| 问题 | 已确认原因 | 处理状态 |
|---|---|---|
| `Reshape_139` 编译失败 | opset 12 的 Split `[32,32,64]` 在 EP 子图被错推为 `[43,43,42]` | 改为 Slice，或完整模型转 opset 13 第二输入形式可规避 |
| FP16 首次推理 SIGSEGV | `spe_direct_conv2d+402` 的 `vle16.v` 读到 384 KiB TCM 块末端之外 | 已实证越界；具体尾块/padding/容量错误需厂商源码确认 |
| 只替换厂商 OpenBLAS 后启动失败 | 系统 BLAS/LAPACK 依赖厂商固定 x100 库不导出的 `gotoblas` 内部符号 | 独立的动态链接兼容错误，未进入推理 |

当前 FP16 模型是 opset 24，仍有第二输入形式的非等分 Split，故不能认定 EP 不支持所有非等分 Split。

厂商 OpenCV 使用内置 libclapack、pthreads；系统 OpenCV 使用外部 BLAS/LAPACK、TBB。系统 OpenBLAS 也已选择向量内核。换库、OpenBLAS 单线程及 EP 优先初始化改变触发条件；**保护页下两套 OpenCV 各 3/3 崩溃，说明越界仍存在**。

部署继续保留厂商库、`OPENBLAS_NUM_THREADS=1` 和“EP 首次推理 → 打开视频”，标注为规避措施。自动重试不能当作修复；输出是否受到越界影响尚未验证。

`OrtInfer` 是业务适配层，首次普通 `session->Run` 由官方 ORT/EP 执行；崩溃发生在 Decode/NMS/Tracker 之前。

厂商应修复旧 Split 形状处理和 TCM 读取边界，并同时通过保护页与输出数值验收。[单文件复现程序](../tools/reproduce_k3_tcm.cpp) 和 [厂商问题单](K3_SPACEMIT_VENDOR_DEBUG_REPORT.md) 可用于移交。
