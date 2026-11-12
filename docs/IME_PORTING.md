# K3 A100 后处理

后处理整体通过 Spine 的 `spacemit-k3` 后端在 A100 执行，允许 RVV 和标量配合。当前计算内核保持原有算法，未新增 IME 矩阵乘法指令；A100 执行位置与使用 IME 矩阵指令是两个不同概念。

## 启用

```bash
./install_k3_dependencies.sh
./build_k3.sh                         # 默认 A100 后处理 ON
./run_k3_video.sh
# 回到宿主 RVV：
YOLOOP_RISCV_A100=OFF ./run_k3_video.sh
```

直接 CMake 构建时添加 `-DYOLOOP_BUILD_RISCV=ON -DYOLOOP_RISCV_A100=ON`，依赖 K3 的 `spacemit-runtime` 开发头文件与 `libspert`。通用 RVV CMake 构建默认 OFF；开启后若运行时不可用则报错，不静默回退。

## 执行范围

| 函数 | 执行位置 |
|---|---|
| v5 / fused / pose / raw DFL / dist / E2E Decode | A100 |
| `nms` / `nms_indices` 的排序、IoU 和贪心选择 | A100 |
| letterbox 预处理、Tracker、UDP | 原宿主路径 |

公开接口保持同步：返回前任务完成，输入缓冲区可在返回后复用。`a100_dispatch.h/.cpp` 共用一个持久 stream 并串行化外部并发提交；嵌套 NMS 调用直接在当前 worker 执行，异常在完成同步后回传宿主。A100 内不再启动 OpenMP 宿主线程池。

## 核资源

后处理独占 1 个 A100 核。项目 ORT 适配层将 EP 请求限制为 1～7 核，包括环境变量覆盖；另一个核留给后处理，避免 Spine stream 等待已被 EP 占用的全部核。外部自行创建的 EP/Spine stream 也必须为后处理预留至少 1 核，不能在占满 8 核后调用本后端。

初始化日志显示明确后端与实际 worker VLEN：

```text
[postprocess] backend=spacemit-k3 cores=1 arch=41060 vlen=128 bytes
[postprocess] A100 worker vlen=128 bytes
```

## 验证与性能

2026-11-12 K3 实测：

- 全部现有 golden：`ALL PASS`，包括各类 Decode、关键点、NMS 与预处理。
- `test_a100`：验证实际 VLEN=1024、同步/嵌套/四宿主线程并发提交及异常传播。
- 9 种候选长度（含 0、4095/4096/4097）×2 阈值，v5/raw/dist 的数量、标签、数值和顺序对齐。
- 完整应用 CMake 构建与 Tracker/队列/head 布局测试通过；EP 两线程、ncnn CPU + A100 后处理运行通过。
- FP16 EP 7 核 +后处理 1 核：串行 100 帧成功，Decode+NMS 平均 0.124 ms，整链路约 19.8 FPS。

该迁移不保证提速：本次密集 v5 Decode 微基准约 12.1 ms、raw DFL 约 6.2 ms，较此前宿主多线程路径更慢。主要目的为执行位置迁移；后续优化应包含任务提交开销和 EP 少一核的影响。

```bash
./build-k3/riscv/test_a100
OMP_NUM_THREADS=8 ./build-k3/riscv/test_rvv test_data
```

实际日志见 [验证证据](evidence/README.md)。EP 卷积 TCM 越界问题仍独立存在，见 [根因报告](K3_ROOT_CAUSE_ANALYSIS_2026-11-12.md)。

IME 矩阵计算的指令语义与限制依据 [SpaceMIT 官方文档](https://www.spacemit.com/community/document/info?lang=zh&nodepath=ai/architecture/ime_extension.md)。比较、exp、除法和贪心控制继续由 A100 上的 RVV/标量执行；需要矩阵乘累加的模型计算由 EP 负责。
