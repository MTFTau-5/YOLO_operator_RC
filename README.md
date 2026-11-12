# YOLO_operator

YOLO 部署侧算子与检测跟踪应用：预处理、Decode、NMS、ncnn / SpaceMIT ORT 推理、Tracker 和 UDP 输出。

| 目录 | 功能 |
|---|---|
| `common/` | 公共类型、标量参考、统一 CPU 接口 |
| `riscv/` | RVV 算子、K3 A100 后处理调度、标量回退 |
| `x86/` | AVX2 算子 |
| `nvidia/` | CUDA 算子、Python 绑定、TensorRT Decode 插件 |
| `vulkan/` | IoU 矩阵 + CPU 贪心 NMS |
| `app/` | 两线程三槽位流水线、推理适配、跟踪与控制输出 |
| `tools/` | 模型导出、golden 生成、数值验收、K3 故障复现 |

## K3 构建与运行

保持本目录与 `data/` 同级。在 K3 Bianbu 上执行：

```bash
./install_k3_dependencies.sh
./build_k3.sh                         # 编译并运行已有 golden/应用测试
./run_k3_video.sh                     # SpaceMIT EP + FP16 ONNX
BACKEND=ncnn ./run_k3_video.sh        # ncnn CPU 对照
./build-k3/app/yolo_rt --config app/pipeline_k3_ort.yaml
./build-k3/app/yolo_rt --config app/pipeline_k3.yaml
```

脚本优先使用 `/opt/opencv-spacemit` 和 `/opt/openblas-spacemit`，默认限制 OpenBLAS 为单线程，先完成 EP 首次推理再打开视频。EP 线程数由 `THREADS` 控制，视频帧数由 `FRAMES` 控制；`BUILD_DIR`、`JOBS` 等编译参数见 `./build_k3.sh --help`。

**现有 EP 仍存在已复现的 TCM 越界读取。** 换库和启动顺序是规避措施，不能作为根因修复。详见 [K3 调试摘要](docs/K3_DEBUG_SUMMARY.md) 和 [复现报告](docs/K3_ROOT_CAUSE_ANALYSIS_2026-11-12.md)。

`build_k3.sh` 默认将全部 Decode/NMS 迁到 1 个 A100 核，EP 最多使用其余 7 核。计算采用 RVV/标量，预处理与 Tracker 保留宿主执行。详见 [A100 后处理](docs/IME_PORTING.md)。设置 `YOLOOP_RISCV_A100=OFF` 可回到宿主 RVV。

## x86 / CUDA / Vulkan 构建

Python 依赖用于导出和测试，C++ K3 部署使用上述系统安装脚本：

```bash
python -m pip install -r requirements.txt
# NVIDIA Python / TensorRT 测试另装：
python -m pip install -r requirements-cuda.txt
cmake -S . -B build
cmake --build build -j8
```

需要指定 CUDA 编译器时加 `-DCMAKE_CUDA_COMPILER=/path/to/nvcc`。使用绑定的 Python 环境应与构建环境一致；可指定 `-DPython_EXECUTABLE=/path/to/python`。CMake 自动探测 OpenMP、CUDA、Vulkan、TensorRT 和 SpaceMIT SDK；缺少依赖时对应可选模块跳过。

| CMake 参数 | 默认 / 用途 |
|---|---|
| `YOLOOP_BUILD_X86` / `YOLOOP_BUILD_RISCV` | 按目标架构选择 CPU 后端 |
| `YOLOOP_BUILD_CUDA` / `YOLOOP_BUILD_TRT_PLUGIN` / `YOLOOP_BUILD_PYTHON` | ON；CUDA、TRT 插件、Python 绑定 |
| `YOLOOP_BUILD_VULKAN` / `YOLOOP_NCNN_VULKAN` | ON；混合 NMS / ncnn Vulkan |
| `YOLOOP_RISCV_A100` | 通用 CMake 默认 OFF；K3 脚本默认 ON，需 Spine Runtime |
| `YOLOOP_BUILD_APP` | ON；需 ncnn，配置时 FetchContent 拉取 |
| `YOLOOP_BUILD_SPACEMIT_ORT` / `SPACEMIT_ORT_ROOT` | ON / 空；自动探测或指定 SDK |
| `YOLOOP_X86_ARCH` | `native`；7840HS 可设 `znver4` |
| `YOLOOP_RISCV_ARCH` / `YOLOOP_RISCV_ABI` | `rv64gcv` / `lp64d` |
| `NCNN_GIT_TAG` | `20250503`；可指定 ncnn 版本 |

只构建独立 RISC-V 算子（交叉编译时另提供工具链和 sysroot）：

```bash
cmake -S . -B build-k3 -DYOLOOP_BUILD_RISCV=ON -DYOLOOP_BUILD_X86=OFF -DYOLOOP_RISCV_A100=ON \
  -DYOLOOP_BUILD_APP=OFF -DYOLOOP_BUILD_CUDA=OFF -DYOLOOP_BUILD_VULKAN=OFF
cmake --build build-k3 -j8
```

## 模型与接口

支持 v5 anchor、v8/v9/v11 fused、pose、pre-DFL raw、post-DFL dist，以及 v10 E2E。各输出形状、导出和调用示例见 [模型契约](docs/MODEL_CONTRACTS.md)。统一 CPU 接口为 `common/cpu_ops.h`，RVV 接口为 `riscv/yolo_rvv.h`。

```bash
./build/app/yolo_rt --config app/pipeline.example.yaml
./build/app/yolo_rt --config app/pipeline.example.yaml --conf 0.35 --dump-config
```

覆盖顺序：默认值 < YAML < CLI。跟踪器、UDP 包、设备基准及 raw 动态尺寸限制见 [应用说明](app/README.md)；线程和槽位规则见 [流水线](docs/PIPELINE.md)。纯 CPU 两线程运行时应按目标 CPU 调整 `cores-a/cores-b`，避免推理线程受限于少量核。

## 验证

仓库包含 `test_data/`。只重测无需重新生成；修改算法后按需运行：

```bash
./build/x86/test_x86 test_data
./build-k3/riscv/test_a100                    # 仅 A100 构建
./build-k3/riscv/test_rvv test_data
./build/nvidia/test_cuda test_data
PYTHONPATH=build/nvidia python nvidia/tests/test_python_bindings.py
./build/nvidia/test_trt_plugin
./build/app/test_tracker test_data
./build/app/test_queue
./build/app/test_head_layouts
./build/app/yolo_rt --no-model
```

golden 生成入口：`tools/gen_test_vectors.py`、`gen_v8_vectors.py`、`gen_afraw_vectors.py`、`gen_afdist_vectors.py`、`gen_e2e_vectors.py`、`gen_tracker_golden.py`。Tracker golden 默认读取同级 `RK3588-AntiUAV-Tracker/tracker`，也可传入源码目录。

真实模型验收：

```bash
python tools/accept_raw_dynamic.py --weights <detect.pt> --work-dir <dir> --build-dir build
# v11 等包含静态 attention reshape 的模型加 --per-size-export
```

Decode/NMS 数量和标签应一致，坐标误差 < `1e-2`、score < `1e-3`，预处理误差 < `1e-3`。微基准由测试程序打印；端到端性能以目标设备实测为准。

更多：[优化方向](docs/Operator_Suggestion.md) · [K3 SSH 移交](docs/K3_HANDOVER_SSH.md) · [厂商问题单](docs/K3_SPACEMIT_VENDOR_DEBUG_REPORT.md)
