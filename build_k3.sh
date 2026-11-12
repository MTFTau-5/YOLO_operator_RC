#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-k3}"
JOBS="${JOBS:-$(nproc)}"
A100_MODE="${YOLOOP_RISCV_A100:-ON}"
RUN_TESTS=1

usage() {
    cat <<'EOF'
用法:
  ./build_k3.sh              编译并运行 golden/应用测试
  ./build_k3.sh --no-test    只编译
  JOBS=16 ./build_k3.sh      指定编译并行度

可用环境变量:
  BUILD_DIR              构建目录，默认 <项目>/build-k3
  JOBS                   并行度，默认 nproc
  OMP_NUM_THREADS        测试线程数，默认 8
  YOLOOP_RISCV_ARCH      RISC-V ISA，默认 rv64gcv
  YOLOOP_RISCV_ABI       RISC-V ABI，默认 lp64d
  YOLOOP_RISCV_A100      后处理迁到 A100，默认 ON；OFF 使用宿主 RVV
  SPACEMIT_ORT_ROOT      可选的 SpaceMIT ONNX Runtime SDK 根目录
  OpenCV_DIR             可选的 OpenCV CMake 配置目录，默认优先使用 /opt/opencv-spacemit
EOF
}

while (($#)); do
    case "$1" in
        --no-test) RUN_TESTS=0 ;;
        -h|--help) usage; exit 0 ;;
        *) printf '未知参数: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

if [[ "$(uname -m)" != "riscv64" ]]; then
    printf '错误: 该脚本用于 K3 本机原生编译，当前架构为 %s。\n' "$(uname -m)" >&2
    printf '交叉编译请使用 RISC-V 工具链文件，不要直接运行此脚本。\n' >&2
    exit 1
fi

for cmd in cmake git "${CXX:-c++}"; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        printf '错误: 未找到命令 %s，请先安装编译工具和依赖。\n' "$cmd" >&2
        exit 1
    fi
done
if [[ ! -d "${ROOT_DIR}/test_data" ]]; then
    printf '错误: 缺少 %s/test_data，请完整迁移仓库。\n' "$ROOT_DIR" >&2
    exit 1
fi

if [[ -z "${OpenCV_DIR:-}" && -f /opt/opencv-spacemit/lib/cmake/opencv5/OpenCVConfig.cmake ]]; then
    OpenCV_DIR=/opt/opencv-spacemit/lib/cmake/opencv5
fi
OPENCV_CMAKE_ARGS=()
if [[ -n "${OpenCV_DIR:-}" ]]; then
    OPENCV_CMAKE_ARGS=(-DOpenCV_DIR="$OpenCV_DIR")
fi

printf 'K3 构建目录: %s\n' "$BUILD_DIR"
printf '编译并行度: %s\n' "$JOBS"
printf 'RISC-V 目标: %s / %s\n' \
    "${YOLOOP_RISCV_ARCH:-rv64gcv}" "${YOLOOP_RISCV_ABI:-lp64d}"
printf 'SpaceMIT ORT: %s\n' "${SPACEMIT_ORT_ROOT:-自动探测系统安装}"
printf 'OpenCV: %s\n' "${OpenCV_DIR:-系统默认}"

cmake -S "$ROOT_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    "${OPENCV_CMAKE_ARGS[@]}" \
    -DYOLOOP_BUILD_RISCV=ON \
    -DYOLOOP_RISCV_A100="$A100_MODE" \
    -DYOLOOP_BUILD_X86=OFF \
    -DYOLOOP_BUILD_CUDA=OFF \
    -DYOLOOP_BUILD_TRT_PLUGIN=OFF \
    -DYOLOOP_BUILD_PYTHON=OFF \
    -DYOLOOP_BUILD_VULKAN=OFF \
    -DYOLOOP_NCNN_VULKAN=OFF \
    -DYOLOOP_BUILD_SPACEMIT_ORT=ON \
    -DSPACEMIT_ORT_ROOT="${SPACEMIT_ORT_ROOT:-}" \
    -DYOLOOP_RISCV_ARCH="${YOLOOP_RISCV_ARCH:-rv64gcv}" \
    -DYOLOOP_RISCV_ABI="${YOLOOP_RISCV_ABI:-lp64d}"
cmake --build "$BUILD_DIR" -j"$JOBS"

if ((RUN_TESTS)); then
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
    if [[ "${A100_MODE^^}" == ON ]]; then "$BUILD_DIR/riscv/test_a100"; fi
    "$BUILD_DIR/riscv/test_rvv" "$ROOT_DIR/test_data"
    "$BUILD_DIR/app/test_tracker" "$ROOT_DIR/test_data"
    "$BUILD_DIR/app/test_queue"
    "$BUILD_DIR/app/test_head_layouts"
fi

printf '\n完成。应用: %s/app/yolo_rt\n' "$BUILD_DIR"
printf 'ncnn 配置: %s/app/pipeline_k3.yaml\n' "$ROOT_DIR"
printf 'SpaceMIT ORT 配置: %s/app/pipeline_k3_ort.yaml\n' "$ROOT_DIR"
printf '视频测试: BACKEND=spacemit ./run_k3_video.sh（BACKEND=ncnn 可跑原基线）\n'
