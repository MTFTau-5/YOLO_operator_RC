#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "$(uname -m)" != "riscv64" ]]; then
    printf '错误: 请在 K3/RISC-V 环境执行此脚本。\n' >&2
    exit 1
fi

SUDO=""
if ((EUID != 0)); then
    if ! command -v sudo >/dev/null 2>&1; then
        printf '错误: 需要 root 或 sudo 安装系统依赖。\n' >&2
        exit 1
    fi
    SUDO=sudo
fi

export DEBIAN_FRONTEND=noninteractive
$SUDO apt-get update
$SUDO apt-get install -y \
    build-essential \
    ca-certificates \
    cmake \
    git \
    pkg-config \
    spacemit-onnxruntime \
    openblas-spacemit \
    opencv-spacemit

missing=0
for path in \
    /opt/opencv-spacemit/lib/cmake/opencv5/OpenCVConfig.cmake \
    /opt/openblas-spacemit/lib/libopenblas.so.0 \
    /usr/lib/libonnxruntime.so \
    /usr/lib/libspacemit_ep.so; do
    if [[ ! -e "$path" ]]; then
        printf '缺少: %s\n' "$path" >&2
        missing=1
    fi
done

if ((missing)); then
    printf '依赖安装后仍有文件缺失，请检查 Bianbu 软件源和包版本。\n' >&2
    exit 1
fi

printf 'K3 构建依赖已就绪：CMake、编译器、SpaceMIT ORT、OpenCV RVV、OpenBLAS RVV。\n'
printf '下一步: ./build_k3.sh 或 ./run_k3_video.sh\n'
