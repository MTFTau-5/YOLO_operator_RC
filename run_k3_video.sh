#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${ROOT_DIR}/../data"
BUILD_DIR="${ROOT_DIR}/build-k3"
VIDEO="${DATA_DIR}/Screencast from 2026-10-13 20-00-08.webm"
BACKEND="${BACKEND:-spacemit}"

if [[ "$BACKEND" == "spacemit" || "$BACKEND" == "ort" ]]; then
    BACKEND=spacemit
    PARAM=""
    MODEL="${DATA_DIR}/best.spacemit.fp16.onnx"
    INPUT=images
    OUTPUT=output0
elif [[ "$BACKEND" == "ncnn" ]]; then
    PARAM="${DATA_DIR}/best.ncnn.param"
    MODEL="${DATA_DIR}/best.ncnn.bin"
    INPUT=in0
    OUTPUT=out0
else
    printf '错误: BACKEND 仅支持 spacemit 或 ncnn，当前为 %s。\n' "$BACKEND" >&2
    exit 2
fi

if [[ -n "$PARAM" && ! -f "$PARAM" ]]; then
    printf '错误: 缺少 %s。\n' "$PARAM" >&2
    exit 1
fi
for path in "$MODEL" "$VIDEO"; do
    if [[ ! -f "$path" ]]; then
        printf '错误: 缺少 %s，请保持解压后的 YOLO_operator_RC 与 data 目录同级。\n' "$path" >&2
        exit 1
    fi
done

if [[ ! -d /opt/opencv-spacemit/lib || ! -f /opt/openblas-spacemit/lib/libopenblas.so.0 ]]; then
    printf '错误: 默认 RVV 配置需要 openblas-spacemit 和 /opt/opencv-spacemit。\n' >&2
    exit 1
fi

"$ROOT_DIR/build_k3.sh" --no-test
if [[ ! -x "${BUILD_DIR}/app/bench_pipeline" ]]; then
    printf '错误: 未生成 bench_pipeline。请安装 OpenCV 开发包后重新运行。\n' >&2
    printf 'Bianbu/Ubuntu 可安装: libopencv-dev\n' >&2
    exit 1
fi

export LD_LIBRARY_PATH="/opt/opencv-spacemit/lib:/opt/openblas-spacemit/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export LD_PRELOAD="/opt/openblas-spacemit/lib/libopenblas.so.0${LD_PRELOAD:+:${LD_PRELOAD}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
exec taskset -c 0-7 "$BUILD_DIR/app/bench_pipeline" \
    --backend "$BACKEND" \
    --param "$PARAM" \
    --model "$MODEL" \
    --in0 "$INPUT" \
    --model-type v8-pose \
    --heads "$OUTPUT" \
    --num-classes 1 \
    --num-kpts 4 \
    --video "$VIDEO" \
    --frames "${FRAMES:-500}" \
    --serial \
    --gpu -1 \
    --threads "${THREADS:-8}" \
    --cores-a 0-7
