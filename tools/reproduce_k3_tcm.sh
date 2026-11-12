#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_PIPELINE="${BENCH_PIPELINE:-${ROOT_DIR}/build-k3/app/bench_pipeline}"
MODEL="${MODEL:-${ROOT_DIR}/../data/best.spacemit.fp16.onnx}"
VIDEO="${VIDEO:-${ROOT_DIR}/../data/Screencast from 2026-10-13 20-00-08.webm}"
GUARD_SRC="${GUARD_SRC:-${ROOT_DIR}/docs/evidence/k3-2026-11-12/tcm_guard.c}"
TRACE_SCRIPT="${TRACE_SCRIPT:-${ROOT_DIR}/docs/evidence/k3-2026-11-12/trace-entry.py}"
RUNS="${RUNS:-3}"
BLAS_THREADS="${BLAS_THREADS:-1}"
EP_THREADS="${EP_THREADS:-8}"
CORES="${CORES:-0-7}"
BASELINE="${BASELINE:-1}"
GDB="${GDB:-0}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
用法:
  ./tools/reproduce_k3_tcm.sh
  RUNS=1 GDB=1 ./tools/reproduce_k3_tcm.sh
  BENCH_PIPELINE=/tmp/bench-system ./tools/reproduce_k3_tcm.sh

默认执行 3 次无保护页基线和 3 次 TCM 保护页测试，保护页测试预期出现 SIGSEGV。
脚本只诊断 best.spacemit.fp16.onnx，不修改系统库或全局环境。

可用环境变量:
  BENCH_PIPELINE    测试二进制，默认 build-k3/app/bench_pipeline
  MODEL / VIDEO     FP16 ONNX 模型和测试视频
  RUNS              每种配置的冷启动次数，默认 3
  BLAS_THREADS      OpenBLAS 线程数，默认 1
  EP_THREADS        SpaceMIT EP 线程数，默认 8
  CORES             宿主绑核范围，默认 0-7
  BASELINE=0        跳过无保护页基线
  GDB=1             保护页出现 SIGSEGV 后抓取卷积入口参数和故障地址
  OUTPUT_DIR        日志目录，默认在 /tmp 新建独立目录
EOF
    exit 0
fi

if [[ "$(uname -m)" != "riscv64" ]]; then
    printf '错误: 该脚本必须在 K3 riscv64 本机运行。\n' >&2
    exit 2
fi
for cmd in gcc taskset awk mktemp; do
    command -v "$cmd" >/dev/null 2>&1 || { printf '错误: 缺少命令 %s。\n' "$cmd" >&2; exit 2; }
done
if ((GDB)); then
    command -v gdb >/dev/null 2>&1 || { printf '错误: GDB=1 需要 gdb。\n' >&2; exit 2; }
fi
for path in "$BENCH_PIPELINE" "$MODEL" "$VIDEO" "$GUARD_SRC" "$TRACE_SCRIPT"; do
    [[ -f "$path" ]] || { printf '错误: 缺少文件 %s。\n' "$path" >&2; exit 2; }
done
[[ -x "$BENCH_PIPELINE" ]] || { printf '错误: 不可执行 %s。\n' "$BENCH_PIPELINE" >&2; exit 2; }
case "$RUNS" in
    ''|*[!0-9]*) printf '错误: RUNS 必须为正整数。\n' >&2; exit 2 ;;
esac
if ((RUNS < 1)); then
    printf '错误: RUNS 必须为正整数。\n' >&2
    exit 2
fi

if [[ -n "${OUTPUT_DIR:-}" ]]; then
    WORK_DIR="$OUTPUT_DIR"
    mkdir -p -- "$WORK_DIR"
else
    WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/k3-tcm-repro.XXXXXX")"
fi
if [[ -e "$WORK_DIR/results.tsv" ]]; then
    printf '错误: %s 已有复现结果，请换用新的 OUTPUT_DIR。\n' "$WORK_DIR" >&2
    exit 2
fi
GUARD_LIB="$WORK_DIR/tcm_guard.so"
gcc -shared -fPIC -O2 "$GUARD_SRC" -o "$GUARD_LIB" -ldl
ulimit -c 0

COMMON_ARGS=(
    --backend spacemit
    --model "$MODEL"
    --in0 images
    --heads output0
    --model-type v8-pose
    --num-classes 1
    --num-kpts 4
    --video "$VIDEO"
    --frames 1
    --serial
    --gpu -1
    --threads "$EP_THREADS"
    --cores-a "$CORES"
    --video-first
)
BASE_ENV=(
    env -u LD_PRELOAD
    -u SPACEMIT_EP_DUMP_SUBGRAPHS
    -u SPACEMIT_EP_DEBUG_PROFILE
    -u SPACEMIT_EP_DUMP_TENSORS
    "OPENBLAS_NUM_THREADS=$BLAS_THREADS"
)
RESULTS="$WORK_DIR/results.tsv"
printf 'mode\trun\treturn_code\ttcm_mappings\tlog\n' > "$RESULTS"
{
    printf 'binary=%s\nmodel=%s\nvideo=%s\nep_threads=%s\nblas_threads=%s\ncores=%s\n' \
        "$BENCH_PIPELINE" "$MODEL" "$VIDEO" "$EP_THREADS" "$BLAS_THREADS" "$CORES"
    printf '%q ' "${BASE_ENV[@]}" taskset -c "$CORES" "$BENCH_PIPELINE" "${COMMON_ARGS[@]}"
    printf '\n'
} > "$WORK_DIR/command.txt"

run_case() {
    local mode="$1"
    local index="$2"
    local log="$WORK_DIR/${mode}-${index}.log"
    local rc mappings=0
    if [[ "$mode" == guard ]]; then
        if "${BASE_ENV[@]}" "LD_PRELOAD=$GUARD_LIB" taskset -c "$CORES" \
            "$BENCH_PIPELINE" "${COMMON_ARGS[@]}" > "$log" 2>&1; then
            rc=0
        else
            rc=$?
        fi
        mappings="$(awk '/^TCM_GUARD / {n++} END {print n+0}' "$log")"
    else
        if "${BASE_ENV[@]}" taskset -c "$CORES" \
            "$BENCH_PIPELINE" "${COMMON_ARGS[@]}" > "$log" 2>&1; then
            rc=0
        else
            rc=$?
        fi
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' "$mode" "$index" "$rc" "$mappings" "$log" >> "$RESULTS"
    printf '%s 第 %s 次: rc=%s, TCM 映射=%s\n' "$mode" "$index" "$rc" "$mappings"
}

printf '日志目录: %s\n' "$WORK_DIR"
if ((BASELINE)); then
    for ((i = 1; i <= RUNS; i++)); do
        run_case baseline "$i"
    done
fi

guard_crashes=0
for ((i = 1; i <= RUNS; i++)); do
    run_case guard "$i"
    rc="$(awk -F '\t' -v i="$i" '$1 == "guard" && $2 == i {print $3}' "$RESULTS")"
    if [[ "$rc" == 139 ]]; then
        ((guard_crashes += 1))
    fi
done

if ((GDB && guard_crashes > 0)); then
    printf -v GDB_ARGS '%q ' "${COMMON_ARGS[@]}"
    "${BASE_ENV[@]}" gdb -q -batch \
        -ex 'set pagination off' \
        -ex 'set confirm off' \
        -ex "set exec-wrapper taskset -c $CORES" \
        -ex "set environment LD_PRELOAD $GUARD_LIB" \
        -ex "set environment OPENBLAS_NUM_THREADS $BLAS_THREADS" \
        -ex "set args $GDB_ARGS" \
        -ex start \
        -ex "source $TRACE_SCRIPT" \
        -ex continue \
        -ex 'x/i $pc' \
        "$BENCH_PIPELINE" > "$WORK_DIR/guard-gdb.log" 2>&1
    awk '/TCM_GUARD|ENTRY_CALLS|FAULT_REGS|si_addr|vle16\.v/' "$WORK_DIR/guard-gdb.log"
    printf 'GDB 日志: %s\n' "$WORK_DIR/guard-gdb.log"
fi

if ((guard_crashes > 0)); then
    printf '复现结果: %s/%s 次保护页测试出现 SIGSEGV。故障发生在用户态，脚本未修改系统库。\n' \
        "$guard_crashes" "$RUNS"
    printf '无保护页时成功不代表没有越界；需要结合 GDB 地址和 TCM 映射判断。\n'
    exit 0
fi
printf '未复现保护页 SIGSEGV。请检查日志中的 TCM_GUARD 数量和加载错误。\n' >&2
printf '保护页未崩溃不能单独证明不存在越界。\n' >&2
exit 1
