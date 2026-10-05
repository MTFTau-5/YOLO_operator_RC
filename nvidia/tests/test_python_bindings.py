#!/usr/bin/env python3
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))
import gen_test_vectors as gtv  # noqa: E402

import yolo_ops_cuda  # noqa: E402

COORD_TOL = 1e-2
SCORE_TOL = 1e-3


def greedy_match(got, ref):
    got = np.asarray(got, dtype=np.float32).reshape(-1, 6)
    ref = np.asarray(ref, dtype=np.float32).reshape(-1, 6)
    if len(got) != len(ref):
        return -1, float("nan"), float("nan")
    used = np.zeros(len(ref), dtype=bool)
    unmatched = 0
    max_ce = max_se = 0.0
    for g in got:
        found = False
        for j in range(len(ref)):
            if used[j] or int(ref[j, 5]) != int(g[5]):
                continue
            ce = float(np.max(np.abs(g[:4] - ref[j, :4])))
            se = float(abs(g[4] - ref[j, 4]))
            if ce < COORD_TOL and se < SCORE_TOL:
                used[j] = True
                found = True
                max_ce = max(max_ce, ce)
                max_se = max(max_se, se)
                break
        if not found:
            unmatched += 1
    return unmatched, max_ce, max_se


def run_decode_case(iw, ih, rng, fails):
    """构建 (iw, ih) 对应的 anchor 表与随机 pred, 对比 CUDA decode 与 numpy 参考"""
    levels = [(8, iw // 8, ih // 8, [(10.0, 13.0), (16.0, 30.0), (33.0, 23.0)]),
              (16, iw // 16, ih // 16, [(30.0, 61.0), (62.0, 45.0), (59.0, 119.0)]),
              (32, iw // 32, ih // 32, [(116.0, 90.0), (156.0, 198.0), (373.0, 326.0)])]
    tab = gtv.build_anchor_table(levels)
    N = len(tab[0])
    num_classes, conf_thr = 80, 0.25
    C = 4 + 1 + num_classes
    pred = rng.normal(0.0, 1.0, size=(C, N)).astype(np.float32)
    pred[4] = rng.normal(-3.0, 1.5, size=N).astype(np.float32)
    dets = yolo_ops_cuda.decode(pred, num_classes, conf_thr, input_w=iw, input_h=ih)
    ref = gtv.decode(pred, tab, num_classes, conf_thr)
    un, ce, se = greedy_match(dets, np.asarray(ref, dtype=np.float32))
    ok = un == 0
    print(f"[decode {iw}x{ih}] cuda={len(dets)} ref={len(ref)} unmatched={un} "
          f"max_coord_err={ce:.3e} max_score_err={se:.3e} -> {'PASS' if ok else 'FAIL'}")
    return fails + (not ok), N


def main():
    num_classes, conf_thr, nms_thr = 80, 0.25, 0.45
    rng = np.random.default_rng(7)
    fails = 0

    fails, N = run_decode_case(640, 640, rng, fails)
    fails, _ = run_decode_case(992, 992, rng, fails)
    fails, _ = run_decode_case(992, 544, rng, fails)
    tab = gtv.build_anchor_table()
    C = 4 + 1 + num_classes
    pred = rng.normal(0.0, 1.0, size=(C, N)).astype(np.float32)
    pred[4] = rng.normal(-3.0, 1.5, size=N).astype(np.float32)
    ref = gtv.decode(pred, tab, num_classes, conf_thr)
    dets6 = np.asarray(ref, dtype=np.float32).reshape(-1, 6)
    kept = yolo_ops_cuda.nms(dets6, nms_thr)
    ref_kept = gtv.nms(ref, nms_thr)
    un, ce, se = greedy_match(kept, np.asarray(ref_kept, dtype=np.float32))
    ok = un == 0
    print(f"[nms] cuda={len(kept)} ref={len(ref_kept)} unmatched={un} "
          f"max_coord_err={ce:.3e} max_score_err={se:.3e} -> {'PASS' if ok else 'FAIL'}")
    fails += (not ok)

    try:
        yolo_ops_cuda.decode(np.zeros((C, 10), np.float32), num_classes, conf_thr,
                             input_w=1000, input_h=640)
        print("[decode 1000x640] 未抛异常 -> FAIL")
        fails += 1
    except ValueError:
        print("[decode 1000x640] 正确抛出 ValueError(1000 不能被 32 整除) -> PASS")

    img = rng.integers(0, 256, size=(603, 811, 3), dtype=np.uint8)
    out = yolo_ops_cuda.letterbox(img, 640, 640)
    ref_pre = gtv.letterbox_normalize(img, 640, 640)
    maxerr = float(np.max(np.abs(out - ref_pre)))
    ok = maxerr < 1e-3
    print(f"[letterbox] out{out.shape} max_abs_err={maxerr:.3e} (tol 1e-3) "
          f"-> {'PASS' if ok else 'FAIL'}")
    fails += (not ok)

    print("==== ALL PASS ====" if fails == 0 else f"==== FAIL ({fails}) ====")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
