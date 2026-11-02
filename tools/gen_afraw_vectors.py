from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / 'test_data'

IN_W = IN_H = 640
STRIDES = (8, 16, 32)


def build_point_table():
    """与 yoloop::build_point_table(default_point_levels(640, 640)) 一致。"""
    px, py, stride = [], [], []
    for s in STRIDES:
        gw, gh = IN_W // s, IN_H // s
        for y in range(gh):
            for x in range(gw):
                px.append(x + 0.5)
                py.append(y + 0.5)
                stride.append(float(s))
    f32 = lambda v: np.asarray(v, dtype=np.float32)
    return f32(px), f32(py), f32(stride)


def sigmoid(x):
    return np.float32(1.0) / (np.float32(1.0) + np.exp(-x, dtype=np.float32))


def dfl_expect(logits):
    """logits: [reg_max] float32 -> 标量期望。"""
    m = logits.max()
    p = np.exp(logits - m, dtype=np.float32)
    p /= p.sum()
    return float((p * np.arange(len(p), dtype=np.float32)).sum())


def decode(pred, pts, nc, reg_max, conf_thr):
    """与 ref::decode_af_raw 同数学：完整 sigmoid/softmax，无快筛。"""
    px, py, stride = pts
    N = pred.shape[1]
    box = pred[:4 * reg_max]      # [4*rm, N]，按边分组
    cls = pred[4 * reg_max:]      # [nc, N]
    dets = []
    for i in range(N):
        best = int(np.argmax(cls[:, i]))
        score = float(sigmoid(cls[best, i]))
        if score < conf_thr:
            continue
        dl = dfl_expect(box[0 * reg_max:1 * reg_max, i])
        dt = dfl_expect(box[1 * reg_max:2 * reg_max, i])
        dr = dfl_expect(box[2 * reg_max:3 * reg_max, i])
        db = dfl_expect(box[3 * reg_max:4 * reg_max, i])
        s = float(stride[i])
        dets.append(((float(px[i]) - dl) * s, (float(py[i]) - dt) * s,
                     (float(px[i]) + dr) * s, (float(py[i]) + db) * s, score, best))
    return dets


def nms(dets, nms_thr):
    """class-aware 贪心 NMS，stable sort 按 score 降序（与 gen_test_vectors.py 一致）。"""
    d = [v for _, v in sorted(enumerate(dets), key=lambda kv: -kv[1][4])]
    suppressed = [False] * len(d)
    keep = []
    for i in range(len(d)):
        if suppressed[i]:
            continue
        keep.append(d[i])
        x1a, y1a, x2a, y2a, _, la = d[i]
        area_a = (x2a - x1a) * (y2a - y1a)
        for j in range(i + 1, len(d)):
            if suppressed[j] or d[j][5] != la:
                continue
            x1b, y1b, x2b, y2b = d[j][0], d[j][1], d[j][2], d[j][3]
            iw = max(min(x2a, x2b) - max(x1a, x1b), 0.0)
            ih = max(min(y2a, y2b) - max(y1a, y1b), 0.0)
            inter = iw * ih
            ua = area_a + (x2b - x1b) * (y2b - y1b) - inter
            if ua > 0 and inter / ua > nms_thr:
                suppressed[j] = True
    return keep


def write_dets(path, dets):
    with open(path, 'wb') as f:
        f.write(np.int32(len(dets)).tobytes())
        for d in dets:
            f.write(np.asarray(d, dtype=np.float32).tobytes())


def main():
    rng = np.random.default_rng(9)
    nc, reg_max, conf, nth = 3, 16, 0.25, 0.45
    pts = build_point_table()
    N = len(pts[0])
    C = 4 * reg_max + nc

    pred = np.zeros((C, N), np.float32)
    pred[:4 * reg_max] = rng.normal(0.0, 1.2, (4 * reg_max, N)).astype(np.float32)
    pred[4 * reg_max:] = rng.normal(-6.0, 1.5, (nc, N)).astype(np.float32)
    hi = rng.choice(N, 500, replace=False)  # 500 个高分 anchor 点
    pred[4 * reg_max:, hi] = rng.normal(3.0, 1.0, (nc, 500)).astype(np.float32)
    # 人为造高重叠同类框（box logits 近似 → 解码框高重叠），让 NMS 有活干
    for k in range(20):
        j, jj = hi[k], hi[400 + k]
        pred[:4 * reg_max, jj] = pred[:4 * reg_max, j] + rng.normal(0, 0.05, 4 * reg_max)
        pred[4 * reg_max:, jj] = pred[4 * reg_max:, j] * 0.95

    dets = decode(pred, pts, nc, reg_max, conf)
    kept = nms(dets, nth)

    d = OUT / 'case9_afraw'
    d.mkdir(parents=True, exist_ok=True)
    (d / 'meta.txt').write_text(
        f'num_classes={nc}\nnum_anchors={N}\nreg_max={reg_max}\n'
        f'conf_thr={conf}\nnms_thr={nth}\ninput_w={IN_W}\ninput_h={IN_H}\n')
    pred.tofile(d / 'pred.bin')
    write_dets(d / 'golden_decode.bin', dets)
    write_dets(d / 'golden_nms.bin', kept)
    print(f'case9_afraw: N={N} nc={nc} reg_max={reg_max} -> decode {len(dets)}, nms keep {len(kept)}')


if __name__ == '__main__':
    main()
