#!/usr/bin/env python3
"""生成 YOLOv8/v11（Detect 头烘在图内）decode 的 golden 用例。

输出契约（channel-major [C, N]，box=xywh 像素，cls=已 sigmoid 概率）：
  case7_v8_detect : C = 4 + nc
  case8_v8_pose   : C = 4 + nc + 3*nkpt，关键点 (x, y, conf) 像素

文件格式与 tools/gen_test_vectors.py 一致；pose 另有 golden_pose.bin：
  int32 count，随后每框 6 个 float（同 golden_decode）+ 3*nkpt 个 float 关键点，
  顺序 = nms_indices 保留顺序（score 降序，并列按原索引升序，topk 先截断）。

用法: python3 tools/gen_v8_vectors.py [test_data_dir]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / 'test_data'


def write_dets(path, dets):
    with open(path, 'wb') as f:
        f.write(np.int32(len(dets)).tobytes())
        for d in dets:
            f.write(np.asarray(d, dtype=np.float32).tobytes())


def nms_indices(dets, thr, topk):
    """与 x86::nms_indices 同语义：score 降序（并列索引升序），topk 先截断，类感知。"""
    order = sorted(range(len(dets)), key=lambda i: (-dets[i][4], i))
    if topk > 0:
        order = order[:topk]
    suppressed = [False] * len(dets)
    keep = []
    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        a = dets[i]
        for j in order:
            if suppressed[j] or j == i or dets[j][5] != a[5]:
                continue
            b = dets[j]
            iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
            inter = iw * ih
            ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
            if ua > 0 and inter / ua > thr:
                suppressed[j] = True
    return keep


def decode(pred, nc, n, thr):
    """逐 anchor argmax 类别 + 阈值 + xywh→xyxy。返回 list[[x1,y1,x2,y2,score,label]]。"""
    box = pred[:4]                      # [4, N]
    cls = pred[4:4 + nc]                # [nc, N]
    label = cls.argmax(axis=0)
    score = cls[label, np.arange(n)]
    alive = score >= thr
    out = []
    for j in np.nonzero(alive)[0]:
        cx, cy, w, h = box[:, j]
        out.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2,
                    float(score[j]), int(label[j])])
    return out


def gen_detect():
    rng = np.random.default_rng(7)
    nc, n, conf, nth = 3, 8400, 0.25, 0.45
    pred = np.zeros((4 + nc, n), np.float32)
    pred[0:2] = rng.uniform(0, 640, (2, n))          # cx, cy
    pred[2:4] = rng.uniform(4, 200, (2, n))          # w, h
    pred[4:] = rng.uniform(0, 0.2, (nc, n))          # 大部分低分
    hi = rng.choice(n, 500, replace=False)           # 500 个高分 anchor
    pred[4:, hi] = rng.uniform(0.3, 0.99, (nc, 500))
    # 人为造几对高重叠同类框，让 NMS 有活干
    for k in range(20):
        j, jj = hi[k], hi[400 + k]
        pred[:4, jj] = pred[:4, j] + rng.uniform(-2, 2, 4)
        pred[4:, jj] = pred[4:, j] * 0.95

    dets = decode(pred, nc, n, conf)
    d = OUT / 'case7_v8_detect'
    d.mkdir(parents=True, exist_ok=True)
    (d / 'meta.txt').write_text(
        f'num_classes={nc}\nnum_anchors={n}\nconf_thr={conf}\nnms_thr={nth}\n')
    pred.tofile(d / 'pred.bin')
    write_dets(d / 'golden_decode.bin', dets)
    print(f'case7_v8_detect: N={n} nc={nc} -> {len(dets)} dets')


def gen_pose():
    rng = np.random.default_rng(8)
    nc, nk, n, conf, nth, topk = 1, 7, 8400, 0.25, 0.45, 100
    C = 4 + nc + 3 * nk
    pred = np.zeros((C, n), np.float32)
    pred[0:2] = rng.uniform(0, 640, (2, n))
    pred[2:4] = rng.uniform(4, 200, (2, n))
    pred[4:4 + nc] = rng.uniform(0, 0.2, (nc, n))
    hi = rng.choice(n, 300, replace=False)
    pred[4:4 + nc, hi] = rng.uniform(0.3, 0.99, (nc, 300))
    for k in range(15):  # 高重叠对
        pred[:4, hi[200 + k]] = pred[:4, hi[k]] + rng.uniform(-2, 2, 4)
        pred[4, hi[200 + k]] = pred[4, hi[k]] * 0.95
    kb = 4 + nc
    pred[kb:kb + 3 * nk:3] = rng.uniform(0, 640, (nk, n))    # kpt x
    pred[kb + 1:kb + 3 * nk:3] = rng.uniform(0, 640, (nk, n))  # kpt y
    pred[kb + 2:kb + 3 * nk:3] = rng.uniform(0, 1, (nk, n))    # kpt conf

    dets = decode(pred, nc, n, conf)
    alive = np.nonzero(pred[4:4 + nc].max(axis=0) >= conf)[0]
    assert len(alive) == len(dets)
    kpts = pred[kb:]                                   # [3K, N]
    alive_kpts = [kpts[:, j] for j in alive]
    keep = nms_indices(dets, nth, topk)

    d = OUT / 'case8_v8_pose'
    d.mkdir(parents=True, exist_ok=True)
    (d / 'meta.txt').write_text(
        f'num_classes={nc}\nnum_kpts={nk}\nnum_anchors={n}\nconf_thr={conf}\n'
        f'nms_thr={nth}\ntopk={topk}\n')
    pred.tofile(d / 'pred.bin')
    with open(d / 'golden_pose.bin', 'wb') as f:
        f.write(np.int32(len(keep)).tobytes())
        for i in keep:
            f.write(np.asarray(dets[i], np.float32).tobytes())
            f.write(np.asarray(alive_kpts[i], np.float32).tobytes())
    print(f'case8_v8_pose: N={n} nc={nc} nkpt={nk} -> decode {len(dets)}, nms keep {len(keep)}')


if __name__ == '__main__':
    gen_detect()
    gen_pose()
