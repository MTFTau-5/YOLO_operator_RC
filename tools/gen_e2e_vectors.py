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


def main():
    rng = np.random.default_rng(10)
    nc, n, conf = 3, 300, 0.25

    pred = np.zeros((6, n), np.float32)
    pred[0] = rng.uniform(0, 560, n)                    # x1
    pred[1] = rng.uniform(0, 560, n)                    # y1
    pred[2] = pred[0] + rng.uniform(4, 80, n)           # x2
    pred[3] = pred[1] + rng.uniform(4, 80, n)           # y2
    pred[4] = rng.uniform(0, 0.2, n)                    # score 大部分低于阈值
    hi = rng.choice(n, 60, replace=False)
    pred[4, hi] = rng.uniform(0.3, 0.99, 60)
    pred[5] = rng.integers(0, nc, n).astype(np.float32)  # 类别

    dets = []
    for j in range(n):
        if pred[4, j] < conf:
            continue
        dets.append((float(pred[0, j]), float(pred[1, j]), float(pred[2, j]),
                     float(pred[3, j]), float(pred[4, j]), int(pred[5, j])))

    d = OUT / 'case10_e2e'
    d.mkdir(parents=True, exist_ok=True)
    (d / 'meta.txt').write_text(
        f'num_classes={nc}\nnum_anchors={n}\nconf_thr={conf}\n')
    pred.tofile(d / 'pred.bin')
    write_dets(d / 'golden_decode.bin', dets)
    print(f'case10_e2e: N={n} nc={nc} -> {len(dets)} dets (no nms)')


if __name__ == '__main__':
    main()
