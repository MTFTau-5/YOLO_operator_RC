#!/usr/bin/env python3
"""生成 Tracker golden —— 直接调用 RK3588-AntiUAV-Tracker 的 BoxKalman/门控逻辑。

合成 4 段检测序列（静止 / 匀速 / 丢失-恢复 / 出界滑行），逐帧记录
predict 输出框、update 后输出框、KF 内部状态，写 test_data/case6_tracker/。
C++ 侧 test_tracker 回放比对（坐标误差 < 1e-3）。

用法: python3 tools/gen_tracker_golden.py [RK3588_tracker_dir] [out_dir]
run.py 顶层 import cv2（本机无），BoxKalman 路径实际不用，stub 掉即可。
"""
from __future__ import annotations

import csv
import sys
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RK_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO.parent / 'RK3588-AntiUAV-Tracker' / 'tracker'
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / 'test_data' / 'case6_tracker'

sys.modules.setdefault('cv2', types.ModuleType('cv2'))  # BoxKalman 不触 cv2
sys.path.insert(0, str(RK_DIR))
from run import BoxKalman  # noqa: E402
from tracker_core import clamp_bbox  # noqa: E402  coast 输出与 track3.py 一致要过 clamp

W, H = 1280, 720


def center(b):
    return np.asarray(b[:2], float) + np.asarray(b[2:], float) / 2


def build_frames():
    """返回 [(frame_idx, [xywh 候选框...]), ...]。空列表 = 该帧无检测。"""
    frames = []
    # 段1: 静止目标（轻微抖动），10 帧
    for i in range(10):
        j = np.sin(i * 1.7) * 0.8
        frames.append([[100 + j, 200 - j, 40, 30]])
    # 段2: 匀速运动 vx=5, vy=-2，12 帧
    for i in range(12):
        frames.append([[110 + 5 * i, 195 - 2 * i, 40, 30]])
    # 段3: 3 帧丢失后恢复（coast 内），随后检测带一个尺寸异常干扰框
    for i in range(3):
        frames.append([])
    frames.append([[170, 173, 40, 30], [400, 300, 400, 300]])  # 第二个面积比超限
    for i in range(3):
        frames.append([[175 + 5 * i, 171 - 2 * i, 40, 30]])
    # 段4: 8 帧全丢（超出 coast_frames=5 → LOST），再重新锁定
    for i in range(8):
        frames.append([])
    for i in range(5):
        frames.append([[600 + 3 * i, 400, 50, 50]])
    return list(enumerate(frames))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    kal = None
    ever = False
    misses = 0
    coast_frames, max_misses = 5, 15
    gate_px, gate_scale = 120.0, 2.0
    size_ratio, size_mem = 3.0, 120
    last_box = None
    last_frame = -10**6

    det_rows = []    # frame, x1, y1, x2, y2, score
    golden_rows = []  # frame, phase, px,py,pw,ph, bx,by,bw,bh, x0..x7

    for idx, cands in build_frames():
        cands = [np.asarray(c, float) for c in cands]
        for j, c in enumerate(cands):
            det_rows.append([idx, c[0], c[1], c[0] + c[2], c[1] + c[3], 0.9 - j * 0.3])

        # ---- 尺寸记忆过滤（track3.py _size_ok）
        elig = cands
        if last_box is not None and idx - last_frame <= size_mem:
            prev_area = float(np.prod(last_box[2:]))
            elig = [c for c in cands
                    if 1 / size_ratio <= np.prod(c[2:]) / max(prev_area, 1.) <= size_ratio]

        pred_box = [np.nan] * 4
        if kal is not None:
            pred_box = kal.predict().tolist()

        if kal is None:
            if elig:
                c = max(elig, key=lambda c: 0.9)  # 单候选/首候选即最高分
                kal = BoxKalman(c)
                misses = 0
                ever = True
                last_box = np.asarray(c, float)
                last_frame = idx
                out = np.asarray(c, float)
            else:
                out = np.asarray([np.nan] * 4)
        else:
            cand = None
            if elig:
                cand = min(elig, key=lambda c: np.linalg.norm(center(c) - center(pred_box)))
                dist = float(np.linalg.norm(center(cand) - center(pred_box)))
                gate = max(gate_px, gate_scale * np.linalg.norm(np.asarray(pred_box)[2:]))
                gate *= 1 + min(misses, max_misses) / max_misses
                if dist > gate:
                    cand = None
            if cand is not None:
                kal.update(np.asarray(cand, float))
                misses = 0
                last_box = np.asarray(cand, float)
                last_frame = idx
                out = np.asarray(cand, float)
            else:
                misses += 1
                if misses <= coast_frames:
                    out = clamp_bbox(np.asarray(pred_box, float), (H, W)).astype(float)
                else:
                    kal = None
                    misses = 0
                    out = np.asarray([np.nan] * 4)

        state = kal.x.tolist() if kal is not None else [np.nan] * 8
        golden_rows.append([idx] + pred_box + out.tolist() + state)

    with (OUT / 'dets.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frame', 'x1', 'y1', 'x2', 'y2', 'score'])
        w.writerows(det_rows)
    with (OUT / 'golden.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frame', 'pred_x', 'pred_y', 'pred_w', 'pred_h',
                    'out_x', 'out_y', 'out_w', 'out_h',
                    'x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'])
        w.writerows(golden_rows)
    (OUT / 'meta.txt').write_text(
        f'frames={len(golden_rows)} img_w={W} img_h={H} coast_frames={coast_frames} '
        f'max_misses={max_misses} gate_px={gate_px} gate_scale={gate_scale} '
        f'size_ratio={size_ratio} size_mem={size_mem}\n')
    print(f'golden: {len(golden_rows)} frames -> {OUT}')


if __name__ == '__main__':
    main()
