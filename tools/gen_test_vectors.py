import os
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_data")

LEVELS = [  
    (8, 80, 80, [(10.0, 13.0), (16.0, 30.0), (33.0, 23.0)]),
    (16, 40, 40, [(30.0, 61.0), (62.0, 45.0), (59.0, 119.0)]),
    (32, 20, 20, [(116.0, 90.0), (156.0, 198.0), (373.0, 326.0)]),
]


def build_anchor_table(levels=LEVELS):
    gx, gy, stride, aw, ah = [], [], [], [], []
    for s, gw, gh, anchors in levels:
        for y in range(gh):
            for x in range(gw):
                for (w, h) in anchors:
                    gx.append(float(x)); gy.append(float(y)); stride.append(float(s))
                    aw.append(w); ah.append(h)
    f32 = lambda v: np.asarray(v, dtype=np.float32)
    return f32(gx), f32(gy), f32(stride), f32(aw), f32(ah)


def sigmoid(x):
    return np.float32(1.0) / (np.float32(1.0) + np.exp(-x, dtype=np.float32))


def decode(pred, tab, num_classes, conf_thr):
    """pred: [C, N] float32 channel-major。完整 sigmoid，与参考实现一致。"""
    gx, gy, stride, aw, ah = tab
    N = pred.shape[1]
    dets = []
    obj = sigmoid(pred[4])
    for i in range(N):
        if obj[i] < conf_thr:
            continue
        cls_logits = pred[5:5 + num_classes, i]
        best = int(np.argmax(cls_logits))
        score = float(np.float32(obj[i]) * sigmoid(cls_logits[best]))
        if score < conf_thr:
            continue
        x = float((sigmoid(pred[0, i]) * 2.0 - 0.5 + gx[i]) * stride[i])
        y = float((sigmoid(pred[1, i]) * 2.0 - 0.5 + gy[i]) * stride[i])
        w = float((sigmoid(pred[2, i]) * 2.0) ** 2 * aw[i])
        h = float((sigmoid(pred[3, i]) * 2.0) ** 2 * ah[i])
        dets.append((x - w / 2, y - h / 2, x + w / 2, y + h / 2, score, best))
    return dets


def nms(dets, nms_thr):
    """class-aware 贪心 NMS，stable sort 按 score 降序。"""
    dets = sorted(enumerate(dets), key=lambda kv: -kv[1][4])  # stable: python sort
    order = [i for i, _ in dets]
    d = [v for _, v in dets]
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
            iou = inter / ua if ua > 0 else 0.0
            if iou > nms_thr:
                suppressed[j] = True
    return keep
    assert order  # silence linter


def letterbox_normalize(img, dst_w, dst_h):
    """img: uint8 HWC -> CHW float32，双线性，pad=114/255。"""
    src_h, src_w = img.shape[:2]
    scale = min(dst_w / src_w, dst_h / src_h)
    new_w, new_h = int(round(src_w * scale)), int(round(src_h * scale))
    pad_x, pad_y = (dst_w - new_w) // 2, (dst_h - new_h) // 2
    out = np.full((3, dst_h, dst_w), 114.0 / 255.0, dtype=np.float32)
    for dy in range(new_h):
        sy = (dy + 0.5) / scale - 0.5
        y0 = int(np.floor(sy)); fy = np.float32(sy - y0)
        y1 = y0 + 1
        y0c, y1c = min(max(y0, 0), src_h - 1), min(max(y1, 0), src_h - 1)
        for dx in range(new_w):
            sx = (dx + 0.5) / scale - 0.5
            x0 = int(np.floor(sx)); fx = np.float32(sx - x0)
            x1 = x0 + 1
            x0c, x1c = min(max(x0, 0), src_w - 1), min(max(x1, 0), src_w - 1)
            v00 = img[y0c, x0c].astype(np.float32)
            v01 = img[y0c, x1c].astype(np.float32)
            v10 = img[y1c, x0c].astype(np.float32)
            v11 = img[y1c, x1c].astype(np.float32)
            v = (v00 * (1 - fx) + v01 * fx) * (1 - fy) + (v10 * (1 - fx) + v11 * fx) * fy
            out[:, pad_y + dy, pad_x + dx] = v / np.float32(255.0)
    return out


def write_dets(path, dets):
    with open(path, "wb") as f:
        f.write(np.int32(len(dets)).tobytes())
        for d in dets:
            f.write(np.asarray(d, dtype=np.float32).tobytes())


def write_meta(path, kv):
    with open(path, "w") as f:
        for k, v in kv.items():
            f.write(f"{k}={v}\n")


def gen_decode_case(name, num_classes, conf_thr, nms_thr, seed, mode):
    rng = np.random.default_rng(seed)
    tab = build_anchor_table()
    N = len(tab[0])
    C = 4 + 1 + num_classes
    pred = rng.normal(0.0, 1.0, size=(C, N)).astype(np.float32)
    if mode == "sparse":
        pred[4] = rng.normal(-8.0, 1.5, size=N).astype(np.float32)
        idx = rng.choice(N, size=12, replace=False)
        pred[4, idx] = rng.normal(4.0, 1.0, size=12).astype(np.float32)
    elif mode == "dense":
        pred[4] = rng.normal(-5.0, 1.5, size=N).astype(np.float32)
        # 在 stride=8 level 中部制造一个高密度簇：大量重叠候选
        cluster = []
        for i in range(N):
            if tab[2][i] == 8.0 and 30 <= tab[0][i] <= 50 and 30 <= tab[1][i] <= 50:
                cluster.append(i)
        cluster = np.asarray(cluster)
        pred[4, cluster] += 6.0
        pred[0, cluster] *= 0.05
        pred[1, cluster] *= 0.05
        pred[2, cluster] *= 0.1
        pred[3, cluster] *= 0.1
    elif mode == "few_classes":
        pred[4] = rng.normal(-6.0, 2.0, size=N).astype(np.float32)
        idx = rng.choice(N, size=80, replace=False)
        pred[4, idx] = rng.normal(3.0, 1.0, size=80).astype(np.float32)
    dets = decode(pred, tab, num_classes, conf_thr)
    kept = nms(dets, nms_thr)

    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    pred.tofile(os.path.join(d, "pred.bin"))
    write_dets(os.path.join(d, "golden_decode.bin"), dets)
    write_dets(os.path.join(d, "golden_nms.bin"), kept)
    write_meta(os.path.join(d, "meta.txt"), {
        "num_anchors": N, "num_classes": num_classes,
        "conf_thr": conf_thr, "nms_thr": nms_thr,
        "input_w": 640, "input_h": 640,
        "decode_count": len(dets), "nms_count": len(kept),
    })
    print(f"[{name}] N={N} C={C} decode={len(dets)} nms={len(kept)}")


def gen_preprocess_case(name, src_w, src_h, dst, seed):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, size=(src_h, src_w, 3), dtype=np.uint8)
    out = letterbox_normalize(img, dst, dst)
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    img.tofile(os.path.join(d, "img.bin"))
    out.astype(np.float32).tofile(os.path.join(d, "golden_pre.bin"))
    write_meta(os.path.join(d, "meta.txt"), {
        "img_w": src_w, "img_h": src_h, "dst_w": dst, "dst_h": dst,
    })
    print(f"[{name}] {src_w}x{src_h} -> {dst}x{dst}")


if __name__ == "__main__":
    gen_decode_case("case1_sparse", 80, 0.25, 0.45, seed=42, mode="sparse")
    gen_decode_case("case2_dense", 80, 0.10, 0.45, seed=43, mode="dense")
    gen_decode_case("case3_few_classes", 10, 0.25, 0.50, seed=44, mode="few_classes")
    gen_preprocess_case("case4_preprocess", 1280, 720, 640, seed=45)
    gen_preprocess_case("case5_preprocess_odd", 811, 603, 640, seed=46)
