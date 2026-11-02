"""真实模型 raw 导出 + 多尺寸动态验收驱动（对应评审"真实模型动态尺寸验收"）。

两种模式：
  默认（动态尺寸验收，v8 等纯 conv 骨干；v9 待验证）：同一 ncnn param/bin 逐尺寸
    NcnnInfer::load（N 自动发现）→ infer → 与 torch raw 对拍 → decode+nms
    → 与 python 参考 golden 框对拍；
  --per-size-export（v11 等含 attention 的模型）：**当前工具链下** C2PSA 的静态
    空间 Reshape 使同一 param 换尺寸数值错误（实测确认；非数学上不能动态，
    是本链路目前无法恢复动态 shape），故逐尺寸导出、逐尺寸对拍——验证 raw
    Detect 等价性与契约符合性，动态性不适用。v9 跨尺寸能力**待验证**。

golden：确定性输入（默认 ultralytics 自带 bus.jpg）经部署同款 letterbox 得 blob；
torch patched 模型产出逐层 raw；python 参考 decode（DFL+dist2bbox）+ NMS 出框。
末尾以 ultralytics predict（原始 PyTorch Detect 完整输出）交叉核对参考实现。

用法（仓库根目录）：
  python tools/accept_raw_dynamic.py --weights ../yolov8n.pt \
      --work-dir ../build/accept_v8 --build-dir ../build
  python tools/accept_raw_dynamic.py --weights ../yolo11n.pt \
      --work-dir ../build/accept_v11 --build-dir ../build --per-size-export
依赖与已固定的工具链版本见 requirements.txt；另需可用的 C++ 构建树
（--build-dir 指定或自动探测）。
"""

import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_af_raw import load_and_patch, log_toolchain_versions  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def letterbox(img, w, h):
    import cv2

    h0, w0 = img.shape[:2]
    s = min(w / w0, h / h0)
    nw, nh = round(w0 * s), round(h0 * s)
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((h, w, 3), 114, np.uint8)
    top, left = (h - nh) // 2, (w - nw) // 2
    out[top:top + nh, left:left + nw] = r
    return out


def sigmoid(x):
    return np.float32(1.0) / (np.float32(1.0) + np.exp(-x, dtype=np.float32))


def dfl_expect(logits):
    m = logits.max()
    p = np.exp(logits - m, dtype=np.float32)
    p /= p.sum()
    return float((p * np.arange(len(p), dtype=np.float32)).sum())


def ref_decode(raws, nc, reg_max, conf):
    """raws: [3][C, hw]（与 pred 同布局）→ dets（x1,y1,x2,y2,score,label）。"""
    dets = []
    for l, raw in enumerate(raws):
        hw = raw.shape[1]
        # 列序即 ncnn 逐层拼接序；stride 由输入/网格推得，调用方已校验一致
        box, cls = raw[: 4 * reg_max], raw[4 * reg_max:]
        for i in range(hw):
            best = int(np.argmax(cls[:, i]))
            score = float(sigmoid(cls[best, i]))
            if score < conf:
                continue
            dl = dfl_expect(box[0 * reg_max: 1 * reg_max, i])
            dt = dfl_expect(box[1 * reg_max: 2 * reg_max, i])
            dr = dfl_expect(box[2 * reg_max: 3 * reg_max, i])
            db = dfl_expect(box[3 * reg_max: 4 * reg_max, i])
            dets.append((dl, dt, dr, db, score, best, l, i))
    return dets


def to_xyxy(dets, grids):
    out = []
    for dl, dt, dr, db, score, best, l, i in dets:
        gw, gh, stride = grids[l]
        x, y = (i % gw) + 0.5, (i // gw) + 0.5
        out.append(((x - dl) * stride, (y - dt) * stride,
                    (x + dr) * stride, (y + db) * stride, score, best))
    return out


def nms(dets, nms_thr):
    """class-aware 贪心 NMS，stable sort 按 score 降序（与 tools/gen_afraw_vectors.py 一致）。"""
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
    with open(path, "wb") as f:
        f.write(np.int32(len(dets)).tobytes())
        for d in dets:
            f.write(np.asarray(d, dtype=np.float32).tobytes())


def find_build_dir(arg):
    cands = [arg] if arg else [
        os.path.join(REPO, "build"),
        os.path.join(REPO, "..", "build"),
    ]
    for c in cands:
        if c and os.path.exists(os.path.join(c, "CMakeCache.txt")):
            return os.path.abspath(c)
    sys.exit("找不到 CMake 构建树，请 --build-dir 指定")


def default_image():
    try:
        import ultralytics
        p = os.path.join(os.path.dirname(ultralytics.__file__), "assets", "bus.jpg")
        if os.path.exists(p):
            return p
    except Exception:
        pass
    sys.exit("无默认输入图（ultralytics assets/bus.jpg 缺失），请 --image 指定")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="detect 模型 .pt（v8/v9/v11）")
    ap.add_argument("--work-dir", required=True, help="产物与 case 目录（已存在则复用）")
    ap.add_argument("--sizes", default="640x640,640x480,992x544")
    ap.add_argument("--image", default=None, help="输入图（默认 ultralytics bus.jpg）")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--nms", type=float, default=0.45)
    ap.add_argument("--tol-raw", type=float, default=0.05)
    ap.add_argument("--tol-px", type=float, default=1.0)
    ap.add_argument("--tol-score", type=float, default=0.05)
    ap.add_argument("--margin", type=float, default=0.05,
                    help="贴阈值边缘带宽度：score∈[conf, conf+margin) 的框允许因"
                         "fp16/图融合抖动出没，不参与硬性对拍")
    ap.add_argument("--build-dir", default=None)
    ap.add_argument("--gpu", type=int, default=-1, help="test_raw_model 的 Vulkan 设备")
    ap.add_argument("--per-size-export", action="store_true",
                    help="逐尺寸导出验收（v11 等含 attention 的模型：同一 param 换尺寸"
                         "数值错误，须每个尺寸单独导出；契约与数值仍逐尺寸对拍）")
    args = ap.parse_args()

    log_toolchain_versions()
    import cv2
    import torch

    work = os.path.abspath(args.work_dir)
    case = os.path.join(work, "case")

    def export_at(out_dir, w, h):
        param = os.path.join(out_dir, "model.ncnn.param")
        binp = os.path.join(out_dir, "model.ncnn.bin")
        if os.path.exists(param) and os.path.exists(binp):
            print(f"[accept] 复用已导出模型 {out_dir}", file=sys.stderr)
            return param, binp
        print(f"[accept] 导出 raw ncnn → {out_dir}（{w}x{h}）", file=sys.stderr)
        rc = subprocess.run(
            [sys.executable, os.path.join(REPO, "tools", "export_af_raw.py"),
             args.weights, "--out-dir", out_dir, "--cut", "pre-dfl",
             "--imgsz", str(h), str(w)],
        ).returncode
        if rc != 0:
            sys.exit(f"导出失败（{rc}）")
        return param, binp

    sizes = []
    for s in args.sizes.split(","):
        w, h = map(int, s.split("x"))
        sizes.append((w, h))

    # 1) 导出：动态模式单一模型（纯 conv 骨干，v8/v9）；逐尺寸模式各尺寸一份（v11）
    param = binp = None
    if not args.per_size_export:
        param, binp = export_at(os.path.join(work, "model"), 640, 640)

    # 2) torch golden
    _, model, det, _ = load_and_patch(args.weights, "pre-dfl")
    nc, rm, nl = det.nc, det.reg_max, det.nl
    print(f"[accept] nc={nc} reg_max={rm} nl={nl}", file=sys.stderr)

    img = cv2.imread(args.image or default_image())  # BGR
    if img is None:
        sys.exit("输入图读取失败")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    os.makedirs(case, exist_ok=True)

    for w, h in sizes:
        tag = f"{w}x{h}"
        lb = letterbox(img, w, h)
        blob = lb.astype(np.float32) / 255.0
        blob = np.ascontiguousarray(blob.transpose(2, 0, 1))  # CHW
        blob.tofile(os.path.join(case, f"blob_{tag}.bin"))

        with torch.no_grad():
            outs = model(torch.from_numpy(blob).unsqueeze(0))
        assert len(outs) == nl
        raws = []
        grids = []
        for l, o in enumerate(outs):
            _, c, gh, gw = o.shape
            assert c == 4 * rm + nc, f"层 {l} 通道 {c} != {4 * rm + nc}"
            assert w % gw == 0 and h % gh == 0 and w // gw == h // gh
            raws.append(o[0].reshape(c, -1).numpy().astype(np.float32))
            raws[-1].tofile(os.path.join(case, f"raw_{tag}_l{l}.bin"))
            grids.append((gw, gh, w // gw))
        dets = to_xyxy(ref_decode(raws, nc, rm, args.conf), grids)
        kept = nms(dets, args.nms)
        write_dets(os.path.join(case, f"golden_{tag}.bin"), kept)
        print(f"[accept] {tag}: N={sum(gw * gh for gw, gh, _ in grids)}, "
              f"golden 框 {len(kept)}", file=sys.stderr)
        if not kept:
            sys.exit(f"{tag}: golden 无检测框（换 --image 或降 --conf），验收无意义")

    with open(os.path.join(case, "meta.txt"), "w") as f:
        f.write(f"head=afraw\nnum_classes={nc}\nreg_max={rm}\n"
                f"conf_thr={args.conf}\nnms_thr={args.nms}\n"
                f"tol_raw={args.tol_raw}\ntol_px={args.tol_px}\n"
                f"tol_score={args.tol_score}\nmargin={args.margin}\n"
                f"sizes={','.join(f'{w}x{h}' for w, h in sizes)}\n")

    # 3) C++ 验收
    build = find_build_dir(args.build_dir)
    subprocess.run(["cmake", "--build", build, "--target", "test_raw_model", "-j"],
                   check=True)
    exe = os.path.join(build, "app", "test_raw_model")
    if args.per_size_export:
        for w, h in sizes:
            p, b = export_at(os.path.join(work, f"model_{w}x{h}"), w, h)
            rc = subprocess.run([exe, "--case", case, "--param", p, "--model", b,
                                 "--gpu", str(args.gpu),
                                 "--only-size", f"{w}x{h}"]).returncode
            if rc != 0:
                sys.exit(f"C++ 验收失败（{w}x{h}）")
    else:
        rc = subprocess.run([exe, "--case", case, "--param", param, "--model", binp,
                             "--gpu", str(args.gpu)]).returncode
        if rc != 0:
            sys.exit("C++ 验收失败")

    # 4) 交叉核对：python 参考 golden vs ultralytics predict（原始 PyTorch Detect
    #    完整输出，同 letterbox 坐标系）。C++ 已对拍 golden，此步验证参考实现本身。
    from ultralytics import YOLO
    yolo = YOLO(args.weights)
    for w, h in sizes:
        tag = f"{w}x{h}"
        lb_rgb = letterbox(img, w, h)
        r = yolo.predict(cv2.cvtColor(lb_rgb, cv2.COLOR_RGB2BGR), imgsz=[h, w],
                         conf=args.conf, iou=args.nms, verbose=False)[0]
        up = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        up_score = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros(0)
        up_cls = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else np.zeros(0, int)

        with open(os.path.join(case, f"golden_{tag}.bin"), "rb") as f:
            cnt = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
            gd = np.fromfile(f, dtype=np.float32).reshape(cnt, 6)
        # 与 C++ 侧同一 margin 口径：贴阈值带内（[conf, conf+margin)）的框允许因
        # 双方 letterbox 重采样差异出没，严格框必须双向匹配
        gd_strict = np.array([d for d in gd if d[4] >= args.conf + args.margin])
        up_strict = np.array([u for u, s in zip(up, up_score)
                              if s >= args.conf + args.margin])
        used = np.zeros(len(up), bool)
        matched = 0
        for d in gd_strict:
            best, best_d = -1, 2.0  # px 容差（letterbox 重采样差异）
            for i in range(len(up)):
                if used[i] or up_cls[i] != int(d[5]):
                    continue
                dd = float(np.abs(up[i] - d[:4]).max())
                if dd < best_d:
                    best, best_d = i, dd
            if best >= 0 and abs(up_score[best] - d[4]) <= 0.05:
                used[best] = True
                matched += 1
        n_up_strict_matched = sum(1 for i in range(len(up))
                                  if used[i] and up_score[i] >= args.conf + args.margin)
        ok = (matched == len(gd_strict) == len(up_strict) == n_up_strict_matched)
        print(f"[accept] {tag}: 交叉核对 predict {len(up)} 框 vs 参考 {len(gd)} 框, "
              f"严格框 {len(gd_strict)} vs {len(up_strict)} 匹配 {matched} "
              f"{'OK' if ok else 'MISMATCH'}", file=sys.stderr)
        if not ok:
            sys.exit(f"{tag}: python 参考与原始 PyTorch Detect 输出不一致")
    print("[accept] 完成：多尺寸动态验收与交叉核对全部通过", file=sys.stderr)


if __name__ == "__main__":
    main()
