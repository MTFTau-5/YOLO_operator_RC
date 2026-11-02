"""Ultralytics v8/v9/v11 detect 模型 raw head 导出（ncnn）。

切分点（--cut）：
  pre-dfl  （AfRaw 契约）：Detect 头切在 DFL 之前，逐层 [4*reg_max+nc, h, w]，
             box 通道按 l/t/r/b 边分组在前，cls raw logits 在后（纯 conv+cat，
             无空间 reshape，图天然尺寸动态）。
  post-dfl （AfDist 契约）：DFL 期望留图内，逐层 [4+nc, h, w]，box 通道为已
             投影的 l/t/r/b 距离，cls raw logits；grid+dist2bbox 在 C++ 侧。

流程：monkeypatch Detect.forward → torch.onnx（静态形状）→ onnxsim → pnnx → ncnn。
pre-dfl 头部为纯 conv+cat（无空间 reshape）；v8（无 attention 模块）的整图在
ncnn 侧尺寸动态，同一 param/bin 可在任意（32 倍数的）输入尺寸下推理（已实测）；
v9 同构但跨尺寸能力**待验证**。
**v11 注意（绑定当前工具链的结论）**：YOLO11 经当前 PyTorch → ONNX → onnxsim
→ pnnx → ncnn 链路导出时，C2PSA 中的静态空间 Reshape（torch 侧 `view(B,C,H,W)`
把 H/W 烘为常量）导致同一模型跨尺寸推理数值错误，因此**当前工具链下**需逐尺寸
导出（验收见 tools/accept_raw_dynamic.py --per-size-export）。这不是 attention
在数学上不能动态——动态恢复路径（dynamo 符号 shape 导出 + pnnx inputshape2）
已实测不可行：pnnx 对命名符号维 stoi 崩溃。
依赖与已固定的工具链版本见 requirements.txt（含 onnxsim/pnnx 版本敏感行为
说明；pip install -r requirements.txt 一次装齐）。

用法：
  python tools/export_af_raw.py yolo11n.pt --out-dir OUT [--cut pre-dfl|post-dfl] \
      [--imgsz 640 640]
产物：OUT/model.ncnn.param、OUT/model.ncnn.bin（fp32），stdout 打印输出 blob 名。
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import torch
from torch import nn

CUTS = ("pre-dfl", "post-dfl")


def log_toolchain_versions():
    """导出器行为随版本变化（见模块 docstring），每次运行记录实际版本。"""
    import importlib.metadata as im

    pkgs = ["torch", "ultralytics", "onnx", "onnxsim", "onnxscript", "ncnn", "pnnx"]
    vers = []
    for p in pkgs:
        try:
            vers.append(f"{p} {im.version(p)}")
        except im.PackageNotFoundError:
            vers.append(f"{p} N/A")
    print("[toolchain] " + " / ".join(vers), file=sys.stderr)


def find_detect(model):
    from ultralytics.nn.modules.head import Detect

    dets = [m for m in model.modules() if isinstance(m, Detect)]
    if len(dets) != 1:
        sys.exit(f"期望恰好 1 个 Detect 头，找到 {len(dets)} 个（pose/seg/obb 不走本契约）")
    det = dets[0]
    if getattr(det, "end2end", False):
        sys.exit("end2end（v10）头不适用 raw 切分，请用 v10 契约")
    if not hasattr(det, "cv2") or not hasattr(det, "cv3"):
        sys.exit("Detect 头缺少 cv2/cv3（非 v8/v9/v11 同构检测头）")
    return det


def patch_detect_raw(det, cut):
    """把 Detect.forward 替换为逐层 raw 输出（list[Tensor]）。"""

    def raw_forward(self, x):
        outs = []
        bs = x[0].shape[0]
        for i in range(self.nl):
            box = self.cv2[i](x[i])  # [B, 4*reg_max, h, w]
            cls = self.cv3[i](x[i])  # [B, nc, h, w]
            if cut == "post-dfl":
                # DFL 期望在图内：[B,4rm,h,w]→[B,4,rm,hw] softmax·arange → [B,4,hw]
                d = self.dfl(box.view(bs, 4 * self.reg_max, -1))  # [B, 4, hw]
                box = d.view(bs, 4, *x[i].shape[2:])
            outs.append(torch.cat((box, cls), 1))
        return outs

    det.forward = raw_forward.__get__(det, type(det))
    return det


class _Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, x):
        return self.m(x)


def load_and_patch(weights, cut):
    from ultralytics import YOLO

    yolo = YOLO(weights)
    model = yolo.model.eval()
    det = find_detect(model)
    patch_detect_raw(det, cut)
    names = [f"out{s}" for s in (8, 16, 32)[: det.nl]]
    return yolo, model, det, names


def export_onnx(model, det, names, imgsz, onnx_path):
    h, w = imgsz
    dummy = torch.zeros(1, 3, h, w)
    with torch.no_grad():
        outs = model(dummy)
    assert isinstance(outs, (list, tuple)) and len(outs) == det.nl, (
        f"patched forward 应返回 {det.nl} 层 list，实际 {type(outs)}")
    for n, o in zip(names, outs):
        print(f"[export] {n}: {tuple(o.shape)}", file=sys.stderr)
    # 静态导出。pre-dfl 纯 conv+cat 图（v8/v9 无 attention）在 ncnn 侧天然尺寸动态；
    # v11 的 C2PSA attention（torch 侧 .view(B,C,H,W) 烘焙空间常量）同 param 换尺寸
    # 数值错误——该限制在 accept_raw_dynamic.py 中以逐尺寸导出验收覆盖。
    # （dynamo 符号 shape 导出 + pnnx inputshape2 路径已实测不可行：pnnx 对命名
    # 符号维 stoi 崩溃。）
    torch.onnx.export(
        _Wrapper(model), dummy, onnx_path,
        input_names=["in0"], output_names=names,
        opset_version=12, do_constant_folding=True,
        dynamo=False,
    )


def run_pnnx(onnx_path, imgsz, work):
    exe = shutil.which("pnnx")
    if not exe:
        sys.exit("pnnx 不在 PATH（.venv/bin 是否有 pnnx？）")
    h, w = imgsz
    cmd = [exe, onnx_path, f"inputshape=[1,3,{h},{w}]"]
    print(f"[export] {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, cwd=work, check=True)
    stem = os.path.splitext(os.path.basename(onnx_path))[0]
    param = os.path.join(work, stem + ".ncnn.param")
    binp = os.path.join(work, stem + ".ncnn.bin")
    for p in (param, binp):
        if not os.path.exists(p):
            sys.exit(f"pnnx 未产出 {p}")
    return param, binp


def ncnn_output_names(param_path, nl):
    """pnnx 把多输出重命名为 out0..out{nl-1}；从 param 校验并返回。"""
    names = [f"out{i}" for i in range(nl)]
    text = open(param_path, encoding="utf-8").read()
    for n in names:
        if f" {n}" not in text:
            sys.exit(f"ncnn param 中找不到输出 blob '{n}'（pnnx 命名约定变更？）: {param_path}")
    return names


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights", help="输入 .pt 权重（v8/v9/v11 detect）")
    ap.add_argument("--out-dir", required=True, help="产物目录（不得已存在）")
    ap.add_argument("--cut", choices=CUTS, default="pre-dfl")
    ap.add_argument("--imgsz", nargs=2, type=int, default=[640, 640],
                    metavar=("H", "W"), help="导出输入尺寸，两边须被 32 整除")
    args = ap.parse_args()

    if os.path.exists(args.out_dir):
        sys.exit(f"--out-dir 已存在: {args.out_dir}")
    log_toolchain_versions()
    h, w = args.imgsz
    if h % 32 or w % 32:
        sys.exit("imgsz 两边都须被 32 整除")
    try:
        import onnx  # noqa: F401
        import onnxsim  # noqa: F401
    except ImportError:
        sys.exit("缺依赖：pip install onnx onnxsim（torch.onnx / pnnx 前置；"
                 "未过 onnxsim 的图会让 pnnx 段错误）")

    _, model, det, names = load_and_patch(args.weights, args.cut)
    nc, rm = det.nc, det.reg_max
    exp_c = (4 * rm + nc) if args.cut == "pre-dfl" else (4 + nc)
    print(f"[export] nc={nc} reg_max={rm} nl={det.nl} cut={args.cut} "
          f"期望通道 {exp_c}", file=sys.stderr)

    work = tempfile.mkdtemp(prefix="afraw_export_")
    try:
        onnx_path = os.path.join(work, "raw.onnx")
        export_onnx(model, det, names, (h, w), onnx_path)
        # 未化简的图会让 pnnx 段错误（yolo11 PSA 块实测复现），先过 onnxsim
        import onnx
        import onnxsim
        sim_path = os.path.join(work, "raw_sim.onnx")
        sim, ok = onnxsim.simplify(onnx_path)
        if not ok:
            sys.exit("onnxsim 化简失败")
        onnx.save(sim, sim_path)
        param, binp = run_pnnx(sim_path, (h, w), work)
        ncnn_names = ncnn_output_names(param, det.nl)
        os.makedirs(os.path.dirname(os.path.abspath(args.out_dir)), exist_ok=True)
        os.makedirs(args.out_dir)
        shutil.move(param, os.path.join(args.out_dir, "model.ncnn.param"))
        shutil.move(binp, os.path.join(args.out_dir, "model.ncnn.bin"))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"[export] 完成: {args.out_dir}（heads: {','.join(ncnn_names)}）", file=sys.stderr)
    print("[export] 多尺寸运行时验收请运行 tools/accept_raw_dynamic.py", file=sys.stderr)


if __name__ == "__main__":
    main()
