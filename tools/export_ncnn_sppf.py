import argparse
import os
import shutil
import sys
import tempfile

import torch
from torch import nn


class SPPFParallel(nn.Module):

    def __init__(self, orig):
        super().__init__()
        k = orig.m.kernel_size
        self.cv1 = orig.cv1
        self.cv2 = orig.cv2
        self.m1 = nn.MaxPool2d(k, 1, k // 2)
        self.m2 = nn.MaxPool2d(2 * k - 1, 1, 2 * (k // 2))
        self.m3 = nn.MaxPool2d(3 * k - 2, 1, 3 * (k // 2))

    def forward(self, x):
        y = self.cv1(x)
        return self.cv2(torch.cat([y, self.m1(y), self.m2(y), self.m3(y)], 1))


def parallelize_sppf(model):
    from ultralytics.nn.modules import SPPF

    seq = model.model 
    n = 0
    for i, m in enumerate(seq):
        if not isinstance(m, SPPF):
            continue
        par = SPPFParallel(m).eval()
        par.f, par.i = m.f, m.i
        par.type = "SPPFParallel"
        with torch.no_grad():
            x = torch.randn(1, m.cv1.conv.in_channels, 9, 9)
            d = (m(x) - par(x)).abs().max().item()
        if d != 0.0:
            raise RuntimeError(f"SPPF 并行化不等价（第 {i} 层 max|Δ|={d}），中止导出")
        seq[i] = par
        n += 1
    if n == 0:
        raise RuntimeError("模型里没有 SPPF 模块，无需本脚本")
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights", help="输入 .pt 权重")
    ap.add_argument("--imgsz", nargs=2, type=int, required=True, metavar=("H", "W"),
                    help="导出输入尺寸 [高 宽]，两边须被 32 整除")
    ap.add_argument("--out-dir", required=True, help="产物目录（不得已存在）")
    args = ap.parse_args()

    if os.path.exists(args.out_dir):
        sys.exit(f"--out-dir 已存在: {args.out_dir}")
    h, w = args.imgsz
    if h % 32 or w % 32:
        sys.exit("imgsz 两边都须被 32 整除")

    from ultralytics import YOLO
    base = os.path.basename(args.out_dir.rstrip("/"))
    stem = base[: -len("_ncnn_model")] if base.endswith("_ncnn_model") else base
    work = tempfile.mkdtemp(prefix="sppf_export_")
    try:
        staged = os.path.join(work, stem + ".pt")
        shutil.copy(args.weights, staged)
        yolo = YOLO(staged)
        n = parallelize_sppf(yolo.model)
        print(f"[sppf] 并行化替换 {n} 处 SPPF（exact-0 校验通过）", file=sys.stderr)
        yolo.export(format="ncnn", imgsz=[h, w], half=True)
        produced = os.path.join(work, stem + "_ncnn_model")
        os.makedirs(os.path.dirname(os.path.abspath(args.out_dir)), exist_ok=True)
        shutil.move(produced, args.out_dir)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"[sppf] 导出完成: {args.out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
