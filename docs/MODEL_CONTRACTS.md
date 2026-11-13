# 模型契约与调用

算子使用 float32 channel-major `[C,N]`；推理适配层负责 batch=1 的布局转换。框坐标属于网络输入尺度，应用在跟踪前还原 letterbox。输出名以实际模型为准。

| model-type | 模型输出 | 后处理 |
|---|---|---|
| `v5` | 逐层 `[na*(5+nc),h,w]` 原始 logits | 重排后 anchor decode + NMS |
| `v8/v9/v11` | `[1,4+nc,N]`，xywh + 类别概率 | 阈值筛选 + NMS |
| `v8-pose` | `[1,4+nc+3K,N]` | 框筛选、NMS 同步保留关键点 |
| `v8/v9/v11-raw` | 逐层 `[4*reg_max+nc,h,w]` logits | DFL + grid/dist2bbox + NMS |
| `v8/v9/v11-dist` | 逐层 `[4+nc,h,w]`，四边距离 + 类别 logits | sigmoid + grid/dist2bbox + NMS |
| `v10` | `[1,N,6]`，xyxy/score/class | 阈值筛选，不执行 NMS |

v5 每个 anchor 的通道为 tx/ty/tw/th/obj/cls；anchor 按 stride 8/16/32、网格行优先、anchor 索引最快排列。训练修改过 anchors 时必须显式配置。raw 按 l/t/r/b 分组，每边 `reg_max` 个 logits；网格表由输出尺寸推导。fused 概率不应再次 sigmoid。

输入尺寸须按模型契约配置。fused 导出含固定网格时换尺寸需重导；v8 raw 动态尺寸已有验收，v9 待验证；当前 v11 导出链路包含固定 attention reshape，使用逐尺寸导出。raw 输出更大，是否更快应比较完整链路。

## 导出与应用

```bash
python tools/export_af_raw.py best.pt --out-dir <dir>                # pre-DFL
python tools/export_af_raw.py best.pt --out-dir <dir> --cut post-dfl
python tools/export_ncnn_sppf.py best.pt --imgsz 640 640 --out-dir <dir>
```

Ultralytics fused 导出可用 `YOLO('best.pt').export(format='ncnn', imgsz=[H,W], half=True)`；NVIDIA 可导出 ONNX 后构建 TensorRT engine。完整应用参数及运行实例见 [app/README.md](../app/README.md) 与 `app/pipeline.example.yaml`。

```yaml
backend: spacemit
model: ../data/best.spacemit.fp16.onnx
in0: images
heads: [output0]
model-type: v8-pose
in-size: 640x640
num-classes: 1
num-kpts: 4
```

ncnn 则设 `backend: ncnn`、`param` / `model`、对应输入输出名称。`conf/nms/topk` 控制筛选；`gate-px/gate-scale/size-ratio/size-memory/coast/max-misses` 控制跟踪；`serial/cores-a/cores-b/max-w/max-h/max-frames` 控制执行；`udp` 指定控制目标。E2E 路径不使用 NMS/topk。

## C++ 与 Python

```cpp
#include "cpu_ops.h"
using namespace yoloop;
auto tab = build_anchor_table(default_levels(640, 640));
cpu::letterbox_normalize(bgr, width, height, blob, 640, 640);
// 推理后得到 [5+nc,N] 原始 v5 head：
auto kept = cpu::nms(cpu::decode(pred, tab, nc, 0.25f), 0.45f, 300);
```

链接对应 CPU 静态库、`yoloop_common` 和 OpenMP。其他 head 使用 `decode_v8`、`decode_v8_pose`、`decode_af_raw`、`decode_af_dist`、`decode_e2e`；`nms_indices` 返回输入下标，供关键点同步筛选。

```python
import yolo_ops_cuda
# pred: float32 [5+nc,N]，img: uint8 [H,W,3]
dets = yolo_ops_cuda.decode(pred, num_classes=80, conf_thr=0.25,
                            input_w=640, input_h=640)
kept = yolo_ops_cuda.nms(dets, nms_thr=0.45)
blob = yolo_ops_cuda.letterbox(img, 640, 640)
```

C++ CUDA 接口位于 `nvidia/yolo_ops.h`：复用 `CudaAnchorTable`，在同一 stream 调用 `letterbox_normalize → 推理 → decode → nms`。decode 的计数可能超过输出容量，消费时取 `min(count,cap)`。

TRT `YoloDecodePlugin` 位于 `nvidia/trt/yolo_decode_plugin.cpp`，字段为 `num_classes/conf_thr/input_w/input_h/max_output_boxes`；输入 `[1,C,N]`，输出 `[1,max_boxes,6]` 和 count。用 `test_trt_plugin` 验证序列化和执行。
