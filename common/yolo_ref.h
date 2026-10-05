#pragma once
// 标量参考实现：decode / NMS / 预处理
// 作为数值基准 —— 与 tools/gen_test_vectors.py 的 numpy 实现严格同数学

#include "yolo_types.h"

namespace yoloop {
namespace ref {

// pred: channel-major [4 + 1 + num_classes, tab.n]
// 完整 sigmoid，不做 logit 快筛；输出未做 NMS 的 Detection 列表（按 anchor 顺序）
std::vector<Detection> decode(const float* pred, const AnchorTable& tab,
                              int num_classes, float conf_thr);

// class-aware 贪心 NMS：按 score 降序，同类且 IoU > nms_thr 则抑制
// 输入为 decode 输出（任意顺序），内部排序
std::vector<Detection> nms(std::vector<Detection> dets, float nms_thr);

// letterbox（双线性）+ /255 + HWC(uint8) -> CHW(float) 单 pass
// 输出 out 尺寸 3 * dst_w * dst_h，填充值 114/255
void letterbox_normalize(const uint8_t* src, int src_w, int src_h,
                         float* out, int dst_w, int dst_h);

}  // namespace ref
}  // namespace yoloop
