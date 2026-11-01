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

// AfRaw（v8/v9/v11 切头导出）：pred channel-major [4*reg_max + num_classes, pts.n]
//   通道 0..4*reg_max-1 : DFL logits，按边分组（l,t,r,b 各 reg_max 个）
//   通道 4*reg_max..    : num_classes 个类别 logit
// pts: build_point_table 产物（gx/gy = 网格中心 x+0.5，stride = 层步长）
// 完整 softmax/sigmoid，不做快筛；输出未做 NMS 的 Detection 列表
std::vector<Detection> decode_af_raw(const float* pred, const AnchorTable& pts,
                                     int num_classes, int reg_max, float conf_thr);

// E2E（v10 NMS-free）：pred channel-major [6, N]
//   row 0..3 = x1,y1,x2,y2（输入图像素），row 4 = score，row 5 = 类别
// 仅阈值过滤，保序输出，不做 NMS
std::vector<Detection> decode_e2e(const float* pred, int n, float conf_thr);

// class-aware 贪心 NMS：按 score 降序，同类且 IoU > nms_thr 则抑制
// 输入为 decode 输出（任意顺序），内部排序
std::vector<Detection> nms(std::vector<Detection> dets, float nms_thr);

// letterbox（双线性）+ /255 + HWC(uint8) -> CHW(float) 单 pass
// 输出 out 尺寸 3 * dst_w * dst_h，填充值 114/255
void letterbox_normalize(const uint8_t* src, int src_w, int src_h,
                         float* out, int dst_w, int dst_h);

}  // namespace ref
}  // namespace yoloop
