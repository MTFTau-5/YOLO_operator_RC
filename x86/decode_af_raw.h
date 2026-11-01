#pragma once
// v8/v9/v11 原生 raw 输出（Detect 头切在 DFL 之前）的 C++ 侧 decode。
// 输出契约（逐层 [4*reg_max+nc, h, w]，ncnn_infer 拼接为 channel-major [C, N]）：
//   通道 0..4*reg_max-1 : DFL logits，按边分组（l,t,r,b 各 reg_max 个）
//   通道 4*reg_max..    : num_classes 个类别 logit
// pts: build_point_table 产物（gx/gy = 网格中心，stride = 层步长，逐层顺序无关）
// 与图内烘头的 AfFused 路径相比，本路径解除"换输入尺寸需重新导出模型"的限制。

#include <vector>

#include "../common/yolo_types.h"

namespace yoloop {
namespace x86 {

// 返回阈值后的候选（label = argmax 类别），未做 NMS
std::vector<Detection> decode_af_raw(const float* pred, const AnchorTable& pts,
                                     int num_classes, int reg_max, float conf_thr);

}  // namespace x86
}  // namespace yoloop
