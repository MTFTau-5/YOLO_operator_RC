#pragma once
// v10 端到端 NMS-free 输出（Ultralytics end2end 导出）的 C++ 侧 decode。
// 输出契约（ncnn_infer 统一转为 channel-major [6, N]）：
//   row 0..3 = x1, y1, x2, y2（输入图像素）
//   row 4    = score（已 sigmoid）
//   row 5    = 类别
// 模型自带 one-to-one 分配，仅阈值过滤，不做 NMS。

#include <vector>

#include "../common/yolo_types.h"

namespace yoloop {
namespace x86 {

// 返回阈值后的检测（保输入顺序）
std::vector<Detection> decode_e2e(const float* pred, int n, float conf_thr);

}  // namespace x86
}  // namespace yoloop
