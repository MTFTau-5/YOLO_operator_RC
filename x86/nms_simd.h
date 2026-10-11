#pragma once

#include "../common/yolo_types.h"

namespace yoloop {
namespace x86 {

std::vector<Detection> nms(std::vector<Detection> dets, float nms_thr, int topk = 300);

// 与 nms 同一实现路径，但返回保留框在**输入数组中的索引**（score 降序）。
// 供 pose 等需要把附加数据（关键点）随框穿过 NMS 的场景使用。
std::vector<int32_t> nms_indices(const std::vector<Detection>& dets, float nms_thr,
                                 int topk = 300);

}  // namespace x86
}  // namespace yoloop
