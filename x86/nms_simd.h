#pragma once

#include "../common/yolo_types.h"

namespace yoloop {
namespace x86 {

std::vector<Detection> nms(std::vector<Detection> dets, float nms_thr, int topk = 300);

}  // namespace x86
}  // namespace yoloop
