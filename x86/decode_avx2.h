#pragma once

#include "../common/yolo_types.h"

namespace yoloop {
namespace x86 {

std::vector<Detection> decode(const float* pred, const AnchorTable& tab,
                              int num_classes, float conf_thr);

}  // namespace x86
}  // namespace yoloop
