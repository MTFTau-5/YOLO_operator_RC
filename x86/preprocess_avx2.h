#pragma once

#include <cstdint>

namespace yoloop {
namespace x86 {

void letterbox_normalize(const uint8_t* src, int src_w, int src_h,
                         float* out, int dst_w, int dst_h);

}  // namespace x86
}  // namespace yoloop
