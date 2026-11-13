#pragma once

#if defined(YOLOOP_CPU_BACKEND_RISCV)
#include "../riscv/yolo_rvv.h"
#else
#include "../x86/decode_af_dist.h"
#include "../x86/decode_af_raw.h"
#include "../x86/decode_avx2.h"
#include "../x86/decode_e2e.h"
#include "../x86/decode_v8.h"
#include "../x86/nms_simd.h"
#include "../x86/preprocess_avx2.h"
#endif

namespace yoloop {
namespace cpu {

#if defined(YOLOOP_CPU_BACKEND_RISCV)
using namespace riscv;
#else
using namespace x86;
#endif

}  // namespace cpu
}  // namespace yoloop
