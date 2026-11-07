#pragma once

#ifdef __riscv_vector
#include <riscv_vector.h>

namespace yoloop {
namespace riscv {

inline size_t compact_indices(vbool32_t mask, size_t vl, uint32_t* out) {
    const size_t count = __riscv_vcpop_m_b32(mask, vl);
    if (count == 0) return 0;
    const vuint32m1_t ids = __riscv_vid_v_u32m1(vl);
    const vuint32m1_t packed = __riscv_vcompress_vm_u32m1(ids, mask, vl);
    __riscv_vse32_v_u32m1(out, packed, count);
    return count;
}

}  // namespace riscv
}  // namespace yoloop
#endif
