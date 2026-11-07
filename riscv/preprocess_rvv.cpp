#include "yolo_rvv.h"

#include <algorithm>
#include <cmath>
#include <vector>

#ifdef __riscv_vector
#include <riscv_vector.h>
#endif

namespace yoloop {
namespace riscv {

void letterbox_normalize(const uint8_t* src, int src_w, int src_h,
                         float* out, int dst_w, int dst_h) {
    const float scale = std::min((float)dst_w / src_w, (float)dst_h / src_h);
    const int new_w = (int)roundf(src_w * scale);
    const int new_h = (int)roundf(src_h * scale);
    const int pad_x = (dst_w - new_w) / 2;
    const int pad_y = (dst_h - new_h) / 2;
    const float pad_val = 114.f / 255.f;
    const int plane = dst_w * dst_h;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < 3 * plane; ++i) out[i] = pad_val;

    std::vector<int> x0t(new_w), x1t(new_w);
    std::vector<float> fxt(new_w);
    for (int dx = 0; dx < new_w; ++dx) {
        const float sx = (dx + 0.5f) / scale - 0.5f;
        int x0 = (int)floorf(sx);
        fxt[dx] = sx - x0;
        const int x1 = x0 + 1;
        x0t[dx] = std::max(0, std::min(x0, src_w - 1));
        x1t[dx] = std::max(0, std::min(x1, src_w - 1));
    }
#ifdef __riscv_vector
    std::vector<uint32_t> x0off(new_w), x1off(new_w);
    for (int dx = 0; dx < new_w; ++dx) {
        x0off[dx] = (uint32_t)x0t[dx] * 3u;
        x1off[dx] = (uint32_t)x1t[dx] * 3u;
    }
    const size_t total_bytes = (size_t)src_w * (size_t)src_h * 3u;
#endif

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int dy = 0; dy < new_h; ++dy) {
        const float sy = (dy + 0.5f) / scale - 0.5f;
        int y0 = (int)floorf(sy);
        const float fy = sy - y0;
        int y1 = y0 + 1;
        y0 = std::max(0, std::min(y0, src_h - 1));
        y1 = std::max(0, std::min(y1, src_h - 1));
        const uint8_t* row0 = src + (size_t)y0 * (size_t)src_w * 3u;
        const uint8_t* row1 = src + (size_t)y1 * (size_t)src_w * 3u;
        const size_t dst_row = (size_t)(pad_y + dy) * (size_t)dst_w + (size_t)pad_x;

        int dx = 0;
#ifdef __riscv_vector
        while (dx < new_w) {
            const size_t vl = __riscv_vsetvl_e32m1((size_t)(new_w - dx));
            const size_t max_off =
                (size_t)((size_t)y1 * (size_t)src_w + (size_t)x1t[dx + (int)vl - 1]) * 3u + 2u;
            if (max_off + 4u > total_bytes) break;
            const vfloat32m1_t fx = __riscv_vle32_v_f32m1(fxt.data() + dx, vl);
            const vfloat32m1_t one = __riscv_vfmv_v_f_f32m1(1.f, vl);
            const vfloat32m1_t fx1 = __riscv_vfsub_vv_f32m1(one, fx, vl);
            const vuint32m1_t ox0 = __riscv_vle32_v_u32m1(x0off.data() + dx, vl);
            const vuint32m1_t ox1 = __riscv_vle32_v_u32m1(x1off.data() + dx, vl);
            for (int c = 0; c < 3; ++c) {
                const vuint32m1_t c0 = __riscv_vadd_vx_u32m1(ox0, (uint32_t)c, vl);
                const vuint32m1_t c1 = __riscv_vadd_vx_u32m1(ox1, (uint32_t)c, vl);
                const vuint32m1_t raw00 = __riscv_vloxei32_v_u32m1(
                    (const uint32_t*)row0, c0, vl);
                const vuint32m1_t raw01 = __riscv_vloxei32_v_u32m1(
                    (const uint32_t*)row0, c1, vl);
                const vuint32m1_t raw10 = __riscv_vloxei32_v_u32m1(
                    (const uint32_t*)row1, c0, vl);
                const vuint32m1_t raw11 = __riscv_vloxei32_v_u32m1(
                    (const uint32_t*)row1, c1, vl);
                const vuint32m1_t mask = __riscv_vmv_v_x_u32m1(0xffu, vl);
                const vfloat32m1_t v00 = __riscv_vfcvt_f_xu_v_f32m1(
                    __riscv_vand_vv_u32m1(raw00, mask, vl), vl);
                const vfloat32m1_t v01 = __riscv_vfcvt_f_xu_v_f32m1(
                    __riscv_vand_vv_u32m1(raw01, mask, vl), vl);
                const vfloat32m1_t v10 = __riscv_vfcvt_f_xu_v_f32m1(
                    __riscv_vand_vv_u32m1(raw10, mask, vl), vl);
                const vfloat32m1_t v11 = __riscv_vfcvt_f_xu_v_f32m1(
                    __riscv_vand_vv_u32m1(raw11, mask, vl), vl);
                const vfloat32m1_t top = __riscv_vfadd_vv_f32m1(
                    __riscv_vfmul_vv_f32m1(v00, fx1, vl),
                    __riscv_vfmul_vv_f32m1(v01, fx, vl), vl);
                const vfloat32m1_t bot = __riscv_vfadd_vv_f32m1(
                    __riscv_vfmul_vv_f32m1(v10, fx1, vl),
                    __riscv_vfmul_vv_f32m1(v11, fx, vl), vl);
                vfloat32m1_t v = __riscv_vfadd_vv_f32m1(
                    __riscv_vfmul_vf_f32m1(top, 1.f - fy, vl),
                    __riscv_vfmul_vf_f32m1(bot, fy, vl), vl);
                v = __riscv_vfmul_vf_f32m1(v, 1.f / 255.f, vl);
                __riscv_vse32_v_f32m1(out + (size_t)c * (size_t)plane + dst_row + dx, v, vl);
            }
            dx += (int)vl;
        }
#endif
        for (; dx < new_w; ++dx) {
            const float fx = fxt[dx];
            const int x0 = x0t[dx];
            const int x1 = x1t[dx];
            for (int c = 0; c < 3; ++c) {
                const float v00 = (float)row0[(size_t)x0 * 3u + (size_t)c];
                const float v01 = (float)row0[(size_t)x1 * 3u + (size_t)c];
                const float v10 = (float)row1[(size_t)x0 * 3u + (size_t)c];
                const float v11 = (float)row1[(size_t)x1 * 3u + (size_t)c];
                const float v = (v00 * (1 - fx) + v01 * fx) * (1 - fy) +
                                (v10 * (1 - fx) + v11 * fx) * fy;
                out[(size_t)c * (size_t)plane + dst_row + (size_t)dx] = v / 255.f;
            }
        }
    }
}

}  // namespace riscv
}  // namespace yoloop
