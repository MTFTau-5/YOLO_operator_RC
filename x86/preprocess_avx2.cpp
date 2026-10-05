#include "preprocess_avx2.h"

#include <algorithm>
#include <cmath>
#include <vector>

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace yoloop {
namespace x86 {

void letterbox_normalize(const uint8_t* src, int src_w, int src_h,
                         float* out, int dst_w, int dst_h) {
    float scale = std::min((float)dst_w / src_w, (float)dst_h / src_h);
    int new_w = (int)roundf(src_w * scale);
    int new_h = (int)roundf(src_h * scale);
    int pad_x = (dst_w - new_w) / 2;
    int pad_y = (dst_h - new_h) / 2;
    const float pad_val = 114.f / 255.f;
    const int plane = dst_w * dst_h;

    // 先铺 padding 底色（图像区随后覆盖）
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < 3 * plane; ++i) out[i] = pad_val;

    // 列查找表：与 ref 逐像素计算的 sx/x0/fx/x1 完全同值
    std::vector<int> x0t(new_w), x1t(new_w);
    std::vector<float> fxt(new_w);
    for (int dx = 0; dx < new_w; ++dx) {
        float sx = (dx + 0.5f) / scale - 0.5f;
        int x0 = (int)floorf(sx);
        fxt[dx] = sx - x0;
        int x1 = x0 + 1;
        x0t[dx] = std::max(0, std::min(x0, src_w - 1));
        x1t[dx] = std::max(0, std::min(x1, src_w - 1));
    }
#ifdef __AVX2__
    std::vector<int32_t> x0off(new_w), x1off(new_w);
    for (int dx = 0; dx < new_w; ++dx) {
        x0off[dx] = x0t[dx] * 3;
        x1off[dx] = x1t[dx] * 3;
    }
    const size_t total_bytes = (size_t)src_w * (size_t)src_h * 3u;
#endif

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int dy = 0; dy < new_h; ++dy) {
        float sy = (dy + 0.5f) / scale - 0.5f;
        int y0 = (int)floorf(sy);
        float fy = sy - y0;
        int y1 = y0 + 1;
        y0 = std::max(0, std::min(y0, src_h - 1));
        y1 = std::max(0, std::min(y1, src_h - 1));
        const uint8_t* row0 = src + (size_t)y0 * (size_t)src_w * 3u;
        const uint8_t* row1 = src + (size_t)y1 * (size_t)src_w * 3u;
        const size_t dst_row = (size_t)(pad_y + dy) * (size_t)dst_w + (size_t)pad_x;

        int dx = 0;
#ifdef __AVX2__
        const __m256 one = _mm256_set1_ps(1.f);
        const __m256 v255 = _mm256_set1_ps(255.f);
        const __m256 fyv = _mm256_set1_ps(fy);
        const __m256 fy1 = _mm256_set1_ps(1.f - fy);
        const __m256i byte_mask = _mm256_set1_epi32(0xFF);
        for (; dx + 8 <= new_w; dx += 8) {
            // gather 每次读 4 字节：本块最大读取地址为 row1 + x1t[dx+7]*3 + 2，
            // 越界风险时整行余量回退标量
            size_t max_off = (size_t)((size_t)y1 * (size_t)src_w + (size_t)x1t[dx + 7]) * 3u + 2u;
            if (max_off + 4u > total_bytes) break;
            __m256 fx = _mm256_loadu_ps(fxt.data() + dx);
            __m256 fx1 = _mm256_sub_ps(one, fx);
            __m256i ox0 = _mm256_loadu_si256((const __m256i*)(x0off.data() + dx));
            __m256i ox1 = _mm256_loadu_si256((const __m256i*)(x1off.data() + dx));
            for (int c = 0; c < 3; ++c) {
                __m256i ci = _mm256_set1_epi32(c);
                __m256 v00 = _mm256_cvtepi32_ps(_mm256_and_si256(
                    _mm256_i32gather_epi32((const int*)row0, _mm256_add_epi32(ox0, ci), 1), byte_mask));
                __m256 v01 = _mm256_cvtepi32_ps(_mm256_and_si256(
                    _mm256_i32gather_epi32((const int*)row0, _mm256_add_epi32(ox1, ci), 1), byte_mask));
                __m256 v10 = _mm256_cvtepi32_ps(_mm256_and_si256(
                    _mm256_i32gather_epi32((const int*)row1, _mm256_add_epi32(ox0, ci), 1), byte_mask));
                __m256 v11 = _mm256_cvtepi32_ps(_mm256_and_si256(
                    _mm256_i32gather_epi32((const int*)row1, _mm256_add_epi32(ox1, ci), 1), byte_mask));
                __m256 top = _mm256_add_ps(_mm256_mul_ps(v00, fx1), _mm256_mul_ps(v01, fx));
                __m256 bot = _mm256_add_ps(_mm256_mul_ps(v10, fx1), _mm256_mul_ps(v11, fx));
                __m256 v = _mm256_add_ps(_mm256_mul_ps(top, fy1), _mm256_mul_ps(bot, fyv));
                v = _mm256_div_ps(v, v255);
                _mm256_storeu_ps(out + (size_t)c * (size_t)plane + dst_row + (size_t)dx, v);
            }
        }
#endif
        for (; dx < new_w; ++dx) {
            float fx = fxt[dx];
            int x0 = x0t[dx], x1 = x1t[dx];
            for (int c = 0; c < 3; ++c) {
                float v00 = (float)row0[(size_t)x0 * 3u + (size_t)c];
                float v01 = (float)row0[(size_t)x1 * 3u + (size_t)c];
                float v10 = (float)row1[(size_t)x0 * 3u + (size_t)c];
                float v11 = (float)row1[(size_t)x1 * 3u + (size_t)c];
                float v = (v00 * (1 - fx) + v01 * fx) * (1 - fy) +
                          (v10 * (1 - fx) + v11 * fx) * fy;
                out[(size_t)c * (size_t)plane + dst_row + (size_t)dx] = v / 255.f;
            }
        }
    }
}

}  // namespace x86
}  // namespace yoloop
