#include "nms_simd.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace yoloop {
namespace x86 {

static inline float iou_scalar(const Detection& a, const float* x1, const float* y1,
                               const float* x2, const float* y2, const float* area, size_t j) {
    float iw = std::max(std::min(a.x2, x2[j]) - std::max(a.x1, x1[j]), 0.f);
    float ih = std::max(std::min(a.y2, y2[j]) - std::max(a.y1, y1[j]), 0.f);
    float inter = iw * ih;
    float ua = (a.x2 - a.x1) * (a.y2 - a.y1) + area[j] - inter;
    return ua > 0.f ? inter / ua : 0.f;
}

std::vector<Detection> nms(std::vector<Detection> dets, float nms_thr, int topk) {
    const size_t n0 = dets.size();
    if (n0 == 0) return {};

    // score 降序；并列按原索引升序（全序 ⇒ 等价 stable sort，与 ref 语义一致）
    std::vector<size_t> idx(n0);
    std::iota(idx.begin(), idx.end(), (size_t)0);
    auto cmp = [&](size_t a, size_t b) {
        float sa = dets[a].score, sb = dets[b].score;
        if (sa != sb) return sa > sb;
        return a < b;
    };
    if (topk > 0 && n0 > (size_t)topk) {
        std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(), cmp);
        idx.resize((size_t)topk);
    } else {
        std::sort(idx.begin(), idx.end(), cmp);
    }
    const size_t n = idx.size();
    const size_t n8 = (n + 7) & ~(size_t)7;  // SoA 尾部 padding 到 8 的倍数

    // SoA 布局；padding 槽位 label = -1，整块比较时天然失配，永远不会被抑制
    std::vector<float> x1(n8, 0.f), y1(n8, 0.f), x2(n8, 0.f), y2(n8, 0.f);
    std::vector<float> area(n8, 0.f), score(n8, 0.f);
    std::vector<int32_t> label(n8, -1);
    for (size_t k = 0; k < n; ++k) {
        const Detection& d = dets[idx[k]];
        x1[k] = d.x1; y1[k] = d.y1; x2[k] = d.x2; y2[k] = d.y2;
        area[k] = (d.x2 - d.x1) * (d.y2 - d.y1);
        score[k] = d.score;
        label[k] = d.label;
    }

    std::vector<uint8_t> suppressed(n8, 0);
    for (size_t k = n; k < n8; ++k) suppressed[k] = 1;

    std::vector<Detection> keep;
    keep.reserve(n);
    size_t removed = 0;  // kept ∪ suppressed 的框数；== n 时早停
    for (size_t i = 0; i < n; ++i) {
        if (suppressed[i]) continue;
        keep.push_back(dets[idx[i]]);
        if (++removed == n) break;  // 早停：剩余框均已无意义

        size_t j = i + 1;
#ifdef __AVX2__
        const __m256 bx1 = _mm256_set1_ps(x1[i]);
        const __m256 by1 = _mm256_set1_ps(y1[i]);
        const __m256 bx2 = _mm256_set1_ps(x2[i]);
        const __m256 by2 = _mm256_set1_ps(y2[i]);
        const __m256 barea = _mm256_set1_ps(area[i]);
        const __m256 bthr = _mm256_set1_ps(nms_thr);
        const __m256 zero = _mm256_setzero_ps();
        const __m256i blab = _mm256_set1_epi32(label[i]);
        for (; j + 8 <= n8; j += 8) {
            __m256i lj = _mm256_loadu_si256((const __m256i*)(label.data() + j));
            __m256i same = _mm256_cmpeq_epi32(lj, blab);
            if (_mm256_testz_si256(same, same)) continue;  // 整块异类，跳过
            __m256 iw = _mm256_max_ps(
                _mm256_sub_ps(_mm256_min_ps(bx2, _mm256_loadu_ps(x2.data() + j)),
                              _mm256_max_ps(bx1, _mm256_loadu_ps(x1.data() + j))), zero);
            __m256 ih = _mm256_max_ps(
                _mm256_sub_ps(_mm256_min_ps(by2, _mm256_loadu_ps(y2.data() + j)),
                              _mm256_max_ps(by1, _mm256_loadu_ps(y1.data() + j))), zero);
            __m256 inter = _mm256_mul_ps(iw, ih);
            __m256 ua = _mm256_sub_ps(
                _mm256_add_ps(barea, _mm256_loadu_ps(area.data() + j)), inter);
            __m256 iou = _mm256_div_ps(inter, ua);
            // ua <= 0 时 IoU 记 0（与 ref 一致），同时挡掉 0/0
            iou = _mm256_blendv_ps(zero, iou, _mm256_cmp_ps(ua, zero, _CMP_GT_OQ));
            __m256 hit = _mm256_and_ps(_mm256_cmp_ps(iou, bthr, _CMP_GT_OQ),
                                       _mm256_castsi256_ps(same));
            unsigned m = (unsigned)_mm256_movemask_ps(hit);
            while (m) {
                int lane = __builtin_ctz(m);
                m &= m - 1;
                size_t jj = j + (size_t)lane;
                if (!suppressed[jj]) { suppressed[jj] = 1; ++removed; }
            }
        }
#endif
        const Detection& di = dets[idx[i]];
        for (; j < n; ++j) {
            if (suppressed[j] || label[j] != label[i]) continue;
            if (iou_scalar(di, x1.data(), y1.data(), x2.data(), y2.data(), area.data(), j) > nms_thr) {
                suppressed[j] = 1;
                ++removed;
            }
        }
    }
    return keep;
}

}  // namespace x86
}  // namespace yoloop
