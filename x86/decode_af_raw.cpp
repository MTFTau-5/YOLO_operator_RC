#include "decode_af_raw.h"

#include <algorithm>
#include <cmath>
#include <vector>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace yoloop {
namespace x86 {

static inline float sigmoid_f(float x) { return 1.f / (1.f + expf(-x)); }

// DFL：reg_max 个 logit 的 softmax 期望 Σk·p(k)。减 max 防溢出（softmax 平移不变）。
// 存活点稀疏（~1% 量级），标量即可，向量化无收益。
static inline float dfl_expect(const float* logits, size_t stride_n, int reg_max) {
    float m = logits[0];
    for (int k = 1; k < reg_max; ++k) m = std::max(m, logits[(size_t)k * stride_n]);
    float sum = 0.f, acc = 0.f;
    for (int k = 0; k < reg_max; ++k) {
        float p = expf(logits[(size_t)k * stride_n] - m);
        sum += p;
        acc += p * (float)k;
    }
    return acc / sum;
}

static inline void decode_one(const float* pred, const AnchorTable& pts, int num_classes,
                              int reg_max, float conf_thr, int i,
                              std::vector<Detection>& out) {
    const int N = pts.n;
    const float* cls = pred + (size_t)4 * reg_max * N + i;
    int best = 0;
    float best_logit = cls[0];
    for (int c = 1; c < num_classes; ++c) {
        float v = cls[(size_t)c * N];
        if (v > best_logit) { best_logit = v; best = c; }
    }
    float score = sigmoid_f(best_logit);
    if (score < conf_thr) return;

    const float* box = pred + i;  // 4 边 × reg_max 通道，通道步长 N
    float dl = dfl_expect(box + (size_t)0 * reg_max * N, (size_t)N, reg_max);
    float dt = dfl_expect(box + (size_t)1 * reg_max * N, (size_t)N, reg_max);
    float dr = dfl_expect(box + (size_t)2 * reg_max * N, (size_t)N, reg_max);
    float db = dfl_expect(box + (size_t)3 * reg_max * N, (size_t)N, reg_max);
    float s = pts.stride[i];
    float cx = pts.gx[i], cy = pts.gy[i];  // 网格中心

    Detection d;
    d.x1 = (cx - dl) * s; d.y1 = (cy - dt) * s;
    d.x2 = (cx + dr) * s; d.y2 = (cy + db) * s;
    d.score = score; d.label = best;
    out.push_back(d);
}

static void decode_range(const float* pred, const AnchorTable& pts, int num_classes,
                         int reg_max, float logit_thr, float conf_thr, int begin, int end,
                         std::vector<Detection>& out) {
    const int N = pts.n;
    const float* cls = pred + (size_t)4 * reg_max * N;
    int i = begin;
#ifdef __AVX2__
    const __m256 thr = _mm256_set1_ps(logit_thr);
    const int vend = begin + ((end - begin) & ~7);
    for (; i < vend; i += 8) {
        // 逐类取 max 类别 logit；keep ⟺ max_logit >= logit_thr
        // （sigmoid 单调递增，等价于 ref 的 sigmoid(max_logit) >= conf_thr）
        __m256 m = _mm256_loadu_ps(cls + i);
        for (int c = 1; c < num_classes; ++c)
            m = _mm256_max_ps(m, _mm256_loadu_ps(cls + (size_t)c * N + i));
        unsigned mask = (unsigned)_mm256_movemask_ps(_mm256_cmp_ps(m, thr, _CMP_GE_OQ));
        while (mask) {
            int lane = __builtin_ctz(mask);
            mask &= mask - 1;
            decode_one(pred, pts, num_classes, reg_max, conf_thr, i + lane, out);
        }
    }
#endif
    for (; i < end; ++i) {
        float m = cls[i];
        for (int c = 1; c < num_classes; ++c)
            m = std::max(m, cls[(size_t)c * N + i]);
        if (m < logit_thr) continue;
        decode_one(pred, pts, num_classes, reg_max, conf_thr, i, out);
    }
}

std::vector<Detection> decode_af_raw(const float* pred, const AnchorTable& pts,
                                     int num_classes, int reg_max, float conf_thr) {
    const float logit_thr = logit_threshold(conf_thr);
    const int N = pts.n;
    std::vector<Detection> out;

#ifdef _OPENMP
    const int nth = omp_get_max_threads();
    if (nth > 1 && N >= 4096) {
        // 线程局部收集，最后合并（输出顺序不保证与 ref 一致，测试按集合比较）
        std::vector<std::vector<Detection>> buckets((size_t)nth);
#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt = omp_get_num_threads();
            size_t chunk = ((size_t)N + (size_t)nt - 1) / (size_t)nt;
            int begin = (int)std::min(chunk * (size_t)tid, (size_t)N);
            int end = (int)std::min((size_t)begin + chunk, (size_t)N);
            if (begin < end)
                decode_range(pred, pts, num_classes, reg_max, logit_thr, conf_thr, begin,
                             end, buckets[(size_t)tid]);
        }
        size_t total = 0;
        for (const auto& b : buckets) total += b.size();
        out.reserve(total);
        for (auto& b : buckets)
            out.insert(out.end(), b.begin(), b.end());
        return out;
    }
#endif

    decode_range(pred, pts, num_classes, reg_max, logit_thr, conf_thr, 0, N, out);
    return out;
}

}  // namespace x86
}  // namespace yoloop
