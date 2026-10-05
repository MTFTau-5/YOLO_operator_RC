#include "decode_avx2.h"

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

#ifdef __AVX2__

#if defined(__FMA__)
#define YOLOOP_FMADD(a, b, c) _mm256_fmadd_ps((a), (b), (c))
#define YOLOOP_FNMADD(a, b, c) _mm256_fnmadd_ps((a), (b), (c))
#else
#define YOLOOP_FMADD(a, b, c) _mm256_add_ps(_mm256_mul_ps((a), (b)), (c))
#define YOLOOP_FNMADD(a, b, c) _mm256_sub_ps((c), _mm256_mul_ps((a), (b)))
#endif

// exp 近似（avx_mathfun / ncnn sigmoid_ps 同源），相对误差 ~1e-7 量级
static inline __m256 exp256_ps(__m256 x) {
    x = _mm256_min_ps(x, _mm256_set1_ps(88.3762626647949f));
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.3762626647949f));

    __m256 fx = YOLOOP_FMADD(x, _mm256_set1_ps(1.44269504088896341f), _mm256_set1_ps(0.5f));
    fx = _mm256_floor_ps(fx);

    x = YOLOOP_FNMADD(fx, _mm256_set1_ps(0.693359375f), x);
    x = YOLOOP_FNMADD(fx, _mm256_set1_ps(-2.12194440e-4f), x);

    __m256 y = _mm256_set1_ps(1.9875691500E-4f);
    y = YOLOOP_FMADD(y, x, _mm256_set1_ps(1.3981999507E-3f));
    y = YOLOOP_FMADD(y, x, _mm256_set1_ps(8.3334519073E-3f));
    y = YOLOOP_FMADD(y, x, _mm256_set1_ps(4.1665795894E-2f));
    y = YOLOOP_FMADD(y, x, _mm256_set1_ps(1.6666665459E-1f));
    y = YOLOOP_FMADD(y, x, _mm256_set1_ps(5.0000001201E-1f));
    y = YOLOOP_FMADD(y, _mm256_mul_ps(x, x),
                     _mm256_add_ps(x, _mm256_set1_ps(1.f)));

    __m256i emm = _mm256_cvttps_epi32(fx);
    emm = _mm256_add_epi32(emm, _mm256_set1_epi32(0x7f));
    emm = _mm256_slli_epi32(emm, 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(emm));
}

static inline __m256 sigmoid256_ps(__m256 x) {
    const __m256 one = _mm256_set1_ps(1.f);
    return _mm256_div_ps(one, _mm256_add_ps(one, exp256_ps(_mm256_sub_ps(_mm256_setzero_ps(), x))));
}

#else  // !__AVX2__ —— 标量回退路径

static inline float sigmoid_fast(float x) { return 1.f / (1.f + expf(-x)); }

#endif

static inline void decode_one(const float* pred, const AnchorTable& tab, int num_classes,
                              float conf_thr, int i, std::vector<Detection>& out) {
    const int N = tab.n;
    const float* cls = pred + (size_t)5 * N + i;
    int best = 0;
    float best_logit = cls[0];
    for (int c = 1; c < num_classes; ++c) {
        float v = cls[(size_t)c * N];
        if (v > best_logit) { best_logit = v; best = c; }
    }

#ifdef __AVX2__
    __m256 v = _mm256_setr_ps(pred[(size_t)4 * N + i], best_logit,
                              pred[(size_t)0 * N + i], pred[(size_t)1 * N + i],
                              pred[(size_t)2 * N + i], pred[(size_t)3 * N + i],
                              0.f, 0.f);
    float s[8];
    _mm256_storeu_ps(s, sigmoid256_ps(v));
    float score = s[0] * s[1];
    if (score < conf_thr) return;
    float x = (s[2] * 2.f - 0.5f + tab.gx[i]) * tab.stride[i];
    float y = (s[3] * 2.f - 0.5f + tab.gy[i]) * tab.stride[i];
    float w = s[4] * 2.f; w = w * w * tab.aw[i];
    float h = s[5] * 2.f; h = h * h * tab.ah[i];
#else
    float obj = sigmoid_fast(pred[(size_t)4 * N + i]);
    float score = obj * sigmoid_fast(best_logit);
    if (score < conf_thr) return;
    float x = (sigmoid_fast(pred[(size_t)0 * N + i]) * 2.f - 0.5f + tab.gx[i]) * tab.stride[i];
    float y = (sigmoid_fast(pred[(size_t)1 * N + i]) * 2.f - 0.5f + tab.gy[i]) * tab.stride[i];
    float sw = sigmoid_fast(pred[(size_t)2 * N + i]) * 2.f;
    float sh = sigmoid_fast(pred[(size_t)3 * N + i]) * 2.f;
    float w = sw * sw * tab.aw[i];
    float h = sh * sh * tab.ah[i];
#endif

    Detection d;
    d.x1 = x - w * 0.5f; d.y1 = y - h * 0.5f;
    d.x2 = x + w * 0.5f; d.y2 = y + h * 0.5f;
    d.score = score; d.label = best;
    out.push_back(d);
}

static void decode_range(const float* pred, const AnchorTable& tab, int num_classes,
                         float logit_thr, float conf_thr, int begin, int end,
                         std::vector<Detection>& out) {
    const float* obj = pred + (size_t)4 * tab.n;
    int i = begin;
#ifdef __AVX2__
    const __m256 thr = _mm256_set1_ps(logit_thr);
    const int vend = begin + ((end - begin) & ~7);
    for (; i < vend; i += 8) {
        __m256 v = _mm256_loadu_ps(obj + i);
        // keep ⟺ obj_logit >= logit_thr（等价于 ref 的 sigmoid(obj) >= conf_thr）
        unsigned mask = (unsigned)_mm256_movemask_ps(_mm256_cmp_ps(v, thr, _CMP_GE_OQ));
        while (mask) {
            int lane = __builtin_ctz(mask);
            mask &= mask - 1;
            decode_one(pred, tab, num_classes, conf_thr, i + lane, out);
        }
    }
#endif
    for (; i < end; ++i) {
        if (obj[i] < logit_thr) continue;
        decode_one(pred, tab, num_classes, conf_thr, i, out);
    }
}

std::vector<Detection> decode(const float* pred, const AnchorTable& tab,
                              int num_classes, float conf_thr) {
    const float logit_thr = logit_threshold(conf_thr);
    const int N = tab.n;
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
                decode_range(pred, tab, num_classes, logit_thr, conf_thr, begin, end,
                             buckets[(size_t)tid]);
        }
        size_t total = 0;
        for (const auto& b : buckets) total += b.size();
        out.reserve(total);
        for (auto& b : buckets)
            out.insert(out.end(), b.begin(), b.end());
        return out;
    }
#endif

    decode_range(pred, tab, num_classes, logit_thr, conf_thr, 0, N, out);
    return out;
}

}  // namespace x86
}  // namespace yoloop
