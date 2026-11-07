#include "yolo_rvv.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "rvv_util.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace yoloop {
namespace riscv {

static inline float sigmoid(float x) { return 1.f / (1.f + expf(-x)); }

static inline float dfl_expect(const float* logits, size_t stride_n, int reg_max) {
    float m = logits[0];
    for (int k = 1; k < reg_max; ++k) m = std::max(m, logits[(size_t)k * stride_n]);
    float sum = 0.f;
    float acc = 0.f;
    for (int k = 0; k < reg_max; ++k) {
        const float p = expf(logits[(size_t)k * stride_n] - m);
        sum += p;
        acc += p * (float)k;
    }
    return acc / sum;
}

static inline void decode_one(const float* pred, const AnchorTable& pts, int num_classes,
                              int reg_max, float conf_thr, int i,
                              std::vector<Detection>& out) {
    const int n = pts.n;
    const float* cls = pred + (size_t)4 * reg_max * n + i;
    int best = 0;
    float best_logit = cls[0];
    for (int c = 1; c < num_classes; ++c) {
        const float v = cls[(size_t)c * n];
        if (v > best_logit) {
            best_logit = v;
            best = c;
        }
    }
    const float score = sigmoid(best_logit);
    if (score < conf_thr) return;

    const float* box = pred + i;
    const float dl = dfl_expect(box, (size_t)n, reg_max);
    const float dt = dfl_expect(box + (size_t)reg_max * n, (size_t)n, reg_max);
    const float dr = dfl_expect(box + (size_t)2 * reg_max * n, (size_t)n, reg_max);
    const float db = dfl_expect(box + (size_t)3 * reg_max * n, (size_t)n, reg_max);
    const float s = pts.stride[i];
    const float cx = pts.gx[i];
    const float cy = pts.gy[i];
    out.push_back({(cx - dl) * s, (cy - dt) * s, (cx + dr) * s, (cy + db) * s,
                   score, best});
}

static void decode_range(const float* pred, const AnchorTable& pts, int num_classes,
                         int reg_max, float logit_thr, float conf_thr, int begin, int end,
                         std::vector<Detection>& out) {
    const int n = pts.n;
    const float* cls = pred + (size_t)4 * reg_max * n;
    int i = begin;
#ifdef __riscv_vector
    std::vector<uint32_t> indices(__riscv_vsetvlmax_e32m1());
    while (i < end) {
        const size_t vl = __riscv_vsetvl_e32m1((size_t)(end - i));
        vfloat32m1_t best = __riscv_vle32_v_f32m1(cls + i, vl);
        for (int c = 1; c < num_classes; ++c) {
            const vfloat32m1_t v = __riscv_vle32_v_f32m1(cls + (size_t)c * n + i, vl);
            best = __riscv_vfmax_vv_f32m1(best, v, vl);
        }
        const vbool32_t keep = __riscv_vmfge_vf_f32m1_b32(best, logit_thr, vl);
        const size_t count = compact_indices(keep, vl, indices.data());
        for (size_t k = 0; k < count; ++k)
            decode_one(pred, pts, num_classes, reg_max, conf_thr, i + (int)indices[k], out);
        i += (int)vl;
    }
#else
    for (; i < end; ++i) {
        float best = cls[i];
        for (int c = 1; c < num_classes; ++c)
            best = std::max(best, cls[(size_t)c * n + i]);
        if (best >= logit_thr)
            decode_one(pred, pts, num_classes, reg_max, conf_thr, i, out);
    }
#endif
}

std::vector<Detection> decode_af_raw(const float* pred, const AnchorTable& pts,
                                     int num_classes, int reg_max, float conf_thr) {
    const float logit_thr = logit_threshold(conf_thr);
    const int n = pts.n;
    std::vector<Detection> out;

#ifdef _OPENMP
    const int nth = omp_get_max_threads();
    if (nth > 1 && n >= 4096) {
        std::vector<std::vector<Detection>> buckets((size_t)nth);
#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            const int nt = omp_get_num_threads();
            const size_t chunk = ((size_t)n + (size_t)nt - 1) / (size_t)nt;
            const int begin = (int)std::min(chunk * (size_t)tid, (size_t)n);
            const int end = (int)std::min((size_t)begin + chunk, (size_t)n);
            if (begin < end)
                decode_range(pred, pts, num_classes, reg_max, logit_thr, conf_thr, begin, end,
                             buckets[(size_t)tid]);
        }
        size_t total = 0;
        for (const auto& b : buckets) total += b.size();
        out.reserve(total);
        for (auto& b : buckets) out.insert(out.end(), b.begin(), b.end());
        return out;
    }
#endif

    decode_range(pred, pts, num_classes, reg_max, logit_thr, conf_thr, 0, n, out);
    return out;
}

}  // namespace riscv
}  // namespace yoloop
