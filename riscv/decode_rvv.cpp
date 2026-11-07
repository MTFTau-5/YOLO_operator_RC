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

static inline void decode_one(const float* pred, const AnchorTable& tab, int num_classes,
                              float conf_thr, int i, std::vector<Detection>& out) {
    const int n = tab.n;
    const float* cls = pred + (size_t)5 * n + i;
    int best = 0;
    float best_logit = cls[0];
    for (int c = 1; c < num_classes; ++c) {
        const float v = cls[(size_t)c * n];
        if (v > best_logit) {
            best_logit = v;
            best = c;
        }
    }

    const float score = sigmoid(pred[(size_t)4 * n + i]) * sigmoid(best_logit);
    if (score < conf_thr) return;
    const float x = (sigmoid(pred[i]) * 2.f - 0.5f + tab.gx[i]) * tab.stride[i];
    const float y = (sigmoid(pred[(size_t)n + i]) * 2.f - 0.5f + tab.gy[i]) * tab.stride[i];
    const float sw = sigmoid(pred[(size_t)2 * n + i]) * 2.f;
    const float sh = sigmoid(pred[(size_t)3 * n + i]) * 2.f;
    const float w = sw * sw * tab.aw[i];
    const float h = sh * sh * tab.ah[i];
    out.push_back({x - w * 0.5f, y - h * 0.5f, x + w * 0.5f, y + h * 0.5f,
                   score, best});
}

static void decode_range(const float* pred, const AnchorTable& tab, int num_classes,
                         float logit_thr, float conf_thr, int begin, int end,
                         std::vector<Detection>& out) {
    const float* obj = pred + (size_t)4 * tab.n;
    int i = begin;
#ifdef __riscv_vector
    std::vector<uint32_t> indices(__riscv_vsetvlmax_e32m1());
    while (i < end) {
        const size_t vl = __riscv_vsetvl_e32m1((size_t)(end - i));
        const vfloat32m1_t v = __riscv_vle32_v_f32m1(obj + i, vl);
        const vbool32_t keep = __riscv_vmfge_vf_f32m1_b32(v, logit_thr, vl);
        const size_t count = compact_indices(keep, vl, indices.data());
        for (size_t k = 0; k < count; ++k)
            decode_one(pred, tab, num_classes, conf_thr, i + (int)indices[k], out);
        i += (int)vl;
    }
#else
    for (; i < end; ++i) {
        if (obj[i] >= logit_thr) decode_one(pred, tab, num_classes, conf_thr, i, out);
    }
#endif
}

std::vector<Detection> decode(const float* pred, const AnchorTable& tab,
                              int num_classes, float conf_thr) {
    const float logit_thr = logit_threshold(conf_thr);
    const int n = tab.n;
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
                decode_range(pred, tab, num_classes, logit_thr, conf_thr, begin, end,
                             buckets[(size_t)tid]);
        }
        size_t total = 0;
        for (const auto& b : buckets) total += b.size();
        out.reserve(total);
        for (auto& b : buckets) out.insert(out.end(), b.begin(), b.end());
        return out;
    }
#endif

    decode_range(pred, tab, num_classes, logit_thr, conf_thr, 0, n, out);
    return out;
}

static inline bool scan_anchor(const float* pred, int num_classes, int n, int j,
                               float conf_thr, Detection& d) {
    const float* cls = pred + (size_t)4 * n + j;
    int best = 0;
    float best_p = cls[0];
    for (int c = 1; c < num_classes; ++c) {
        const float p = cls[(size_t)c * n];
        if (p > best_p) {
            best_p = p;
            best = c;
        }
    }
    if (best_p < conf_thr) return false;
    const float cx = pred[j];
    const float cy = pred[(size_t)n + j];
    const float w = pred[(size_t)2 * n + j];
    const float h = pred[(size_t)3 * n + j];
    d = {cx - w * 0.5f, cy - h * 0.5f, cx + w * 0.5f, cy + h * 0.5f, best_p, best};
    return true;
}

std::vector<Detection> decode_v8(const float* pred, int num_classes, int n, float conf_thr) {
    std::vector<Detection> out;
    out.reserve(256);
    int j = 0;
#ifdef __riscv_vector
    const float* cls = pred + (size_t)4 * n;
    std::vector<uint32_t> indices(__riscv_vsetvlmax_e32m1());
    while (j < n) {
        const size_t vl = __riscv_vsetvl_e32m1((size_t)(n - j));
        vfloat32m1_t best = __riscv_vle32_v_f32m1(cls + j, vl);
        for (int c = 1; c < num_classes; ++c) {
            const vfloat32m1_t v = __riscv_vle32_v_f32m1(cls + (size_t)c * n + j, vl);
            best = __riscv_vfmax_vv_f32m1(best, v, vl);
        }
        const vbool32_t keep = __riscv_vmfge_vf_f32m1_b32(best, conf_thr, vl);
        const size_t count = compact_indices(keep, vl, indices.data());
        for (size_t k = 0; k < count; ++k) {
            Detection d;
            if (scan_anchor(pred, num_classes, n, j + (int)indices[k], conf_thr, d))
                out.push_back(d);
        }
        j += (int)vl;
    }
#else
    for (; j < n; ++j) {
        Detection d;
        if (scan_anchor(pred, num_classes, n, j, conf_thr, d)) out.push_back(d);
    }
#endif
    return out;
}

void decode_v8_pose(const float* pred, int num_classes, int num_kpts, int n, float conf_thr,
                    std::vector<Detection>& dets, std::vector<float>& kpts) {
    dets.clear();
    kpts.clear();
    const size_t kpt_stride = (size_t)3 * num_kpts;
    const float* kpt_base = pred + (size_t)(4 + num_classes) * n;
    int j = 0;
#ifdef __riscv_vector
    const float* cls = pred + (size_t)4 * n;
    std::vector<uint32_t> indices(__riscv_vsetvlmax_e32m1());
    while (j < n) {
        const size_t vl = __riscv_vsetvl_e32m1((size_t)(n - j));
        vfloat32m1_t best = __riscv_vle32_v_f32m1(cls + j, vl);
        for (int c = 1; c < num_classes; ++c) {
            const vfloat32m1_t v = __riscv_vle32_v_f32m1(cls + (size_t)c * n + j, vl);
            best = __riscv_vfmax_vv_f32m1(best, v, vl);
        }
        const vbool32_t keep = __riscv_vmfge_vf_f32m1_b32(best, conf_thr, vl);
        const size_t count = compact_indices(keep, vl, indices.data());
        for (size_t k = 0; k < count; ++k) {
            Detection d;
            const int index = j + (int)indices[k];
            if (!scan_anchor(pred, num_classes, n, index, conf_thr, d)) continue;
            dets.push_back(d);
            for (size_t c = 0; c < kpt_stride; ++c)
                kpts.push_back(kpt_base[c * (size_t)n + index]);
        }
        j += (int)vl;
    }
#else
    for (; j < n; ++j) {
        Detection d;
        if (!scan_anchor(pred, num_classes, n, j, conf_thr, d)) continue;
        dets.push_back(d);
        for (size_t c = 0; c < kpt_stride; ++c)
            kpts.push_back(kpt_base[c * (size_t)n + j]);
    }
#endif
}

std::vector<Detection> decode_e2e(const float* pred, int n, float conf_thr) {
    std::vector<Detection> out;
    const float* score = pred + (size_t)4 * n;
    const float* cls = pred + (size_t)5 * n;
    for (int j = 0; j < n; ++j) {
        if (score[j] < conf_thr) continue;
        out.push_back({pred[j], pred[(size_t)n + j], pred[(size_t)2 * n + j],
                       pred[(size_t)3 * n + j], score[j], (int32_t)cls[j]});
    }
    return out;
}

}  // namespace riscv
}  // namespace yoloop
