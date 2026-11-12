#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "yolo_rvv.h"
#include "rvv_util.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace yoloop::riscv::detail {

inline float sigmoid(float x) { return 1.f / (1.f + expf(-x)); }

// Contiguous partitions and ordered merging preserve the original anchor order.
template <class DecodeRange>
std::vector<Detection> parallel_decode(int n, DecodeRange range) {
    std::vector<Detection> out;
#if defined(_OPENMP) && !defined(YOLOOP_RISCV_A100)
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
            if (begin < end) range(begin, end, buckets[(size_t)tid]);
        }
        size_t total = 0;
        for (const auto& b : buckets) total += b.size();
        out.reserve(total);
        for (const auto& b : buckets) out.insert(out.end(), b.begin(), b.end());
        return out;
    }
#endif
    range(0, n, out);
    return out;
}

// Class logits/probabilities are channel-major; the callback handles survivors.
template <class Emit>
void candidates(const float* cls, int n, int num_classes, float threshold,
                int begin, int end, Emit emit) {
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
        const vbool32_t keep = __riscv_vmfge_vf_f32m1_b32(best, threshold, vl);
        const size_t count = compact_indices(keep, vl, indices.data());
        for (size_t k = 0; k < count; ++k) emit(i + (int)indices[k]);
        i += (int)vl;
    }
#else
    for (; i < end; ++i) {
        float best = cls[i];
        for (int c = 1; c < num_classes; ++c)
            best = std::max(best, cls[(size_t)c * n + i]);
        if (best >= threshold) emit(i);
    }
#endif
}

}  // namespace yoloop::riscv::detail
