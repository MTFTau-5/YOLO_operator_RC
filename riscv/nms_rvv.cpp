#include "yolo_rvv.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

#include "rvv_util.h"

namespace yoloop {
namespace riscv {

static inline float iou_scalar(const Detection& a, const float* x1, const float* y1,
                               const float* x2, const float* y2, const float* area, size_t j) {
    const float iw = std::max(std::min(a.x2, x2[j]) - std::max(a.x1, x1[j]), 0.f);
    const float ih = std::max(std::min(a.y2, y2[j]) - std::max(a.y1, y1[j]), 0.f);
    const float inter = iw * ih;
    const float ua = (a.x2 - a.x1) * (a.y2 - a.y1) + area[j] - inter;
    return ua > 0.f ? inter / ua : 0.f;
}

std::vector<int32_t> nms_indices(const std::vector<Detection>& dets, float nms_thr, int topk) {
    const size_t n0 = dets.size();
    if (n0 == 0) return {};

    std::vector<size_t> idx(n0);
    std::iota(idx.begin(), idx.end(), (size_t)0);
    auto cmp = [&](size_t a, size_t b) {
        const float sa = dets[a].score;
        const float sb = dets[b].score;
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

    std::vector<float> x1(n), y1(n), x2(n), y2(n), area(n);
    std::vector<int32_t> label(n);
    for (size_t k = 0; k < n; ++k) {
        const Detection& d = dets[idx[k]];
        x1[k] = d.x1;
        y1[k] = d.y1;
        x2[k] = d.x2;
        y2[k] = d.y2;
        area[k] = (d.x2 - d.x1) * (d.y2 - d.y1);
        label[k] = d.label;
    }

    std::vector<uint8_t> suppressed(n, 0);
    std::vector<int32_t> keep;
    keep.reserve(n);
    size_t removed = 0;
#ifdef __riscv_vector
    std::vector<uint32_t> hits(__riscv_vsetvlmax_e32m1());
#endif
    for (size_t i = 0; i < n; ++i) {
        if (suppressed[i]) continue;
        keep.push_back((int32_t)idx[i]);
        if (++removed == n) break;

        size_t j = i + 1;
#ifdef __riscv_vector
        while (j < n) {
            const size_t vl = __riscv_vsetvl_e32m1(n - j);
            const vint32m1_t labels = __riscv_vle32_v_i32m1(label.data() + j, vl);
            const vbool32_t same = __riscv_vmseq_vx_i32m1_b32(labels, label[i], vl);
            if (__riscv_vcpop_m_b32(same, vl) == 0) {
                j += vl;
                continue;
            }
            const vfloat32m1_t jx1 = __riscv_vle32_v_f32m1(x1.data() + j, vl);
            const vfloat32m1_t jy1 = __riscv_vle32_v_f32m1(y1.data() + j, vl);
            const vfloat32m1_t jx2 = __riscv_vle32_v_f32m1(x2.data() + j, vl);
            const vfloat32m1_t jy2 = __riscv_vle32_v_f32m1(y2.data() + j, vl);
            const vfloat32m1_t ix1 = __riscv_vfmv_v_f_f32m1(x1[i], vl);
            const vfloat32m1_t iy1 = __riscv_vfmv_v_f_f32m1(y1[i], vl);
            const vfloat32m1_t ix2 = __riscv_vfmv_v_f_f32m1(x2[i], vl);
            const vfloat32m1_t iy2 = __riscv_vfmv_v_f_f32m1(y2[i], vl);
            vfloat32m1_t iw = __riscv_vfsub_vv_f32m1(
                __riscv_vfmin_vv_f32m1(ix2, jx2, vl),
                __riscv_vfmax_vv_f32m1(ix1, jx1, vl), vl);
            vfloat32m1_t ih = __riscv_vfsub_vv_f32m1(
                __riscv_vfmin_vv_f32m1(iy2, jy2, vl),
                __riscv_vfmax_vv_f32m1(iy1, jy1, vl), vl);
            iw = __riscv_vfmax_vf_f32m1(iw, 0.f, vl);
            ih = __riscv_vfmax_vf_f32m1(ih, 0.f, vl);
            const vfloat32m1_t inter = __riscv_vfmul_vv_f32m1(iw, ih, vl);
            const vfloat32m1_t jarea = __riscv_vle32_v_f32m1(area.data() + j, vl);
            const vfloat32m1_t iarea = __riscv_vfmv_v_f_f32m1(area[i], vl);
            const vfloat32m1_t ua = __riscv_vfsub_vv_f32m1(
                __riscv_vfadd_vv_f32m1(jarea, iarea, vl), inter, vl);
            const vfloat32m1_t iou = __riscv_vfdiv_vv_f32m1(inter, ua, vl);
            vbool32_t hit = __riscv_vmfgt_vf_f32m1_b32(iou, nms_thr, vl);
            const vbool32_t valid = __riscv_vmfgt_vf_f32m1_b32(ua, 0.f, vl);
            hit = __riscv_vmand_mm_b32(hit, valid, vl);
            hit = __riscv_vmand_mm_b32(hit, same, vl);
            const size_t count = compact_indices(hit, vl, hits.data());
            for (size_t k = 0; k < count; ++k) {
                const size_t jj = j + hits[k];
                if (!suppressed[jj]) {
                    suppressed[jj] = 1;
                    ++removed;
                }
            }
            j += vl;
        }
#endif
        const Detection& di = dets[idx[i]];
        for (; j < n; ++j) {
            if (suppressed[j] || label[j] != label[i]) continue;
            if (iou_scalar(di, x1.data(), y1.data(), x2.data(), y2.data(), area.data(), j) >
                nms_thr) {
                suppressed[j] = 1;
                ++removed;
            }
        }
    }
    return keep;
}

std::vector<Detection> nms(std::vector<Detection> dets, float nms_thr, int topk) {
    const auto idx = nms_indices(dets, nms_thr, topk);
    std::vector<Detection> out;
    out.reserve(idx.size());
    for (const int32_t i : idx) out.push_back(dets[(size_t)i]);
    return out;
}

}  // namespace riscv
}  // namespace yoloop
