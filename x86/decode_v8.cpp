#include "decode_v8.h"

#include <cstring>

namespace yoloop {
namespace x86 {

namespace {

// 逐 anchor：argmax 类别概率，过阈值则收集框；返回是否存活，存活时写出 d
inline bool scan_anchor(const float* pred, int num_classes, int n, int j, float conf_thr,
                        Detection& d) {
    const float* cls = pred + (size_t)4 * n + j;  // 各类别概率列（步长 n）
    int best_c = 0;
    float best_p = cls[0];
    for (int c = 1; c < num_classes; ++c) {
        float p = cls[(size_t)c * n];
        if (p > best_p) {
            best_p = p;
            best_c = c;
        }
    }
    if (best_p < conf_thr) return false;
    float cx = pred[j], cy = pred[(size_t)n + j];
    float w = pred[(size_t)2 * n + j], h = pred[(size_t)3 * n + j];
    d = {cx - w * 0.5f, cy - h * 0.5f, cx + w * 0.5f, cy + h * 0.5f, best_p, best_c};
    return true;
}

}  // namespace

std::vector<Detection> decode_v8(const float* pred, int num_classes, int n, float conf_thr) {
    std::vector<Detection> out;
    out.reserve(256);
    for (int j = 0; j < n; ++j) {
        Detection d;
        if (scan_anchor(pred, num_classes, n, j, conf_thr, d)) out.push_back(d);
    }
    return out;
}

void decode_v8_pose(const float* pred, int num_classes, int num_kpts, int n, float conf_thr,
                    std::vector<Detection>& dets, std::vector<float>& kpts) {
    dets.clear();
    kpts.clear();
    const size_t kpt_stride = (size_t)3 * num_kpts;
    const float* kpt_base = pred + (size_t)(4 + num_classes) * n;
    for (int j = 0; j < n; ++j) {
        Detection d;
        if (!scan_anchor(pred, num_classes, n, j, conf_thr, d)) continue;
        dets.push_back(d);
        // 关键点通道也是 channel-major：第 k 个点的 x/y/conf 分别在通道 3k/3k+1/3k+2
        size_t dst = kpts.size();
        kpts.resize(dst + kpt_stride);
        for (size_t c = 0; c < kpt_stride; ++c)
            kpts[dst + c] = kpt_base[c * (size_t)n + j];
    }
}

}  // namespace x86
}  // namespace yoloop
