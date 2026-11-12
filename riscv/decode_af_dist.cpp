#include "yolo_rvv.h"
#include "a100_dispatch.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "decode_util.h"

namespace yoloop {
namespace riscv {

using detail::sigmoid;

static inline void decode_one(const float* pred, const AnchorTable& pts, int num_classes,
                              float conf_thr, int i, std::vector<Detection>& out) {
    const int n = pts.n;
    const float* cls = pred + (size_t)4 * n + i;
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

    const float dl = pred[i];
    const float dt = pred[(size_t)n + i];
    const float dr = pred[(size_t)2 * n + i];
    const float db = pred[(size_t)3 * n + i];
    const float s = pts.stride[i];
    const float cx = pts.gx[i];
    const float cy = pts.gy[i];
    out.push_back({(cx - dl) * s, (cy - dt) * s, (cx + dr) * s, (cy + db) * s,
                   score, best});
}

static void decode_range(const float* pred, const AnchorTable& pts, int num_classes,
                         float logit_thr, float conf_thr, int begin, int end,
                         std::vector<Detection>& out) {
    const int n = pts.n;
    const float* cls = pred + (size_t)4 * n;
    detail::candidates(cls, n, num_classes, logit_thr, begin, end, [&](int i) {
        decode_one(pred, pts, num_classes, conf_thr, i, out);
    });
}

std::vector<Detection> decode_af_dist(const float* pred, const AnchorTable& pts,
                                      int num_classes, float conf_thr) {
    return detail::postprocess([&]() -> std::vector<Detection> {
        const float logit_thr = logit_threshold(conf_thr);
        const int n = pts.n;
        return detail::parallel_decode(n, [&](int begin, int end, std::vector<Detection>& out) {
            decode_range(pred, pts, num_classes, logit_thr, conf_thr, begin, end, out);
        });
    });
}

}  // namespace riscv
}  // namespace yoloop
