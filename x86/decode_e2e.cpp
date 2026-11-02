#include "decode_e2e.h"

namespace yoloop {
namespace x86 {

// N≈300 且无 NMS，标量即可，向量化无意义
std::vector<Detection> decode_e2e(const float* pred, int n, float conf_thr) {
    std::vector<Detection> out;
    const float* score = pred + (size_t)4 * n;
    const float* cls = pred + (size_t)5 * n;
    for (int j = 0; j < n; ++j) {
        if (score[j] < conf_thr) continue;
        Detection d;
        d.x1 = pred[j];                 d.y1 = pred[(size_t)n + j];
        d.x2 = pred[(size_t)2 * n + j]; d.y2 = pred[(size_t)3 * n + j];
        d.score = score[j];
        d.label = (int32_t)cls[j];
        out.push_back(d);
    }
    return out;
}

}  // namespace x86
}  // namespace yoloop
