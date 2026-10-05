// 标量参考实现 —— 与 tools/gen_test_vectors.py 的 numpy 实现严格同数学
#include "yolo_ref.h"

#include <algorithm>
#include <cmath>

namespace yoloop {
namespace ref {

static inline float sigmoid(float x) { return 1.f / (1.f + expf(-x)); }

std::vector<Detection> decode(const float* pred, const AnchorTable& tab,
                              int num_classes, float conf_thr) {
    const int C = 4 + 1 + num_classes;
    const int N = tab.n;
    std::vector<Detection> out;
    for (int i = 0; i < N; ++i) {
        float obj = sigmoid(pred[(size_t)4 * N + i]);
        if (obj < conf_thr) continue;

        int best = 0;
        float best_logit = pred[(size_t)5 * N + i];
        for (int c = 1; c < num_classes; ++c) {
            float v = pred[(size_t)(5 + c) * N + i];
            if (v > best_logit) { best_logit = v; best = c; }
        }
        float score = obj * sigmoid(best_logit);
        if (score < conf_thr) continue;

        float x = (sigmoid(pred[(size_t)0 * N + i]) * 2.f - 0.5f + tab.gx[i]) * tab.stride[i];
        float y = (sigmoid(pred[(size_t)1 * N + i]) * 2.f - 0.5f + tab.gy[i]) * tab.stride[i];
        float w = powf(sigmoid(pred[(size_t)2 * N + i]) * 2.f, 2) * tab.aw[i];
        float h = powf(sigmoid(pred[(size_t)3 * N + i]) * 2.f, 2) * tab.ah[i];

        Detection d;
        d.x1 = x - w * 0.5f; d.y1 = y - h * 0.5f;
        d.x2 = x + w * 0.5f; d.y2 = y + h * 0.5f;
        d.score = score; d.label = best;
        out.push_back(d);
    }
    (void)C;
    return out;
}

static inline float iou(const Detection& a, const Detection& b) {
    float iw = std::max(std::min(a.x2, b.x2) - std::max(a.x1, b.x1), 0.f);
    float ih = std::max(std::min(a.y2, b.y2) - std::max(a.y1, b.y1), 0.f);
    float inter = iw * ih;
    float ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
    return ua > 0.f ? inter / ua : 0.f;
}

std::vector<Detection> nms(std::vector<Detection> dets, float nms_thr) {
    std::stable_sort(dets.begin(), dets.end(),
                     [](const Detection& a, const Detection& b) { return a.score > b.score; });
    std::vector<Detection> keep;
    std::vector<char> suppressed(dets.size(), 0);
    for (size_t i = 0; i < dets.size(); ++i) {
        if (suppressed[i]) continue;
        keep.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (suppressed[j]) continue;
            if (dets[j].label != dets[i].label) continue;
            if (iou(dets[i], dets[j]) > nms_thr) suppressed[j] = 1;
        }
    }
    return keep;
}

void letterbox_normalize(const uint8_t* src, int src_w, int src_h,
                         float* out, int dst_w, int dst_h) {
    float scale = std::min((float)dst_w / src_w, (float)dst_h / src_h);
    int new_w = (int)roundf(src_w * scale);
    int new_h = (int)roundf(src_h * scale);
    int pad_x = (dst_w - new_w) / 2;
    int pad_y = (dst_h - new_h) / 2;
    const float pad_val = 114.f / 255.f;

    const int plane = dst_w * dst_h;
    for (int c = 0; c < 3; ++c)
        for (int i = 0; i < plane; ++i) out[c * plane + i] = pad_val;

    for (int dy = 0; dy < new_h; ++dy) {
        float sy = (dy + 0.5f) / scale - 0.5f;
        int y0 = (int)floorf(sy);
        float fy = sy - y0;
        int y1 = y0 + 1;
        y0 = std::max(0, std::min(y0, src_h - 1));
        y1 = std::max(0, std::min(y1, src_h - 1));
        for (int dx = 0; dx < new_w; ++dx) {
            float sx = (dx + 0.5f) / scale - 0.5f;
            int x0 = (int)floorf(sx);
            float fx = sx - x0;
            int x1 = x0 + 1;
            x0 = std::max(0, std::min(x0, src_w - 1));
            x1 = std::max(0, std::min(x1, src_w - 1));
            for (int c = 0; c < 3; ++c) {
                float v00 = src[(size_t)(y0 * src_w + x0) * 3 + c];
                float v01 = src[(size_t)(y0 * src_w + x1) * 3 + c];
                float v10 = src[(size_t)(y1 * src_w + x0) * 3 + c];
                float v11 = src[(size_t)(y1 * src_w + x1) * 3 + c];
                float v = (v00 * (1 - fx) + v01 * fx) * (1 - fy) +
                          (v10 * (1 - fx) + v11 * fx) * fy;
                out[(size_t)c * plane + (pad_y + dy) * dst_w + (pad_x + dx)] = v / 255.f;
            }
        }
    }
}

}  // namespace ref
}  // namespace yoloop
