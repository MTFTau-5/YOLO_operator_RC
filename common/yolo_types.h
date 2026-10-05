#pragma once
// YOLO 算子库公共类型定义 —— 两平台共享
// 对应 README(1).md §0.2/§0.3 的导出约定：
//   - 模型输出为三个 head 的原始卷积输出（decode/NMS 在部署侧实现）
//   - pred 内存布局：channel-major [4 + 1 + num_classes, N]，N = 总 anchor 数
//       row 0..3 : tx, ty, tw, th
//       row 4    : objectness logit
//       row 5..  : num_classes 个类别 logit
//   - anchor 顺序：按 stride 8 -> 16 -> 32 依次排列；同一 level 内 row-major
//     遍历网格 (gy, gx)，anchor 索引变化最快
//   - decode 数学（YOLOv5 式）：
//       x = (sigmoid(tx) * 2 - 0.5 + gx) * stride
//       y = (sigmoid(ty) * 2 - 0.5 + gy) * stride
//       w = (sigmoid(tw) * 2)^2 * anchor_w
//       h = (sigmoid(th) * 2)^2 * anchor_h
//       obj = sigmoid(o)，score = obj * sigmoid(cls_max)
//   - 过滤：logit 域快筛（§0.3），o < logit(conf_thr) 直接跳过；
//           存活 anchor 再算最终 score 并按 conf_thr 过滤

#include <cmath>
#include <cstdint>
#include <vector>

namespace yoloop {

struct Detection {
    float x1, y1, x2, y2;  // 角点坐标（输入图像尺度）
    float score;
    int32_t label;
};

// Anchor 查找表，SoA 连续存储（README §8.4a 要点）
struct AnchorTable {
    std::vector<float> gx;      // 网格列
    std::vector<float> gy;      // 网格行
    std::vector<float> stride;  // 该 anchor 所在 level 的 stride
    std::vector<float> aw;      // anchor 宽（像素）
    std::vector<float> ah;      // anchor 高（像素）
    int n = 0;                  // anchor 总数
};

struct LevelSpec {
    int stride;
    int grid_w, grid_h;
    std::vector<std::pair<float, float>> anchors;  // (w, h)
};

// 默认 YOLOv5s P5 anchor（640 输入：80x80 / 40x40 / 20x20 网格）
inline std::vector<LevelSpec> default_levels(int input_w = 640, int input_h = 640) {
    return {
        {8,  input_w / 8,  input_h / 8,  {{10.f, 13.f}, {16.f, 30.f}, {33.f, 23.f}}},
        {16, input_w / 16, input_h / 16, {{30.f, 61.f}, {62.f, 45.f}, {59.f, 119.f}}},
        {32, input_w / 32, input_h / 32, {{116.f, 90.f}, {156.f, 198.f}, {373.f, 326.f}}},
    };
}

inline AnchorTable build_anchor_table(const std::vector<LevelSpec>& levels) {
    AnchorTable tab;
    int total = 0;
    for (const auto& lv : levels) total += lv.grid_w * lv.grid_h * (int)lv.anchors.size();
    tab.gx.reserve(total); tab.gy.reserve(total); tab.stride.reserve(total);
    tab.aw.reserve(total); tab.ah.reserve(total);
    for (const auto& lv : levels) {
        for (int y = 0; y < lv.grid_h; ++y) {
            for (int x = 0; x < lv.grid_w; ++x) {
                for (const auto& a : lv.anchors) {
                    tab.gx.push_back((float)x);
                    tab.gy.push_back((float)y);
                    tab.stride.push_back((float)lv.stride);
                    tab.aw.push_back(a.first);
                    tab.ah.push_back(a.second);
                }
            }
        }
    }
    tab.n = total;
    return tab;
}

// README §0.3：sigmoid 单调递增，阈值比较在 logit 域完成
inline float logit_threshold(float conf_thr) {
    return logf(conf_thr / (1.f - conf_thr));
}

}  // namespace yoloop
