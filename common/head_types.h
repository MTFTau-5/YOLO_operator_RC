#pragma once
#include <string>
#include <vector>

#include "yolo_types.h"

namespace yoloop {

enum class HeadType {
    AnchorRaw = 0,
    AfFused = 1,
    PoseFused = 2,
    AfRaw = 3,
    E2E = 4,
};

inline const char* head_type_name(HeadType h) {
    switch (h) {
        case HeadType::AnchorRaw: return "v5";
        case HeadType::AfFused: return "v8";
        case HeadType::PoseFused: return "v8-pose";
        case HeadType::AfRaw: return "v8-raw";
        case HeadType::E2E: return "v10";
    }
    return "unknown";
}


inline bool parse_head_type(const std::string& s, HeadType& out) {
    if (s == "v5") out = HeadType::AnchorRaw;
    else if (s == "v8" || s == "v9" || s == "v11" || s == "v8-fused")
        out = HeadType::AfFused;
    else if (s == "v8-pose" || s == "pose" || s == "v9-pose" || s == "v11-pose")
        out = HeadType::PoseFused;
    else if (s == "v8-raw" || s == "v9-raw" || s == "v11-raw" || s == "af-raw")
        out = HeadType::AfRaw;
    else if (s == "v10" || s == "e2e" || s == "end2end" || s == "v10-e2e")
        out = HeadType::E2E;
    else
        return false;
    return true;
}

inline int head_channels(HeadType h, int num_classes, int num_kpts, int reg_max) {
    switch (h) {
        case HeadType::AnchorRaw: return 5 + num_classes;
        case HeadType::AfFused: return 4 + num_classes;
        case HeadType::PoseFused: return 4 + num_classes + 3 * num_kpts;
        case HeadType::AfRaw: return 4 * reg_max + num_classes;
        case HeadType::E2E: return 6;
    }
    return 0;
}


inline AnchorTable build_point_table(const std::vector<LevelSpec>& levels) {
    AnchorTable tab;
    int total = 0;
    for (const auto& lv : levels) total += lv.grid_w * lv.grid_h;
    tab.gx.reserve(total); tab.gy.reserve(total); tab.stride.reserve(total);
    tab.aw.reserve(total); tab.ah.reserve(total);
    for (const auto& lv : levels) {
        for (int y = 0; y < lv.grid_h; ++y) {
            for (int x = 0; x < lv.grid_w; ++x) {
                tab.gx.push_back(x + 0.5f);
                tab.gy.push_back(y + 0.5f);
                tab.stride.push_back((float)lv.stride);
                tab.aw.push_back(1.f);
                tab.ah.push_back(1.f);
            }
        }
    }
    tab.n = total;
    return tab;
}

inline std::vector<LevelSpec> default_point_levels(int input_w = 640, int input_h = 640) {
    return {
        {8,  input_w / 8,  input_h / 8,  {}},
        {16, input_w / 16, input_h / 16, {}},
        {32, input_w / 32, input_h / 32, {}},
    };
}

}  // namespace yoloop
