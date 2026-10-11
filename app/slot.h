#pragma once
// 槽位 —— PIPELINE.md §2/§7：总数写死为 3，在途帧恒 ≤ 3。
// 槽位状态由所在队列表示（q_free / q_gpu / q_back），无独立状态字段。
// 不变式：1) 一个槽任意时刻只在一个队列里；2) img/blob 只有线程A写、pred 只有线程B写；
//         3) 槽总数恒为 3。

#include <chrono>
#include <cstdint>
#include <vector>

#include "yolo_types.h"

namespace yoloop {

constexpr int NUM_SLOTS = 3;  // §7：延迟上界 = NUM_SLOTS × GPU 段耗时，写死不做配置项

struct Slot {
    int id = -1;

    // 采集帧（线程A写）
    std::vector<uint8_t> img;  // HWC u8，容量 = max_w*max_h*3
    int img_w = 0;
    int img_h = 0;

    // 网络输入 blob（线程A写）：CHW f32，in_w*in_h*3
    std::vector<float> blob;

    // 网络输出（线程B写）：channel-major [4+1+num_classes, N]
    std::vector<float> pred;

    // decode/NMS 结果（线程A写）
    std::vector<Detection> dets;
    std::vector<Detection> kept;
    std::vector<float> kpts;  // pose：与 kept 平行，每框 3*num_kpts 个 float

    int64_t t_capture = 0;  // 采集时间戳(us)，端到端延迟统计
    bool error = false;     // GPU 推理失败标记（§6：A 丢弃该帧，KF 纯 predict）
};

inline int64_t now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

}  // namespace yoloop
