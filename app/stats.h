#pragma once
// 运行统计 —— PIPELINE.md §9 验收指标：P50/P99 端到端延迟、吞吐、丢帧率、CPU 占用。

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <vector>

namespace yoloop {

class Stats {
public:
    void add_latency(int64_t us) {
        lat_.push_back(us);
        if (lat_.size() > kWindow) lat_.pop_front();
        frames_++;
    }

    void tick(int64_t now_us_v) {  // 周期性调用（如每 2s），打印一行稳态指标
        if (last_tick_ == 0) {
            last_tick_ = now_us_v;
            frames_at_tick_ = frames_;
            return;
        }
        double dt = (now_us_v - last_tick_) * 1e-6;
        uint64_t nf = frames_ - frames_at_tick_;
        double fps = dt > 0 ? nf / dt : 0;
        printf("[stats] frames=%llu fps=%.1f latency P50=%.1fms P99=%.1fms dropped=%llu\n",
               (unsigned long long)frames_, fps, percentile(0.50) * 1e-3,
               percentile(0.99) * 1e-3, (unsigned long long)dropped_);
        last_tick_ = now_us_v;
        frames_at_tick_ = frames_;
    }

    void set_dropped(uint64_t d) { dropped_ = d; }
    uint64_t frames() const { return frames_; }

private:
    static constexpr size_t kWindow = 512;

    double percentile(double q) const {
        if (lat_.empty()) return 0;
        std::vector<int64_t> v(lat_.begin(), lat_.end());
        std::sort(v.begin(), v.end());
        size_t k = (size_t)(q * (v.size() - 1) + 0.5);
        return (double)v[k];
    }

    std::deque<int64_t> lat_;
    uint64_t frames_ = 0;
    uint64_t dropped_ = 0;
    int64_t last_tick_ = 0;
    uint64_t frames_at_tick_ = 0;
};

}  // namespace yoloop
