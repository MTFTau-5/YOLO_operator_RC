#pragma once
// 跟踪器 —— 从 RK3588-AntiUAV-Tracker 提取的卡尔曼逻辑 C++ 移植：
//   BoxKalman  : tracker/run.py:39-70（8D 等速 [cx,cy,w,h,vx,vy,vw,vh]，dt=1 帧）
//   关联/状态机: tracker/track3.py:102-168（YoloKalman：尺寸记忆过滤、最近中心
//                关联、misses 自适应门限、coast_frames 后 LOST）
//   clamp_bbox : tracker/tracker_core.py:28-36
// 固定尺寸手写矩阵运算，零第三方依赖，与仓库 x86 风格一致。
// 数值口径与 Python float64 对齐（内部全部 double）。

#include <memory>
#include <vector>

#include "yolo_types.h"

namespace yoloop {

// ---------------------------------------------------------------- BoxKalman
// 与 run.py 完全相同的初值与更新数学（Joseph form）。
// 内部状态永不做图像裁剪，出视野运动可持续外推。
class BoxKalman {
public:
    explicit BoxKalman(const float box_xywh[4]);  // xywh 初始化

    void predict();                       // 前推一步
    void update(const float box_xywh[4]); // 量测修正
    void box(float out_xywh[4]) const;    // [x,y,w,h]，边长下限 3

    // KF 速度分量（cx,cy,w,h 的导数），供控制端前馈
    void velocity(float out_v[4]) const {
        for (int i = 0; i < 4; ++i) out_v[i] = (float)x_[4 + i];
    }

    const double* state() const { return x_; }  // 测试/调试用

private:
    double x_[8];   // 状态
    double P_[64];  // 协方差（行主序 8x8）
};

// ------------------------------------------------------------- clamp_bbox
// 移植 tracker_core.py：先截断为 int（同 np.int32 cast），再 min_side=3 回推。
void clamp_bbox(float box_xywh[4], int img_w, int img_h);

// ---------------------------------------------------------------- Tracker
enum class TrackState { SEARCH, TRACK, COAST, LOST };

struct Track {
    float box[4] = {0, 0, 0, 0};  // xywh
    float vel[4] = {0, 0, 0, 0};  // vx,vy,vw,vh
    float score = 0.f;
    int hits = 0;                 // 连续成功关联次数
    int misses = 0;               // 连续未关联次数
    TrackState state = TrackState::SEARCH;
    const char* event = "";       // INITIAL_LOCK / REACQUIRED / LOST / ""
};

struct TrackerParams {
    float association_gate_px = 120.f;    // track3.py --association-gate-px
    float association_gate_scale = 2.f;   // --association-gate-scale
    float size_ratio_limit = 3.f;         // --size-ratio-limit
    int size_memory_frames = 120;         // --size-memory-frames
    int coast_frames = 5;                 // --coast-frames
    int max_misses = 15;                  // --max-misses（门限自适应系数）
};

// 单目标跟踪（YoloKalman 语义）。每帧用法（PIPELINE.md §4）：
//   const Track* t = tracker.predict();        // KF 前推（无轨迹返回 nullptr）
//   t = tracker.update(kept, img_w, img_h, frame_idx);  // 关联/修正/丢失计数
// GPU 出错帧：predict() 后 update({}, ...) 即纯外推一帧（§6）。
class Tracker {
public:
    explicit Tracker(TrackerParams p = {}) : p_(p) {}
    Tracker(Tracker&&) = default;
    Tracker& operator=(Tracker&&) = default;

    const Track* predict();
    const Track* update(const std::vector<Detection>& dets, int img_w, int img_h,
                        int64_t frame_idx);

    const Track& track() const { return trk_; }
    const BoxKalman* kalman() const { return kal_.get(); }  // 测试/调试用
    // 本帧 update 命中的检测在传入 dets 里的下标（-1 = 未命中/coast/丢失），
    // 供 pose 场景取回该框的关键点；下次 update 前有效。
    int last_match() const { return last_match_; }

private:
    TrackerParams p_;
    std::unique_ptr<BoxKalman> kal_;
    Track trk_;
    bool ever_ = false;
    bool have_last_ = false;
    float last_box_[4] = {0, 0, 0, 0};
    int64_t last_frame_ = 0;
    float pred_box_[4] = {0, 0, 0, 0};  // predict() 缓存，update() 关联用
    int last_match_ = -1;

    Tracker(const Tracker&) = delete;
    Tracker& operator=(const Tracker&) = delete;
};

}  // namespace yoloop
