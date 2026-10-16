#include "tracker.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace yoloop {

namespace {

// run.py BoxKalman 常量
constexpr double kP0[8] = {25., 25., 16., 16., 100., 100., 16., 16.};
constexpr double kQ[8] = {1., 1., .2, .2, 4., 4., .1, .1};
constexpr double kR[4] = {4., 4., 16., 16.};

inline void center_of(const float b[4], float& cx, float& cy) {
    cx = b[0] + b[2] * 0.5f;
    cy = b[1] + b[3] * 0.5f;
}

// 4x4 逆（Gauss-Jordan，部分主元）。S 对称正定，不会奇异。
bool inv4(const double a[16], double inv[16]) {
    double m[4][8];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) m[i][j] = a[i * 4 + j];
        for (int j = 0; j < 4; ++j) m[i][4 + j] = (i == j) ? 1. : 0.;
    }
    for (int col = 0; col < 4; ++col) {
        int piv = col;
        for (int r = col + 1; r < 4; ++r)
            if (fabs(m[r][col]) > fabs(m[piv][col])) piv = r;
        if (fabs(m[piv][col]) < 1e-12) return false;
        if (piv != col)
            for (int j = 0; j < 8; ++j) std::swap(m[piv][j], m[col][j]);
        double d = m[col][col];
        for (int j = 0; j < 8; ++j) m[col][j] /= d;
        for (int r = 0; r < 4; ++r) {
            if (r == col) continue;
            double f = m[r][col];
            for (int j = 0; j < 8; ++j) m[r][j] -= f * m[col][j];
        }
    }
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) inv[i * 4 + j] = m[i][4 + j];
    return true;
}

}  // namespace

// ---------------------------------------------------------------- BoxKalman
BoxKalman::BoxKalman(const float box_xywh[4]) {
    float fcx, fcy;
    center_of(box_xywh, fcx, fcy);
    x_[0] = fcx;
    x_[1] = fcy;
    x_[2] = box_xywh[2];
    x_[3] = box_xywh[3];
    for (int i = 4; i < 8; ++i) x_[i] = 0.;
    std::memset(P_, 0, sizeof(P_));
    for (int i = 0; i < 8; ++i) P_[i * 8 + i] = kP0[i];
}

void BoxKalman::box(float out[4]) const {
    double w = std::max(3., x_[2]);
    double h = std::max(3., x_[3]);
    out[0] = (float)(x_[0] - w * 0.5);
    out[1] = (float)(x_[1] - h * 0.5);
    out[2] = (float)w;
    out[3] = (float)h;
}

void BoxKalman::predict() {
    // x = F x, F[:4,4:] = I
    for (int i = 0; i < 4; ++i) x_[i] += x_[4 + i];
    // P = F P F' + Q。FP[i][j] = P[i][j] + (i<4 ? P[i+4][j] : 0)
    double fp[64];
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            fp[i * 8 + j] = P_[i * 8 + j] + (i < 4 ? P_[(i + 4) * 8 + j] : 0.);
    // FPF'[i][j] = FP[i][j] + (j<4 ? FP[i][j+4] : 0)
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            P_[i * 8 + j] = fp[i * 8 + j] + (j < 4 ? fp[i * 8 + j + 4] : 0.);
    for (int i = 0; i < 8; ++i) P_[i * 8 + i] += kQ[i];
}

void BoxKalman::update(const float b[4]) {
    // z = [cx, cy, w, h]，residual = z - H x
    double z[4];
    {
        float fcx, fcy;
        center_of(b, fcx, fcy);
        z[0] = fcx;
        z[1] = fcy;
        z[2] = b[2];
        z[3] = b[3];
    }
    double res[4];
    for (int i = 0; i < 4; ++i) res[i] = z[i] - x_[i];

    // S = H P H' + R = P[:4,:4] + R
    double S[16];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) S[i * 4 + j] = P_[i * 8 + j] + (i == j ? kR[i] : 0.);
    double Sinv[16];
    if (!inv4(S, Sinv)) return;  // 数值灾难时放弃本帧修正

    // gain = (S^{-1} H P)' → K[i][j] = sum_{k<4} P[k][i] * Sinv[j][k]  (8x4)
    double K[8][4];
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 4; ++j) {
            double acc = 0.;
            for (int k = 0; k < 4; ++k) acc += P_[k * 8 + i] * Sinv[j * 4 + k];
            K[i][j] = acc;
        }

    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 4; ++j) x_[i] += K[i][j] * res[j];

    // Joseph form：P = M P M' + K R K'，M = I - K H
    // MP[i][j] = P[i][j] - sum_{k<4} K[i][k] P[k][j]
    double mp[64];
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j) {
            double acc = P_[i * 8 + j];
            for (int k = 0; k < 4; ++k) acc -= K[i][k] * P_[k * 8 + j];
            mp[i * 8 + j] = acc;
        }
    // MPM'[i][j] = MP[i][j] - sum_{k<4} MP[i][k] K[j][k]
    // KRK'[i][j] = sum_{k<4} K[i][k] R_k K[j][k]
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j) {
            double acc = mp[i * 8 + j];
            for (int k = 0; k < 4; ++k)
                acc += -mp[i * 8 + k] * K[j][k] + K[i][k] * kR[k] * K[j][k];
            P_[i * 8 + j] = acc;
        }
}

// ------------------------------------------------------------- clamp_bbox
void clamp_bbox(float b[4], int img_w, int img_h) {
    auto clampf = [](float v, float hi) { return v < 0.f ? 0.f : (v > hi ? hi : v); };
    float x = clampf(b[0], (float)img_w);
    float y = clampf(b[1], (float)img_h);
    float x2 = clampf(b[0] + b[2], (float)img_w);
    float y2 = clampf(b[1] + b[3], (float)img_h);
    // 同 np.int32 截断
    int ix = (int)x, iy = (int)y;
    int iw = (int)(x2 - x), ih = (int)(y2 - y);
    if (iw < 3) {
        iw = 3;
        ix -= std::max(0, ix + iw - img_w);
    }
    if (ih < 3) {
        ih = 3;
        iy -= std::max(0, iy + ih - img_h);
    }
    b[0] = (float)ix;
    b[1] = (float)iy;
    b[2] = (float)iw;
    b[3] = (float)ih;
}

// ---------------------------------------------------------------- Tracker
const Track* Tracker::predict() {
    if (!kal_) return nullptr;
    kal_->predict();
    kal_->box(pred_box_);
    kal_->velocity(trk_.vel);
    std::memcpy(trk_.box, pred_box_, sizeof(trk_.box));
    return &trk_;
}

const Track* Tracker::update(const std::vector<Detection>& dets, int img_w, int img_h,
                             int64_t frame_idx) {
    trk_.event = "";
    last_match_ = -1;

    // 尺寸记忆过滤（track3.py _size_ok）
    std::vector<const Detection*> elig;
    elig.reserve(dets.size());
    bool size_prior = have_last_ && (frame_idx - last_frame_ <= p_.size_memory_frames);
    float prev_area = size_prior ? std::max(last_box_[2] * last_box_[3], 1.f) : 1.f;
    for (const auto& d : dets) {
        if (size_prior) {
            float area = (d.x2 - d.x1) * (d.y2 - d.y1);
            float r = area / prev_area;
            if (r < 1.f / p_.size_ratio_limit || r > p_.size_ratio_limit) continue;
        }
        elig.push_back(&d);
    }

    if (!kal_) {
        // SEARCH：全局检测，最高分直接锁定
        if (!elig.empty()) {
            const Detection* best = elig[0];
            for (const Detection* d : elig)
                if (d->score > best->score) best = d;
            float box[4] = {best->x1, best->y1, best->x2 - best->x1, best->y2 - best->y1};
            kal_.reset(new BoxKalman(box));
            last_match_ = (int)(best - dets.data());
            std::memcpy(trk_.box, box, sizeof(box));
            std::memset(trk_.vel, 0, sizeof(trk_.vel));
            trk_.score = best->score;
            trk_.hits = 1;
            trk_.misses = 0;
            trk_.state = TrackState::TRACK;
            trk_.event = ever_ ? "REACQUIRED" : "INITIAL_LOCK";
            ever_ = true;
            std::memcpy(last_box_, box, sizeof(box));
            last_frame_ = frame_idx;
            have_last_ = true;
        } else {
            trk_.state = TrackState::SEARCH;
            trk_.score = 0.f;
        }
        return &trk_;
    }

    // TRACK/COAST：最近中心关联 + misses 自适应门限（track3.py:144-167）
    float pcx, pcy;
    center_of(pred_box_, pcx, pcy);
    const Detection* cand = nullptr;
    float best_dist = 0.f;
    for (const Detection* d : elig) {
        float b[4] = {d->x1, d->y1, d->x2 - d->x1, d->y2 - d->y1};
        float cx, cy;
        center_of(b, cx, cy);
        float dist = std::hypot(cx - pcx, cy - pcy);
        if (!cand || dist < best_dist) {
            cand = d;
            best_dist = dist;
        }
    }
    if (cand) {
        float gate = std::max(p_.association_gate_px,
                              p_.association_gate_scale *
                                  std::hypot(pred_box_[2], pred_box_[3]));
        gate *= 1.f + (float)std::min(trk_.misses, p_.max_misses) / p_.max_misses;
        if (best_dist > gate) cand = nullptr;
    }

    if (cand) {
        float box[4] = {cand->x1, cand->y1, cand->x2 - cand->x1, cand->y2 - cand->y1};
        kal_->update(box);
        kal_->velocity(trk_.vel);
        last_match_ = (int)(cand - dets.data());
        std::memcpy(trk_.box, box, sizeof(box));
        trk_.score = cand->score;
        trk_.hits += 1;
        trk_.misses = 0;
        trk_.state = TrackState::TRACK;
        std::memcpy(last_box_, box, sizeof(box));
        last_frame_ = frame_idx;
        have_last_ = true;
    } else {
        trk_.misses += 1;
        trk_.hits = 0;
        if (trk_.misses <= p_.coast_frames) {
            trk_.state = TrackState::COAST;
            std::memcpy(trk_.box, pred_box_, sizeof(trk_.box));
            clamp_bbox(trk_.box, img_w, img_h);
            trk_.score = 0.f;
        } else {
            kal_.reset();
            trk_.misses = 0;
            trk_.state = TrackState::LOST;
            trk_.event = "LOST";
            trk_.score = 0.f;
        }
    }
    return &trk_;
}

}  // namespace yoloop
