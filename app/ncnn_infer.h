#pragma once
// ncnn 推理封装 —— 线程B（GPU 提交）唯一使用者。
// 配置口径：docs/Operator_Suggestion.md §8.3（Vulkan fp16 全家桶）。
// 输出契约：按 cfg_.head（common/head_types.h）校验输出形状，统一转为
// channel-major [c_total(), pred_n()] 连续写入调用方给的 pred 缓冲：
//   AnchorRaw : 逐层 [na*(5+nc), h, w]，通道 anchor-major（c=a*(5+nc)+attr，
//               attr 序 tx,ty,tw,th,obj,cls…）；na 由 blob 通道数推导
//               （c % (5+nc) == 0），拷贝时重排为 [5+nc, N]，
//               列序 (y,x,a) 与 build_anchor_table 迭代序一致
//   AfFused/PoseFused : [1, C, N] / [C, N]（box-major [1, N, C] / [N, C] 自动转置）
//   AfRaw     : 逐层 [4*reg_max+nc, h, w] 原样拼接，网格表由探测形状推导（grid()）
//   AfDist    : 逐层 [4+nc, h, w]（4 边投影距离 + cls logit），网格表同 AfRaw
//   E2E       : [1, N, 6] / [N, 6] box-major，转置为 [6, N]（[6, N] 直通亦可）

#include "infer_backend.h"

namespace ncnn {
class Net;
}

namespace yoloop {

class NcnnInfer final : public InferBackend {
public:
    ~NcnnInfer() override;

    // 加载网络并校验输出形状。
    // expected_n > 0（AnchorRaw：anchor 表 N）时校验总数；<= 0 时由模型输出发现 N。
    bool load(const InferConfig& cfg, int expected_n) override;

    // blob: CHW f32 [0,1]（cpu::letterbox_normalize 产物）。
    // pred: 容量 c_total()*pred_n()。失败返回 false（§6 GPU 出错路径）。
    bool infer(const float* blob, float* pred) override;

    int c_total() const override {
        return head_channels(cfg_.head, cfg_.num_classes, cfg_.num_kpts, cfg_.reg_max);
    }
    int pred_n() const override { return pred_n_; }

    // AfRaw/AfDist：由探测到的逐层形状推导的 anchor-free 网格表（stride 由 in_size/grid 推断）
    const AnchorTable& grid() const override { return grid_; }

private:
    InferConfig cfg_;
    ncnn::Net* net_ = nullptr;
    int pred_n_ = 0;
    std::vector<int> head_n_;
    std::vector<char> head_boxmaj_;
    std::vector<int> head_na_;
    AnchorTable grid_;
};

}  // namespace yoloop
