#pragma once
// ncnn 推理封装 —— 线程B（GPU 提交）唯一使用者。
// 配置口径：docs/Operator_Suggestion.md §8.3（Vulkan fp16 全家桶）。
// 输出契约：按 cfg_.head（common/head_types.h）校验输出形状，统一转为
// channel-major [c_total(), pred_n()] 连续写入调用方给的 pred 缓冲：
//   AnchorRaw : 逐层 [na*(5+nc), h, w] 原样拼接
//   AfFused/PoseFused : [1, C, N] / [C, N]（box-major [1, N, C] / [N, C] 自动转置）
//   AfRaw     : 逐层 [4*reg_max+nc, h, w] 原样拼接，网格表由探测形状推导（grid()）
//   E2E       : [1, N, 6] / [N, 6] box-major，转置为 [6, N]（[6, N] 直通亦可）

#include <string>
#include <vector>

#include "../common/head_types.h"

namespace ncnn {
class Net;
}

namespace yoloop {

struct InferConfig {
    std::string param_path;
    std::string model_path;
    std::string input_name = "in0";
    std::vector<std::string> head_names;  // 逐层 head 按 stride 升序；fused/e2e 通常单个
    int in_w = 640;
    int in_h = 640;
    HeadType head = HeadType::AnchorRaw;
    int num_classes = 2;
    int num_kpts = 0;   // PoseFused 时有效
    int reg_max = 16;   // AfRaw 时有效
    int gpu_index = 0;  // Vulkan 设备序号
    int cpu_threads = 1;
};

class NcnnInfer {
public:
    ~NcnnInfer();

    // 加载网络并校验输出形状。
    // expected_n > 0（AnchorRaw：anchor 表 N）时校验总数；<= 0 时由模型输出发现 N。
    bool load(const InferConfig& cfg, int expected_n);

    // blob: CHW f32 [0,1]（x86::letterbox_normalize 产物）。
    // pred: 容量 c_total()*pred_n()。失败返回 false（§6 GPU 出错路径）。
    bool infer(const float* blob, float* pred);

    int c_total() const {
        return head_channels(cfg_.head, cfg_.num_classes, cfg_.num_kpts, cfg_.reg_max);
    }
    int pred_n() const { return pred_n_; }
    int pred_elems() const { return c_total() * pred_n_; }

    // AfRaw：由探测到的逐层形状推导的 anchor-free 网格表（stride 由 in_size/grid 推断）
    const AnchorTable& grid() const { return grid_; }

private:
    InferConfig cfg_;
    ncnn::Net* net_ = nullptr;
    int pred_n_ = 0;
    std::vector<int> head_n_;        // 各输出 blob 的列数（校验/拷贝用）
    std::vector<char> head_boxmaj_;  // 各输出 blob 是否 box-major [N, C]（拷贝时转置）
    AnchorTable grid_;               // 仅 AfRaw 有效
};

}  // namespace yoloop
