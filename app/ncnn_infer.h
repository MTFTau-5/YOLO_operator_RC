#pragma once
// ncnn 推理封装 —— 线程B（GPU 提交）唯一使用者。
// 配置口径：docs/Operator_Suggestion.md §8.3（Vulkan fp16 全家桶）。
// 输出契约：README §3 —— 三个 head 原始输出按 stride 8→16→32 顺序拼成
// channel-major [4+1+num_classes, N] 连续写入调用方给的 pred 缓冲。

#include <string>
#include <vector>

namespace ncnn {
class Net;
}

namespace yoloop {

struct InferConfig {
    std::string param_path;
    std::string model_path;
    std::string input_name = "in0";
    std::vector<std::string> head_names;  // v5: stride 8/16/32 三个；v8: 通常单个拼接输出
    int in_w = 640;
    int in_h = 640;
    int model_type = 0;  // 0 = YOLOv5 契约 [4+1+nc, N]；1 = v8/v11 检测 [4+nc, N]；
                         // 2 = v8/v11 姿态 [4+nc+3*num_kpts, N]
    int num_classes = 2;
    int num_kpts = 0;    // model_type=2 时有效
    int gpu_index = 0;   // Vulkan 设备序号
    int cpu_threads = 1;
};

class NcnnInfer {
public:
    ~NcnnInfer();

    // 加载网络并校验输出形状。
    // expected_n > 0（v5：anchor 表 N）时校验总数；<= 0（v8）时由模型输出发现 N。
    bool load(const InferConfig& cfg, int expected_n);

    // blob: CHW f32 [0,1]（x86::letterbox_normalize 产物）。
    // pred: 容量 c_total()*pred_n()。失败返回 false（§6 GPU 出错路径）。
    bool infer(const float* blob, float* pred);

    int c_total() const {
        if (cfg_.model_type == 2) return 4 + cfg_.num_classes + 3 * cfg_.num_kpts;
        if (cfg_.model_type == 1) return 4 + cfg_.num_classes;
        return 5 + cfg_.num_classes;
    }
    int pred_n() const { return pred_n_; }
    int pred_elems() const { return c_total() * pred_n_; }

private:
    InferConfig cfg_;
    ncnn::Net* net_ = nullptr;
    int pred_n_ = 0;
    std::vector<int> head_n_;  // 各输出 blob 的列数（校验/拷贝用）
};

}  // namespace yoloop
