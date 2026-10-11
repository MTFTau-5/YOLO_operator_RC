#include "ncnn_infer.h"

#include <cstdio>
#include <cstring>

#include "net.h"
#if NCNN_VULKAN
#include "gpu.h"
#endif

namespace yoloop {

NcnnInfer::~NcnnInfer() {
    if (net_) {
        net_->clear();
        delete net_;
    }
}

bool NcnnInfer::load(const InferConfig& cfg, int expected_n) {
    cfg_ = cfg;
    pred_n_ = 0;

    net_ = new ncnn::Net();
    // §8.3：Vulkan fp16 全家桶
    net_->opt.use_vulkan_compute = true;
    net_->opt.use_fp16_packed = true;
    net_->opt.use_fp16_storage = true;
    net_->opt.use_fp16_arithmetic = true;
    net_->opt.use_shader_pack8 = true;
    net_->opt.use_packing_layout = true;
    net_->opt.lightmode = true;
    net_->opt.num_threads = cfg_.cpu_threads;
#if NCNN_VULKAN
    if (cfg_.gpu_index >= 0 && cfg_.gpu_index < ncnn::get_gpu_count())
        net_->set_vulkan_device(cfg_.gpu_index);
#endif

    if (net_->load_param(cfg_.param_path.c_str()) != 0) {
        fprintf(stderr, "[ncnn] load_param 失败: %s\n", cfg_.param_path.c_str());
        return false;
    }
    if (net_->load_model(cfg_.model_path.c_str()) != 0) {
        fprintf(stderr, "[ncnn] load_model 失败: %s\n", cfg_.model_path.c_str());
        return false;
    }

    // 用零输入试跑一次，校验各输出形状并确定总列数 N
    std::vector<float> zeros((size_t)cfg_.in_w * cfg_.in_h * 3, 0.f);
    ncnn::Mat in(cfg_.in_w, cfg_.in_h, 3, zeros.data());
    ncnn::Extractor ex = net_->create_extractor();
    if (ex.input(cfg_.input_name.c_str(), in) != 0) {
        fprintf(stderr, "[ncnn] 输入 blob '%s' 不存在\n", cfg_.input_name.c_str());
        return false;
    }
    int total = 0;
    const int c_total = this->c_total();
    head_n_.clear();
    for (const auto& name : cfg_.head_names) {
        ncnn::Mat out;
        if (ex.extract(name.c_str(), out) != 0) {
            fprintf(stderr, "[ncnn] 输出 blob '%s' 提取失败\n", name.c_str());
            return false;
        }
        int c, n;
        if (out.dims == 3 && out.c > 1) {  // [C, h, w] 逐 channel 平面（v5 逐 head）
            c = out.c;
            n = out.w * out.h;
        } else if (out.dims == 3) {  // [1, C, N] 拼接输出（v8 典型）
            c = out.h;
            n = out.w;
        } else if (out.dims == 2) {  // [C, N] 已压平
            c = out.h;
            n = out.w;
        } else {
            fprintf(stderr, "[ncnn] head '%s' dims=%d 不支持\n", name.c_str(), out.dims);
            return false;
        }
        if (c != c_total) {
            fprintf(stderr, "[ncnn] head '%s' 通道数 %d, 期望 %d（model_type=%d, nc=%d, nkpt=%d）\n",
                    name.c_str(), c, c_total, cfg_.model_type, cfg_.num_classes,
                    cfg_.num_kpts);
            return false;
        }
        head_n_.push_back(n);
        total += n;
    }
    pred_n_ = total;
    if (expected_n > 0 && total != expected_n) {
        fprintf(stderr, "[ncnn] 输出 anchor 总数 %d 与 anchor 表 N=%d 不一致\n", total,
                expected_n);
        return false;
    }
    fprintf(stderr, "[ncnn] 加载完成: %s / %s, type=%d, N=%d, C=%d, Vulkan=%d\n",
            cfg_.param_path.c_str(), cfg_.model_path.c_str(), cfg_.model_type, pred_n_,
            c_total, (int)net_->opt.use_vulkan_compute);
    return true;
}

bool NcnnInfer::infer(const float* blob, float* pred) {
    ncnn::Mat in(cfg_.in_w, cfg_.in_h, 3, (void*)blob);
    ncnn::Extractor ex = net_->create_extractor();
    if (ex.input(cfg_.input_name.c_str(), in) != 0) return false;

    const int c_total = this->c_total();
    int n_off = 0;
    for (size_t i = 0; i < cfg_.head_names.size(); ++i) {
        ncnn::Mat out;
        if (ex.extract(cfg_.head_names[i].c_str(), out) != 0) return false;
        // extract type=0：已转 fp32、解 packing。目标布局 channel-major [C, N]。
        const int n = head_n_[i];
        if (out.dims == 3 && out.c > 1) {
            // [C, h, w]：逐 channel 平面拷贝（v5 逐 head 契约）
            for (int c = 0; c < c_total; ++c) {
                ncnn::Mat ch = out.channel(c);
                std::memcpy(pred + (size_t)c * pred_n_ + n_off, (const float*)ch.data,
                            (size_t)n * sizeof(float));
            }
        } else {
            // [C, N] 或 [1, C, N]：行主序逐行拷贝（多输出列向拼接也正确）
            ncnn::Mat flat = (out.dims == 3) ? out.channel(0) : out;
            for (int c = 0; c < c_total; ++c)
                std::memcpy(pred + (size_t)c * pred_n_ + n_off, flat.row(c),
                            (size_t)n * sizeof(float));
        }
        n_off += n;
    }
    return true;
}

}  // namespace yoloop
