#include "ort_infer.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include "spacemit_ort_env.h"

namespace yoloop {

struct OrtInfer::Impl {
    enum class Layout { ChannelMajor, BoxMajor, Planes };

    struct Head {
        std::vector<int64_t> shape;
        size_t elements = 0;
        int c = 0;
        int n = 0;
        int na = 1;
        Layout layout = Layout::ChannelMajor;
    };

    Impl() : env(ORT_LOGGING_LEVEL_WARNING, "yolo_rt") {}

    int c_total() const {
        return head_channels(cfg.head, cfg.num_classes, cfg.num_kpts, cfg.reg_max);
    }

    bool load(const InferConfig& config, int expected_n) {
        cfg = config;
        pred_n = 0;
        grid = AnchorTable{};
        heads.clear();
        staging.clear();

        try {
            session_options.SetIntraOpNumThreads(1);
            session_options.SetInterOpNumThreads(1);
            std::unordered_map<std::string, std::string> provider_options;
            provider_options["SPACEMIT_EP_INTRA_THREAD_NUM"] =
                std::to_string(std::max(1, cfg.cpu_threads));
            const char* env_options[] = {
                "SPACEMIT_EP_INTRA_THREAD_NUM",
                "SPACEMIT_EP_INTRA_THREAD_AFFINITY",
                "SPACEMIT_EP_DUMP_SUBGRAPHS",
                "SPACEMIT_EP_DEBUG_PROFILE",
                "SPACEMIT_EP_DUMP_TENSORS",
                "SPACEMIT_EP_DISABLE_FLOAT16_EPILOGUE",
                "SPACEMIT_EP_DENSE_ACCURACY_LEVEL",
            };
            for (const char* key : env_options) {
                if (const char* value = std::getenv(key)) provider_options[key] = value;
            }
#ifdef YOLOOP_RISCV_A100
            // Spine streams own disjoint cores. Leave one of K3's eight A100 cores
            // available for postprocessing before the EP checks out its stream.
            const int requested = std::stoi(provider_options["SPACEMIT_EP_INTRA_THREAD_NUM"]);
            const int granted = std::clamp(requested, 1, 7);
            if (requested != granted) {
                std::fprintf(stderr, "[spacemit-ort] A100 后处理预留 1 核，EP线程 %d -> %d\n", requested, granted);
                provider_options["SPACEMIT_EP_INTRA_THREAD_NUM"] = std::to_string(granted);
            }
#endif
            SessionOptionsSpaceMITEnvInit(session_options, provider_options);
            session = std::make_unique<Ort::Session>(env, cfg.model_path.c_str(), session_options);

            Ort::AllocatorWithDefaultOptions allocator;
            if (session->GetInputCount() != 1) {
                std::fprintf(stderr, "[spacemit-ort] 模型输入数 %zu，期望 1\n",
                             session->GetInputCount());
                return false;
            }
            auto actual_input = session->GetInputNameAllocated(0, allocator);
            if (cfg.input_name != actual_input.get()) {
                std::fprintf(stderr, "[spacemit-ort] 输入名 '%s' 与模型实际名称 '%s' 不符\n",
                             cfg.input_name.c_str(), actual_input.get());
                return false;
            }
            input_name = actual_input.get();

            output_names = cfg.head_names;
            if (output_names.empty()) {
                for (size_t i = 0; i < session->GetOutputCount(); ++i) {
                    auto name = session->GetOutputNameAllocated(i, allocator);
                    output_names.emplace_back(name.get());
                }
            }
            if (output_names.size() != session->GetOutputCount()) {
                std::fprintf(stderr,
                             "[spacemit-ort] 配置输出数 %zu，模型实际输出数 %zu；"
                             "当前实现要求显式列出全部输出\n",
                             output_names.size(), session->GetOutputCount());
                return false;
            }

            const int64_t input_shape[] = {1, 3, cfg.in_h, cfg.in_w};
            std::vector<float> zeros((size_t)cfg.in_w * cfg.in_h * 3, 0.f);
            if (std::getenv("YOLOOP_ORT_DEBUG_ADDR"))
                std::fprintf(stderr, "[spacemit-ort] 首次试推理输入地址: %p\n",
                             (const void*)zeros.data());
            auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            auto input = Ort::Value::CreateTensor<float>(memory, zeros.data(), zeros.size(),
                                                         input_shape, 4);
            const char* input_names[] = {input_name.c_str()};
            std::vector<const char*> output_name_ptrs;
            for (const auto& name : output_names) output_name_ptrs.push_back(name.c_str());
            Ort::RunOptions run_options;
            auto outputs = session->Run(run_options, input_names, &input, 1,
                                        output_name_ptrs.data(), output_name_ptrs.size());

            std::vector<LevelSpec> point_levels;
            int total = 0;
            const int expected_c = c_total();
            for (size_t i = 0; i < outputs.size(); ++i) {
                auto info = outputs[i].GetTensorTypeAndShapeInfo();
                if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                    std::fprintf(stderr, "[spacemit-ort] 输出 '%s' 不是 float32\n",
                                 output_names[i].c_str());
                    return false;
                }
                Head head;
                head.shape = info.GetShape();
                head.elements = info.GetElementCount();
                int c = 0;
                int n = 0;
                if (head.shape.size() == 4) {
                    if (head.shape[0] != 1 ||
                        (cfg.head != HeadType::AnchorRaw && cfg.head != HeadType::AfRaw &&
                         cfg.head != HeadType::AfDist)) {
                        std::fprintf(stderr, "[spacemit-ort] 输出 '%s' 四维布局不受支持\n",
                                     output_names[i].c_str());
                        return false;
                    }
                    c = (int)head.shape[1];
                    const int h = (int)head.shape[2];
                    const int w = (int)head.shape[3];
                    n = h * w;
                    if (cfg.in_w % w != 0 || cfg.in_h % h != 0 ||
                        cfg.in_w / w != cfg.in_h / h) {
                        std::fprintf(stderr, "[spacemit-ort] 输出 '%s' 网格与输入不成整数 stride\n",
                                     output_names[i].c_str());
                        return false;
                    }
                    head.layout = Layout::Planes;
                    if (cfg.head == HeadType::AnchorRaw) {
                        if (c % expected_c != 0) {
                            std::fprintf(stderr,
                                         "[spacemit-ort] v5 输出通道 %d 不是 %d 的整数倍\n",
                                         c, expected_c);
                            return false;
                        }
                        head.na = c / expected_c;
                        n *= head.na;
                        c = expected_c;
                    } else {
                        point_levels.push_back({cfg.in_w / w, w, h, {}});
                    }
                } else if (head.shape.size() == 3) {
                    if (head.shape[0] != 1) {
                        std::fprintf(stderr, "[spacemit-ort] 输出 '%s' batch 不为 1\n",
                                     output_names[i].c_str());
                        return false;
                    }
                    const int a = (int)head.shape[1];
                    const int b = (int)head.shape[2];
                    if (a == expected_c) {
                        c = a;
                        n = b;
                    } else if (b == expected_c) {
                        c = b;
                        n = a;
                        head.layout = Layout::BoxMajor;
                    } else {
                        std::fprintf(stderr,
                                     "[spacemit-ort] 输出 '%s' 形状与 C=%d 不符\n",
                                     output_names[i].c_str(), expected_c);
                        return false;
                    }
                } else if (head.shape.size() == 2) {
                    const int a = (int)head.shape[0];
                    const int b = (int)head.shape[1];
                    if (a == expected_c) {
                        c = a;
                        n = b;
                    } else if (b == expected_c) {
                        c = b;
                        n = a;
                        head.layout = Layout::BoxMajor;
                    } else {
                        std::fprintf(stderr,
                                     "[spacemit-ort] 输出 '%s' 形状与 C=%d 不符\n",
                                     output_names[i].c_str(), expected_c);
                        return false;
                    }
                } else {
                    std::fprintf(stderr, "[spacemit-ort] 输出 '%s' dims=%zu 不支持\n",
                                 output_names[i].c_str(), head.shape.size());
                    return false;
                }
                if (c != expected_c) {
                    std::fprintf(stderr, "[spacemit-ort] 输出 '%s' C=%d，期望 %d\n",
                                 output_names[i].c_str(), c, expected_c);
                    return false;
                }
                head.c = c;
                head.n = n;
                total += n;
                heads.push_back(std::move(head));
            }

            pred_n = total;
            if (cfg.head == HeadType::AfRaw || cfg.head == HeadType::AfDist)
                grid = build_point_table(point_levels);
            if (expected_n > 0 && pred_n != expected_n) {
                std::fprintf(stderr, "[spacemit-ort] 输出 N=%d，期望 %d\n", pred_n, expected_n);
                return false;
            }
            direct_output = heads.size() == 1 && heads[0].layout == Layout::ChannelMajor;
            staging.resize(heads.size());
            if (!direct_output) {
                for (size_t i = 0; i < heads.size(); ++i)
                    staging[i].resize(heads[i].elements);
            }
            std::fprintf(stderr,
                         "[spacemit-ort] 加载完成: %s, head=%s, N=%d, C=%d, "
                         "EP线程=%d, 直接输出绑定=%d\n",
                         cfg.model_path.c_str(), head_type_name(cfg.head), pred_n, expected_c,
                         std::stoi(provider_options["SPACEMIT_EP_INTRA_THREAD_NUM"]), (int)direct_output);
            return true;
        } catch (const Ort::Exception& e) {
            std::fprintf(stderr, "[spacemit-ort] 加载失败: %s\n", e.what());
            return false;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[spacemit-ort] 加载失败: %s\n", e.what());
            return false;
        }
    }

    void copy_outputs(float* pred) const {
        int n_off = 0;
        for (size_t i = 0; i < heads.size(); ++i) {
            const Head& head = heads[i];
            const float* src = staging[i].data();
            if (head.layout == Layout::Planes) {
                if (cfg.head == HeadType::AnchorRaw) {
                    const int hw = head.n / head.na;
                    const int raw_c = head.c * head.na;
                    for (int cc = 0; cc < raw_c; ++cc) {
                        float* dst = pred + (size_t)(cc % head.c) * pred_n + n_off +
                                     cc / head.c;
                        for (int j = 0; j < hw; ++j)
                            dst[(size_t)j * head.na] = src[(size_t)cc * hw + j];
                    }
                } else {
                    for (int c = 0; c < head.c; ++c)
                        std::memcpy(pred + (size_t)c * pred_n + n_off,
                                    src + (size_t)c * head.n,
                                    (size_t)head.n * sizeof(float));
                }
            } else if (head.layout == Layout::BoxMajor) {
                for (int j = 0; j < head.n; ++j)
                    for (int c = 0; c < head.c; ++c)
                        pred[(size_t)c * pred_n + n_off + j] = src[(size_t)j * head.c + c];
            } else {
                for (int c = 0; c < head.c; ++c)
                    std::memcpy(pred + (size_t)c * pred_n + n_off,
                                src + (size_t)c * head.n,
                                (size_t)head.n * sizeof(float));
            }
            n_off += head.n;
        }
    }

    bool infer(const float* blob, float* pred) {
        try {
            const int64_t input_shape[] = {1, 3, cfg.in_h, cfg.in_w};
            const size_t input_elements = (size_t)cfg.in_w * cfg.in_h * 3;
            auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            auto input = Ort::Value::CreateTensor<float>(
                memory, const_cast<float*>(blob), input_elements,
                input_shape, 4);
            Ort::IoBinding binding(*session);
            binding.BindInput(input_name.c_str(), input);
            for (size_t i = 0; i < heads.size(); ++i) {
                float* dst = direct_output ? pred : staging[i].data();
                auto output = Ort::Value::CreateTensor<float>(
                    memory, dst, heads[i].elements, heads[i].shape.data(), heads[i].shape.size());
                binding.BindOutput(output_names[i].c_str(), output);
            }
            Ort::RunOptions run_options;
            session->Run(run_options, binding);
            if (!direct_output) copy_outputs(pred);
            return true;
        } catch (const Ort::Exception& e) {
            std::fprintf(stderr, "[spacemit-ort] 推理失败: %s\n", e.what());
            return false;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[spacemit-ort] 推理失败: %s\n", e.what());
            return false;
        }
    }

    InferConfig cfg;
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    std::string input_name;
    std::vector<std::string> output_names;
    std::vector<Head> heads;
    std::vector<std::vector<float>> staging;
    AnchorTable grid;
    int pred_n = 0;
    bool direct_output = false;
};

OrtInfer::OrtInfer() : impl_(std::make_unique<Impl>()) {}
OrtInfer::~OrtInfer() = default;

bool OrtInfer::load(const InferConfig& cfg, int expected_n) {
    return impl_->load(cfg, expected_n);
}

bool OrtInfer::infer(const float* blob, float* pred) {
    return impl_->infer(blob, pred);
}

int OrtInfer::c_total() const { return impl_->c_total(); }
int OrtInfer::pred_n() const { return impl_->pred_n; }
const AnchorTable& OrtInfer::grid() const { return impl_->grid; }

}  // namespace yoloop
