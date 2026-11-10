#pragma once

#include <memory>
#include <string>
#include <vector>

#include "../common/head_types.h"

namespace yoloop {

struct InferConfig {
    std::string param_path;
    std::string model_path;
    std::string input_name = "in0";
    std::vector<std::string> head_names;
    int in_w = 640;
    int in_h = 640;
    HeadType head = HeadType::AnchorRaw;
    int num_classes = 2;
    int num_kpts = 0;
    int reg_max = 16;
    int gpu_index = 0;
    int cpu_threads = 1;
};

class InferBackend {
public:
    virtual ~InferBackend() = default;
    virtual bool load(const InferConfig& cfg, int expected_n) = 0;
    virtual bool infer(const float* blob, float* pred) = 0;
    virtual int c_total() const = 0;
    virtual int pred_n() const = 0;
    virtual const AnchorTable& grid() const = 0;

    int pred_elems() const { return c_total() * pred_n(); }
};

std::unique_ptr<InferBackend> make_infer_backend(const std::string& backend);

}  // namespace yoloop
