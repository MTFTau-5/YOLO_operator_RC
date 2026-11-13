#pragma once

#include <memory>

#include "infer_backend.h"

namespace yoloop {

class OrtInfer final : public InferBackend {
public:
    OrtInfer();
    ~OrtInfer() override;

    bool load(const InferConfig& cfg, int expected_n) override;
    bool infer(const float* blob, float* pred) override;
    int c_total() const override;
    int pred_n() const override;
    const AnchorTable& grid() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace yoloop
