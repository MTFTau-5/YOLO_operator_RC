#include "infer_backend.h"

#include <cstdio>

#include "ncnn_infer.h"
#ifdef YOLOOP_HAS_SPACEMIT_ORT
#include "ort_infer.h"
#endif

namespace yoloop {

std::unique_ptr<InferBackend> make_infer_backend(const std::string& backend) {
    if (backend == "ncnn") return std::make_unique<NcnnInfer>();
    if (backend == "spacemit" || backend == "ort") {
#ifdef YOLOOP_HAS_SPACEMIT_ORT
        return std::make_unique<OrtInfer>();
#else
        std::fprintf(stderr,
                     "[infer] 当前构建未启用 SpaceMIT ONNX Runtime；"
                     "请安装 spacemit-onnxruntime 后重新配置构建\n");
        return nullptr;
#endif
    }
    std::fprintf(stderr, "[infer] 未知后端 '%s'，支持 ncnn / spacemit\n", backend.c_str());
    return nullptr;
}

}  // namespace yoloop
