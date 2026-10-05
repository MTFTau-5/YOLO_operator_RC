// pybind11 绑定: 模块名 yolo_ops_cuda
//   decode(pred[C,N] float32 channel-major, num_classes, conf_thr,
//          input_w=640, input_h=640) -> [M,6]
//   nms(dets[M,6], nms_thr) -> [K,6]
//   letterbox(img uint8[H,W,3], dst_w, dst_h) -> [3,dst_h,dst_w] float32
// anchor 表按 (input_w, input_h) 自适应构建并缓存; 输入边长须能被 32 整除
#include <cuda_runtime.h>

#include <map>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "../common/yolo_types.h"
#include "yolo_ops.h"

namespace py = pybind11;

namespace {

#define CUDA_CHECK(call)                                                           \
    do {                                                                           \
        cudaError_t err_ = (call);                                                 \
        if (err_ != cudaSuccess)                                                   \
            throw std::runtime_error(std::string("CUDA error: ") +                 \
                                     cudaGetErrorString(err_));                    \
    } while (0)

// anchor 表按 (input_w, input_h) 缓存; 每种分辨率首次使用时构建上传
yoloop::cuda::CudaAnchorTable& anchor_table_for(int input_w, int input_h) {
    if (input_w % 32 != 0 || input_h % 32 != 0)
        throw std::invalid_argument("input_w/input_h must be divisible by 32 (max stride)");
    static std::map<std::pair<int, int>, std::unique_ptr<yoloop::cuda::CudaAnchorTable>> cache;
    auto key = std::make_pair(input_w, input_h);
    auto it = cache.find(key);
    if (it == cache.end()) {
        yoloop::AnchorTable host =
            yoloop::build_anchor_table(yoloop::default_levels(input_w, input_h));
        it = cache.emplace(key, std::make_unique<yoloop::cuda::CudaAnchorTable>(host)).first;
    }
    return *it->second;
}

py::array_t<float> decode(py::array_t<float, py::array::c_style | py::array::forcecast> pred,
                          int num_classes, float conf_thr, int input_w, int input_h) {
    if (pred.ndim() != 2) throw std::invalid_argument("pred must be 2-D [C, N]");
    const int C = (int)pred.shape(0);
    const int N = (int)pred.shape(1);
    if (C != 4 + 1 + num_classes)
        throw std::invalid_argument("pred.shape[0] != 4 + 1 + num_classes");
    auto& tab = anchor_table_for(input_w, input_h);
    if (N != tab.get().n)
        throw std::invalid_argument(
            "pred.shape[1] != anchor count for input size " + std::to_string(input_w) + "x" +
            std::to_string(input_h) + " (expected " + std::to_string(tab.get().n) + ")");

    float* pred_dev = nullptr;
    yoloop::Detection* dets_dev = nullptr;
    int* count_dev = nullptr;
    CUDA_CHECK(cudaMalloc(&pred_dev, (size_t)C * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dets_dev, (size_t)N * sizeof(yoloop::Detection)));
    CUDA_CHECK(cudaMalloc(&count_dev, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(pred_dev, pred.data(), (size_t)C * N * sizeof(float),
                          cudaMemcpyHostToDevice));
    yoloop::cuda::decode(pred_dev, tab.get(), num_classes, conf_thr,
                         dets_dev, count_dev, N);
    int cnt = 0;
    CUDA_CHECK(cudaMemcpy(&cnt, count_dev, sizeof(int), cudaMemcpyDeviceToHost));
    cnt = std::min(cnt, N);
    std::vector<yoloop::Detection> host((size_t)cnt);
    CUDA_CHECK(cudaMemcpy(host.data(), dets_dev, (size_t)cnt * sizeof(yoloop::Detection),
                          cudaMemcpyDeviceToHost));
    cudaFree(pred_dev);
    cudaFree(dets_dev);
    cudaFree(count_dev);

    py::array_t<float> out({(py::ssize_t)cnt, (py::ssize_t)6});
    float* o = out.mutable_data();
    for (int i = 0; i < cnt; ++i) {
        o[(size_t)i * 6 + 0] = host[(size_t)i].x1;
        o[(size_t)i * 6 + 1] = host[(size_t)i].y1;
        o[(size_t)i * 6 + 2] = host[(size_t)i].x2;
        o[(size_t)i * 6 + 3] = host[(size_t)i].y2;
        o[(size_t)i * 6 + 4] = host[(size_t)i].score;
        o[(size_t)i * 6 + 5] = (float)host[(size_t)i].label;
    }
    return out;
}

py::array_t<float> nms(py::array_t<float, py::array::c_style | py::array::forcecast> dets,
                       float nms_thr) {
    if (dets.ndim() != 2 || dets.shape(1) != 6)
        throw std::invalid_argument("dets must be [M, 6]");
    const int M = (int)dets.shape(0);
    if (M == 0)
        return py::array_t<float>(py::array::ShapeContainer{(py::ssize_t)0, (py::ssize_t)6});

    const float* p = dets.data();
    std::vector<yoloop::Detection> host((size_t)M);
    for (int i = 0; i < M; ++i) {
        host[(size_t)i] = {p[(size_t)i * 6 + 0], p[(size_t)i * 6 + 1],
                           p[(size_t)i * 6 + 2], p[(size_t)i * 6 + 3],
                           p[(size_t)i * 6 + 4], (int32_t)p[(size_t)i * 6 + 5]};
    }
    yoloop::Detection* dets_dev = nullptr;
    yoloop::Detection* keep_dev = nullptr;
    int* count_dev = nullptr;
    CUDA_CHECK(cudaMalloc(&dets_dev, (size_t)M * sizeof(yoloop::Detection)));
    CUDA_CHECK(cudaMalloc(&keep_dev, (size_t)M * sizeof(yoloop::Detection)));
    CUDA_CHECK(cudaMalloc(&count_dev, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dets_dev, host.data(), (size_t)M * sizeof(yoloop::Detection),
                          cudaMemcpyHostToDevice));
    yoloop::cuda::nms(dets_dev, M, nms_thr, keep_dev, count_dev);
    int cnt = 0;
    CUDA_CHECK(cudaMemcpy(&cnt, count_dev, sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<yoloop::Detection> kept((size_t)cnt);
    CUDA_CHECK(cudaMemcpy(kept.data(), keep_dev, (size_t)cnt * sizeof(yoloop::Detection),
                          cudaMemcpyDeviceToHost));
    cudaFree(dets_dev);
    cudaFree(keep_dev);
    cudaFree(count_dev);

    py::array_t<float> out({(py::ssize_t)cnt, (py::ssize_t)6});
    float* o = out.mutable_data();
    for (int i = 0; i < cnt; ++i) {
        o[(size_t)i * 6 + 0] = kept[(size_t)i].x1;
        o[(size_t)i * 6 + 1] = kept[(size_t)i].y1;
        o[(size_t)i * 6 + 2] = kept[(size_t)i].x2;
        o[(size_t)i * 6 + 3] = kept[(size_t)i].y2;
        o[(size_t)i * 6 + 4] = kept[(size_t)i].score;
        o[(size_t)i * 6 + 5] = (float)kept[(size_t)i].label;
    }
    return out;
}

py::array_t<float> letterbox(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> img,
    int dst_w, int dst_h) {
    if (img.ndim() != 3 || img.shape(2) != 3)
        throw std::invalid_argument("img must be [H, W, 3] uint8");
    const int sh = (int)img.shape(0), sw = (int)img.shape(1);
    uint8_t* img_dev = nullptr;
    float* out_dev = nullptr;
    CUDA_CHECK(cudaMalloc(&img_dev, (size_t)sw * sh * 3));
    CUDA_CHECK(cudaMalloc(&out_dev, (size_t)3 * dst_w * dst_h * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(img_dev, img.data(), (size_t)sw * sh * 3,
                          cudaMemcpyHostToDevice));
    yoloop::cuda::letterbox_normalize(img_dev, sw, sh, out_dev, dst_w, dst_h);
    py::array_t<float> out({(py::ssize_t)3, (py::ssize_t)dst_h, (py::ssize_t)dst_w});
    CUDA_CHECK(cudaMemcpy(out.mutable_data(), out_dev,
                          (size_t)3 * dst_w * dst_h * sizeof(float),
                          cudaMemcpyDeviceToHost));
    cudaFree(img_dev);
    cudaFree(out_dev);
    return out;
}

}  // namespace

PYBIND11_MODULE(yolo_ops_cuda, m) {
    m.doc() = "YOLO CUDA ops: decode / nms / letterbox_normalize";
    m.def("decode", &decode, py::arg("pred"), py::arg("num_classes"), py::arg("conf_thr"),
          py::arg("input_w") = 640, py::arg("input_h") = 640,
          "decode channel-major pred [C,N] -> [M,6] (x1,y1,x2,y2,score,label); "
          "anchor 表按 (input_w,input_h) 自适应, 边长须被 32 整除");
    m.def("nms", &nms, py::arg("dets"), py::arg("nms_thr"),
          "class-aware bitmask NMS, [M,6] -> [K,6]");
    m.def("letterbox", &letterbox, py::arg("img"), py::arg("dst_w"), py::arg("dst_h"),
          "letterbox + /255 + HWC(uint8) -> CHW(float32)");
}
