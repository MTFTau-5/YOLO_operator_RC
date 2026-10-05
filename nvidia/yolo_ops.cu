// host 侧封装: 实现 yolo_ops.h 全部接口
#include "yolo_ops.h"

#include <stdexcept>
#include <vector>

#include "kernel_launch.h"

#define YOLOOP_CUDA_CHECK(call)                                                  \
    do {                                                                         \
        cudaError_t err_ = (call);                                               \
        if (err_ != cudaSuccess)                                                 \
            throw std::runtime_error(std::string("CUDA error: ") +               \
                                     cudaGetErrorString(err_) + " at " +         \
                                     __FILE__ + ":" + std::to_string(__LINE__)); \
    } while (0)

namespace yoloop {
namespace cuda {

CudaAnchorTable::CudaAnchorTable(const AnchorTable& host) {
    dev_.n = host.n;
    if (host.n <= 0) return;
    const size_t bytes = (size_t)host.n * sizeof(float);
    YOLOOP_CUDA_CHECK(cudaMalloc(&buf_, 5 * bytes));
    // host 侧拼成单块后一次上传, 布局: [gx | gy | stride | aw | ah]
    std::vector<float> tmp((size_t)5 * host.n);
    const size_t n = (size_t)host.n;
    std::copy(host.gx.begin(), host.gx.end(), tmp.begin());
    std::copy(host.gy.begin(), host.gy.end(), tmp.begin() + n);
    std::copy(host.stride.begin(), host.stride.end(), tmp.begin() + 2 * n);
    std::copy(host.aw.begin(), host.aw.end(), tmp.begin() + 3 * n);
    std::copy(host.ah.begin(), host.ah.end(), tmp.begin() + 4 * n);
    YOLOOP_CUDA_CHECK(cudaMemcpy(buf_, tmp.data(), 5 * bytes, cudaMemcpyHostToDevice));
    dev_.gx = buf_;
    dev_.gy = buf_ + n;
    dev_.stride = buf_ + 2 * n;
    dev_.aw = buf_ + 3 * n;
    dev_.ah = buf_ + 4 * n;
}

CudaAnchorTable::~CudaAnchorTable() {
    if (buf_) cudaFree(buf_);
}

void decode(const float* pred_dev, const DeviceAnchorTable& tab,
            int num_classes, float conf_thr,
            Detection* dets_dev, int* count_dev, int cap,
            cudaStream_t stream) {
    YOLOOP_CUDA_CHECK(cudaMemsetAsync(count_dev, 0, sizeof(int), stream));
    detail::launch_decode(pred_dev, tab, num_classes, conf_thr,
                          dets_dev, count_dev, cap, stream);
}

void nms(const Detection* dets_dev, int n, float nms_thr,
         Detection* keep_dev, int* count_dev,
         cudaStream_t stream) {
    detail::launch_nms(dets_dev, n, nms_thr, keep_dev, count_dev, stream);
}

void letterbox_normalize(const uint8_t* src_dev, int src_w, int src_h,
                         float* out_dev, int dst_w, int dst_h,
                         cudaStream_t stream) {
    detail::launch_letterbox(src_dev, src_w, src_h, out_dev, dst_w, dst_h, stream);
}

}  // namespace cuda
}  // namespace yoloop

#undef YOLOOP_CUDA_CHECK
