#pragma once
// NVIDIA CUDA 算子 host 侧接口
// 所有函数异步于给定 stream; 设备内存在调用方侧管理或经 DeviceAnchorTable 持有
#include <cuda_runtime.h>
#include <cstdint>

#include "../common/yolo_types.h"

namespace yoloop {
namespace cuda {

struct DeviceAnchorTable {
    const float* gx = nullptr;
    const float* gy = nullptr;
    const float* stride = nullptr;
    const float* aw = nullptr;
    const float* ah = nullptr;
    int n = 0;
};

class CudaAnchorTable {
public:
    explicit CudaAnchorTable(const AnchorTable& host);
    ~CudaAnchorTable();
    CudaAnchorTable(const CudaAnchorTable&) = delete;
    CudaAnchorTable& operator=(const CudaAnchorTable&) = delete;
    const DeviceAnchorTable& get() const { return dev_; }

private:
    DeviceAnchorTable dev_;
    float* buf_ = nullptr;
};

void decode(const float* pred_dev, const DeviceAnchorTable& tab,
            int num_classes, float conf_thr,
            Detection* dets_dev, int* count_dev, int cap,
            cudaStream_t stream = 0);


void nms(const Detection* dets_dev, int n, float nms_thr,
         Detection* keep_dev, int* count_dev,
         cudaStream_t stream = 0);

// 预处理融合 kernel (README §2.4): letterbox(双线性) + /255 + HWC->CHW 单 kernel
void letterbox_normalize(const uint8_t* src_dev, int src_w, int src_h,
                         float* out_dev, int dst_w, int dst_h,
                         cudaStream_t stream = 0);

}  // namespace cuda
}  // namespace yoloop
