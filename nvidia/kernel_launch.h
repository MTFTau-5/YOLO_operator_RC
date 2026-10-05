#pragma once
// 内部 kernel launch 原型 —— 仅 nvidia 模块内部使用
#include <cuda_runtime.h>
#include <cstdint>

#include "../common/yolo_types.h"
#include "yolo_ops.h"

namespace yoloop {
namespace cuda {
namespace detail {

void launch_decode(const float* pred_dev, const DeviceAnchorTable& tab,
                   int num_classes, float conf_thr,
                   Detection* dets_dev, int* count_dev, int cap,
                   cudaStream_t stream);

void launch_nms(const Detection* dets_dev, int n, float nms_thr,
                Detection* keep_dev, int* count_dev,
                cudaStream_t stream);

void launch_letterbox(const uint8_t* src_dev, int src_w, int src_h,
                      float* out_dev, int dst_w, int dst_h,
                      cudaStream_t stream);
void launch_dets_to_f6(const Detection* dets_dev, const int* count_dev,
                       float* out_f6, int* out_count, int cap,
                       cudaStream_t stream);

}  // namespace detail
}  // namespace cuda
}  // namespace yoloop
