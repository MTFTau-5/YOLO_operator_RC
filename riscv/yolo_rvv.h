#pragma once

#include <cstdint>
#include <vector>

#include "../common/yolo_types.h"

namespace yoloop {
namespace riscv {

void letterbox_normalize(const uint8_t* src, int src_w, int src_h,
                         float* out, int dst_w, int dst_h);

std::vector<Detection> decode(const float* pred, const AnchorTable& tab,
                              int num_classes, float conf_thr);
std::vector<Detection> decode_af_raw(const float* pred, const AnchorTable& pts,
                                     int num_classes, int reg_max, float conf_thr);
std::vector<Detection> decode_af_dist(const float* pred, const AnchorTable& pts,
                                      int num_classes, float conf_thr);
std::vector<Detection> decode_v8(const float* pred, int num_classes, int n, float conf_thr);
void decode_v8_pose(const float* pred, int num_classes, int num_kpts, int n, float conf_thr,
                    std::vector<Detection>& dets, std::vector<float>& kpts);
std::vector<Detection> decode_e2e(const float* pred, int n, float conf_thr);

std::vector<Detection> nms(std::vector<Detection> dets, float nms_thr, int topk = 300);
std::vector<int32_t> nms_indices(const std::vector<Detection>& dets, float nms_thr,
                                 int topk = 300);

}  // namespace riscv
}  // namespace yoloop
