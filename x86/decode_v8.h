#pragma once
// YOLOv8/v11 系 decode —— Detect 头保留在 ncnn 图内（DFL+网格解码+sigmoid 由 GPU 完成），
// 模型输出已是"解码后"格式，channel-major [C, N]：
//   检测: C = 4 + num_classes          row 0..3 = cx, cy, w, h（网络输入尺度像素）
//                                     row 4..  = 各类别概率（已 sigmoid）
//   姿态: C = 4 + num_classes + 3*num_kpts，关键点 (x, y, conf) 像素
// 本算子只做：逐 anchor 取最大类别概率 → 阈值筛选 → xywh 转角点 →（pose）收集关键点。
// N 为全部 stride 拼接后的总 anchor 数（通常 8400），顺序无关。

#include <cstdint>
#include <vector>

#include "../common/yolo_types.h"

namespace yoloop {
namespace x86 {

// 检测版：返回阈值后的候选（label = argmax 类别）
std::vector<Detection> decode_v8(const float* pred, int num_classes, int n, float conf_thr);

// 姿态版：dets 与 kpts 平行输出，kpts 按 dets 顺序每框 3*num_kpts 个 float 连续排布
void decode_v8_pose(const float* pred, int num_classes, int num_kpts, int n, float conf_thr,
                    std::vector<Detection>& dets, std::vector<float>& kpts);

}  // namespace x86
}  // namespace yoloop
