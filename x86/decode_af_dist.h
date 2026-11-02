#pragma once
// v8/v9/v11 第三种切分：DFL 期望留在图内（Detect 头切在 DFL 之后、grid+dist2bbox
// 之前），逐层 [4+nc, h, w]，C++ 侧只做网格重建 + dist2bbox。
// 输出契约（ncnn_infer 拼接为 channel-major [C, N]）：
//   通道 0..3 : l/t/r/b 四条边的投影距离（已完成 DFL softmax 期望）
//   通道 4..  : num_classes 个类别 logit
// pts: build_point_table 产物（gx/gy = 网格中心，stride = 层步长，逐层顺序无关）
// 与 AfRaw 相比输出通道从 4*reg_max+nc 降到 4+nc（拷贝体量 ≈ fused）；
// 代价是图内多了 DFL 的 reshape/softmax。**当前实现**下 torch 导出把空间尺寸
// 烘为常量（同 v11 的工具链限制，见 tools/export_af_raw.py 注释），换尺寸需重导；
// shape-safe 的逐层 DFL 投影理论上可同时保住小输出与动态性，未投入。

#include <vector>

#include "../common/yolo_types.h"

namespace yoloop {
namespace x86 {

// 返回阈值后的候选（label = argmax 类别），未做 NMS
std::vector<Detection> decode_af_dist(const float* pred, const AnchorTable& pts,
                                      int num_classes, float conf_thr);

}  // namespace x86
}  // namespace yoloop
