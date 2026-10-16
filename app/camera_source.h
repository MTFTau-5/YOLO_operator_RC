#pragma once
// 图像采集源接口 —— 按需求只保留接口，不接具体设备。
// 后续接入真实相机（V4L2/OpenCV/千兆网口 SDK）时实现本接口即可，
// 流水线其余部分不动。
//
// 契约（PIPELINE.md §4/§6）：
//   grab()  无新帧返回 false（线程A归还槽位、continue，不空转）；
//           成功时把 HWC u8 帧拷入 dst（容量 max_w*max_h*3），写回 w/h。
//   reopen() 相机掉线时由线程A以 1s 周期调用重连；恢复前轨迹 misses 递增，
//           控制端可判视觉失效。

#include <cstdint>

namespace yoloop {

class CameraSource {
public:
    virtual ~CameraSource() = default;
    virtual bool grab(uint8_t* dst, int& w, int& h) = 0;
    virtual bool reopen() = 0;
};

// 占位实现：恒无帧。用于流水线联调与无相机环境冒烟。
class NullSource : public CameraSource {
public:
    bool grab(uint8_t*, int&, int&) override { return false; }
    bool reopen() override { return false; }
};

}  // namespace yoloop
