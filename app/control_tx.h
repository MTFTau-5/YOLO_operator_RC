#pragma once
// 控制输出 —— UDP 非阻塞发送确认轨迹（PIPELINE.md §4：hits>=2 && misses==0 才发）。
// 无接收方不阻塞、不报错，掉包不重传（实时系统发新帧比重传旧帧有用）。

#include <cstdint>
#include <string>
#include <sys/socket.h>

namespace yoloop {

#pragma pack(push, 1)
struct ControlPacket {
    static constexpr int MAX_KPTS = 32;  // 单包关键点上限（角点扫描够用）
    uint32_t magic = 0x5954u;      // 'YT'
    uint32_t frame_id = 0;
    uint32_t state = 0;            // 0 SEARCH / 1 TRACK / 2 COAST / 3 LOST
    float cx = 0, cy = 0;          // 中心（图像坐标）
    float w = 0, h = 0;
    float vx = 0, vy = 0;          // KF 速度（像素/帧）
    float vw = 0, vh = 0;
    float score = 0;
    uint32_t kpt_count = 0;        // pose：本帧命中框的关键点数（0 = 无/非 pose）
    float kpts[MAX_KPTS * 3] = {}; // (x, y, conf) × kpt_count，相机原图坐标
};
#pragma pack(pop)

class ControlTx {
public:
    ~ControlTx();
    bool open(const std::string& host, uint16_t port);
    bool valid() const { return fd_ >= 0; }
    void send(const ControlPacket& pkt);  // 非阻塞，失败静默丢弃
    void close();

private:
    int fd_ = -1;
    struct sockaddr_storage addr_;
    socklen_t addr_len_ = 0;
};

}  // namespace yoloop
