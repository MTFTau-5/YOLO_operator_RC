#include "control_tx.h"

#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <netdb.h>
#include <unistd.h>

namespace yoloop {

ControlTx::~ControlTx() { close(); }

bool ControlTx::open(const std::string& host, uint16_t port) {
    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;
    char port_str[8];
    snprintf(port_str, sizeof(port_str), "%u", port);
    addrinfo* res = nullptr;
    if (getaddrinfo(host.c_str(), port_str, &hints, &res) != 0 || !res) {
        fprintf(stderr, "[control] 解析地址失败: %s:%u\n", host.c_str(), port);
        return false;
    }
    fd_ = socket(res->ai_family, res->ai_socktype | SOCK_NONBLOCK, res->ai_protocol);
    if (fd_ < 0) {
        freeaddrinfo(res);
        return false;
    }
    std::memcpy(&addr_, res->ai_addr, res->ai_addrlen);
    addr_len_ = (socklen_t)res->ai_addrlen;
    freeaddrinfo(res);
    fprintf(stderr, "[control] UDP 目标 %s:%u\n", host.c_str(), port);
    return true;
}

void ControlTx::send(const ControlPacket& pkt) {
    if (fd_ < 0) return;
    sendto(fd_, &pkt, sizeof(pkt), MSG_DONTWAIT, (sockaddr*)&addr_, addr_len_);
}

void ControlTx::close() {
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

}  // namespace yoloop
