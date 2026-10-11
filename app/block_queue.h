#pragma once
// 阻塞队列 —— PIPELINE.md §2/§3
// 锁只保护队列本身；槽位数据面（img/blob/pred）零锁，由槽位状态机保证唯一归属。
// close() 后 pop 立即返回 false，用于关机时不杀线程、排空退出（§6）。

#include <condition_variable>
#include <deque>
#include <mutex>

namespace yoloop {

template <typename T>
class BlockQueue {
public:
    void push(T v) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            q_.push_back(std::move(v));
        }
        cv_.notify_one();
    }

    // 阻塞取最老；close 后且空返回 false
    bool pop(T& out) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return !q_.empty() || closed_; });
        if (q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop_front();
        return true;
    }

    bool try_pop(T& out) {
        std::lock_guard<std::mutex> lk(mu_);
        if (q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop_front();
        return true;
    }

    // 偷最老元素（丢旧帧语义）；空返回 false
    bool pop_oldest(T& out) { return try_pop(out); }

    void close() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            closed_ = true;
        }
        cv_.notify_all();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lk(mu_);
        return q_.size();
    }

private:
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::deque<T> q_;
    bool closed_ = false;
};

}  // namespace yoloop
