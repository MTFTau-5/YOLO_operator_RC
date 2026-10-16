// test_queue —— 槽位/队列语义自查（PIPELINE.md §2 不变式、§3 丢旧帧、§6 关机）。
//
//   1. init 后 q_free 恰有 NUM_SLOTS=3 个互不相同的槽
//   2. free 空时 acquire_slot 从 q_gpu 偷最老槽，dropped 递增（FIFO 顺序）
//   3. 三队列槽数之和恒为 3（槽总数守恒）
//   4. q_gpu 也空时 acquire_slot 返回 nullptr
//   5. close 后阻塞 pop 解阻塞返回 false（不杀线程的关机路径）

#include <cassert>
#include <cstdio>
#include <set>
#include <thread>

#include "pipeline.h"

using namespace yoloop;

static int g_fail = 0;
#define CHECK(cond, msg)                          \
    do {                                          \
        if (!(cond)) {                            \
            printf("FAIL: %s\n", msg);            \
            g_fail++;                             \
        }                                         \
    } while (0)

static size_t total_in_queues(Pipeline& p) {
    return p.q_free.size() + p.q_gpu.size() + p.q_back.size();
}

int main() {
    Pipeline p;
    p.init(64, 64, 2, 100, 128, 128);  // 小缓冲即可

    // 1. 初始三槽全在 q_free
    CHECK(total_in_queues(p) == NUM_SLOTS, "初始槽总数 != 3");
    CHECK(p.q_free.size() == 3, "初始 q_free != 3");

    // 2. 取三个槽推入 q_gpu，再 acquire 应按 FIFO 偷最老
    Slot* s[3];
    std::set<Slot*> seen;
    for (int i = 0; i < 3; ++i) {
        s[i] = p.acquire_slot();
        CHECK(s[i] != nullptr, "acquire 前三次应成功");
        CHECK(seen.insert(s[i]).second, "槽位重复（不变式1 破坏）");
        p.q_gpu.push(s[i]);
    }
    CHECK(p.q_free.size() == 0 && p.q_gpu.size() == 3, "推入 q_gpu 后计数错");
    Slot* stolen[3];
    for (int i = 0; i < 3; ++i) {
        stolen[i] = p.acquire_slot();
        CHECK(stolen[i] == s[i], "丢旧帧应偷到最老槽（FIFO）");
        CHECK(p.dropped == (uint64_t)(i + 1), "dropped 计数错");
    }
    for (int i = 0; i < 3; ++i) p.q_free.push(stolen[i]);  // 归还以便后续
    CHECK(total_in_queues(p) == NUM_SLOTS, "槽总数不守恒（不变式3 破坏）");

    // 3. 全部槽占用（2 在 q_gpu、1 在手里"正在 GPU 上"），free 空 → 偷 q_gpu 最老
    Slot* a = p.acquire_slot();
    Slot* b = p.acquire_slot();
    Slot* c = p.acquire_slot();
    (void)c;
    p.q_gpu.push(a);
    p.q_gpu.push(b);
    uint64_t d0 = p.dropped;
    Slot* st = p.acquire_slot();
    CHECK(st == a, "应偷到 q_gpu 最老槽 a");
    CHECK(p.dropped == d0 + 1, "dropped 未递增");

    // 4. 再偷到 b；之后 free/q_gpu 都空 → nullptr
    Slot* st2 = p.acquire_slot();
    CHECK(st2 == b, "应偷到 b");
    Slot* none1 = p.acquire_slot();
    CHECK(none1 == nullptr, "free/q_gpu 都空应返回 nullptr");

    // 5. close 后 pop 解阻塞
    BlockQueue<int> q;
    bool pop_ret = true;
    std::thread th([&] {
        int v;
        pop_ret = q.pop(v);  // 阻塞直到 close
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    q.close();
    th.join();
    CHECK(!pop_ret, "close 后 pop 应返回 false");

    printf("%s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
