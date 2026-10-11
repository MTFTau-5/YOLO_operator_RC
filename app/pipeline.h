#pragma once
#include <atomic>
#include "block_queue.h"
#include "slot.h"

namespace yoloop {

struct Pipeline {
    BlockQueue<Slot*> q_free;
    BlockQueue<Slot*> q_gpu;
    BlockQueue<Slot*> q_back;

    Slot slots[NUM_SLOTS];

    std::atomic<bool> run{true};
    std::atomic<uint64_t> dropped{0}; 
    std::atomic<int64_t> latency_us{0};
    void init(int in_w, int in_h, int c_total, int pred_n, int max_w, int max_h) {
        for (int i = 0; i < NUM_SLOTS; ++i) {
            Slot& s = slots[i];
            s.id = i;
            s.img.resize((size_t)max_w * max_h * 3);
            s.blob.resize((size_t)in_w * in_h * 3);
            s.pred.resize((size_t)c_total * pred_n);
            s.dets.reserve(1024);
            s.kept.reserve(300);
            q_free.push(&s);
        }
    }

    Slot* acquire_slot() {
        Slot* s = nullptr;
        if (q_free.pop_oldest(s)) return s;
        if (q_gpu.pop_oldest(s)) {
            dropped++;
            return s;
        }
        return nullptr;
    }

    void shutdown() {
        run = false;
        q_gpu.close();
        q_back.close();
        q_free.close();
    }

    void drain_back() {
        Slot* s = nullptr;
        while (q_back.try_pop(s)) q_free.push(s);
    }
};

}  // namespace yoloop
