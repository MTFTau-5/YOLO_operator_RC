#include "a100_dispatch.h"

#include <atomic>
#include <cstdio>
#include <stdexcept>
#include <thread>
#include <vector>

using yoloop::riscv::detail::on_a100_worker;
using yoloop::riscv::detail::postprocess;

int main() {
    try {
        if (on_a100_worker()) throw std::runtime_error("caller incorrectly marked as worker");
        const auto bytes = postprocess([] {
            if (!on_a100_worker()) throw std::runtime_error("task stayed on host");
            size_t value;
            asm volatile("csrr %0, vlenb" : "=r"(value));
            return value;
        });
        if (bytes != 128) throw std::runtime_error("expected K3 A100 VLEN=1024");
        if (postprocess([] { return postprocess([] { return 42; }); }) != 42)
            throw std::runtime_error("nested call failed");
        bool caught = false;
        try { postprocess([] { throw std::runtime_error("worker exception"); }); }
        catch (const std::runtime_error&) { caught = true; }
        if (!caught) throw std::runtime_error("worker exception lost");
        std::atomic<int> count{0};
        std::atomic<int> failures{0};
        std::vector<std::thread> callers;
        for (int t = 0; t < 4; ++t) callers.emplace_back([&] {
            try {
                for (int i = 0; i < 5; ++i) postprocess([&] {
                    if (!on_a100_worker()) throw std::runtime_error("concurrent task stayed on host");
                    ++count;
                });
            } catch (...) { ++failures; }
        });
        for (auto& caller : callers) caller.join();
        if (count != 20 || failures != 0 || on_a100_worker())
            throw std::runtime_error("concurrency/worker state failed");
        std::puts("PASS: A100 VLEN=1024, synchronous/nested/concurrent dispatch, exception propagation");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
