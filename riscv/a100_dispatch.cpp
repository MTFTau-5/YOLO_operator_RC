#include "a100_dispatch.h"

#include <spert.hpp>
#include <cstdio>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>

namespace yoloop::riscv::detail {
namespace {
thread_local bool active = false;

struct Executor {
    spert::BackendHandle backend = spert::backend("spacemit-k3");
    spert::Stream stream{backend, 1};
    std::mutex mutex;
    bool reported_worker = false;

    Executor() {
        if (!backend.valid() || !stream.valid())
            throw std::runtime_error("A100 postprocess: spacemit-k3 backend/core unavailable");
        const auto info = spert::backend_info(backend);
        std::fprintf(stderr, "[postprocess] backend=spacemit-k3 cores=%u arch=%lld vlen=%zu bytes\n",
                     stream.core_count(), (long long)info.core_arch_id, info.vlen);
    }
};
}

bool on_a100_worker() { return active; }

void dispatch_a100(const std::function<void()>& work) {
    static Executor executor;
    std::lock_guard<std::mutex> lock(executor.mutex);
    std::exception_ptr error;
    auto task = executor.stream.launch(spert::Grid{1}, [&](spert::Context*) {
        active = true;
        if (!executor.reported_worker) {
            size_t vlen = 0;
            asm volatile("csrr %0, vlenb" : "=r"(vlen));
            std::fprintf(stderr, "[postprocess] A100 worker vlen=%zu bytes\n", vlen);
            executor.reported_worker = true;
        }
        try { work(); } catch (...) { error = std::current_exception(); }
        active = false;
    });
    const auto status = task.sync();
    if (status != spert::Status::Ok)
        throw std::runtime_error(std::string("A100 postprocess: ") + spert::to_string(status));
    if (error) std::rethrow_exception(error);
}
}  // namespace yoloop::riscv::detail
