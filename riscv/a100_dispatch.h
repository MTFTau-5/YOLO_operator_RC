#pragma once

#ifdef YOLOOP_RISCV_A100
#include <functional>
#include <optional>
#include <type_traits>
#include <utility>
#endif

namespace yoloop::riscv::detail {
#ifdef YOLOOP_RISCV_A100
bool on_a100_worker();
void dispatch_a100(const std::function<void()>& work);
#endif

// Synchronous API: caller buffers remain alive until the A100 task completes.
template <class F>
auto postprocess(F&& work) -> decltype(work()) {
#ifdef YOLOOP_RISCV_A100
    if (on_a100_worker()) return work();
    if constexpr (std::is_void_v<decltype(work())>) {
        dispatch_a100(work);
    } else {
        std::optional<decltype(work())> result;
        dispatch_a100([&] { result.emplace(work()); });
        return std::move(*result);
    }
#else
    return work();
#endif
}
}  // namespace yoloop::riscv::detail
