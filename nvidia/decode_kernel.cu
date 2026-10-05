#include "kernel_launch.h"

namespace yoloop {
namespace cuda {
namespace detail {

__device__ __forceinline__ float sigmoid_fast(float x) {
    return 1.f / (1.f + __expf(-x));
}

__global__ void decode_kernel(const float* __restrict__ pred,
                              DeviceAnchorTable tab,
                              int num_classes, float logit_thr, float conf_thr,
                              Detection* __restrict__ dets,
                              int* __restrict__ count, int cap) {
    const int n = tab.n;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    bool alive = false;
    Detection d{};
    if (i < n) {
        const float o = pred[(size_t)4 * n + i];
        if (o >= logit_thr) {  // logit 域快筛, 等价于 sigmoid(o) >= conf_thr
            int best = 0;
            float best_logit = pred[(size_t)5 * n + i];
            for (int c = 1; c < num_classes; ++c) {
                const float v = pred[(size_t)(5 + c) * n + i];
                if (v > best_logit) { best_logit = v; best = c; }
            }
            const float obj = sigmoid_fast(o);
            const float score = obj * sigmoid_fast(best_logit);
            if (score >= conf_thr) {
                const float sx = sigmoid_fast(pred[(size_t)0 * n + i]);
                const float sy = sigmoid_fast(pred[(size_t)1 * n + i]);
                const float sw = sigmoid_fast(pred[(size_t)2 * n + i]);
                const float sh = sigmoid_fast(pred[(size_t)3 * n + i]);
                const float x = (sx * 2.f - 0.5f + tab.gx[i]) * tab.stride[i];
                const float y = (sy * 2.f - 0.5f + tab.gy[i]) * tab.stride[i];
                const float tw = sw * 2.f;
                const float th = sh * 2.f;
                const float w = tw * tw * tab.aw[i];
                const float h = th * th * tab.ah[i];
                d.x1 = x - w * 0.5f; d.y1 = y - h * 0.5f;
                d.x2 = x + w * 0.5f; d.y2 = y + h * 0.5f;
                d.score = score; d.label = best;
                alive = true;
            }
        }
    }

    const unsigned ballot = __ballot_sync(0xffffffffu, alive);
    if (ballot == 0u) return;
    const int lane = threadIdx.x & 31;
    int base = 0;
    if (lane == 0) base = atomicAdd(count, __popc(ballot));
    base = __shfl_sync(0xffffffffu, base, 0);
    if (alive) {
        const int slot = base + __popc(ballot & ((1u << lane) - 1u));
        if (slot < cap) dets[slot] = d;  // cap 溢出保护
    }
}

void launch_decode(const float* pred_dev, const DeviceAnchorTable& tab,
                   int num_classes, float conf_thr,
                   Detection* dets_dev, int* count_dev, int cap,
                   cudaStream_t stream) {
    const float logit_thr = logit_threshold(conf_thr);
    const int block = 256;
    const int grid = (tab.n + block - 1) / block;
    decode_kernel<<<grid, block, 0, stream>>>(pred_dev, tab, num_classes,
                                              logit_thr, conf_thr,
                                              dets_dev, count_dev, cap);
}

__global__ void dets_to_f6_kernel(const Detection* __restrict__ dets,
                                  const int* __restrict__ count,
                                  float* __restrict__ out_f6,
                                  int* __restrict__ out_count, int cap) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) *out_count = min(*count, cap);
    if (i >= cap) return;
    const int m = min(*count, cap);
    float* o = out_f6 + (size_t)i * 6;
    if (i < m) {
        const Detection d = dets[i];
        o[0] = d.x1; o[1] = d.y1; o[2] = d.x2; o[3] = d.y2;
        o[4] = d.score; o[5] = (float)d.label;
    } else {
        #pragma unroll
        for (int k = 0; k < 6; ++k) o[k] = 0.f;
    }
}

void launch_dets_to_f6(const Detection* dets_dev, const int* count_dev,
                       float* out_f6, int* out_count, int cap,
                       cudaStream_t stream) {
    const int block = 256;
    const int grid = (cap + block - 1) / block;
    dets_to_f6_kernel<<<grid, block, 0, stream>>>(dets_dev, count_dev,
                                                  out_f6, out_count, cap);
}

}  // namespace detail
}  // namespace cuda
}  // namespace yoloop
