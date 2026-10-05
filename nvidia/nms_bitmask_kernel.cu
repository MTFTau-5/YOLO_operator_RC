// 位掩码迭代 NMS (README §2.3, mmcv / TensorRT batchedNMSPlugin 风格)
//   1. fill kernel: key = -score (降序 stable 排序后并列保持原顺序,
//      与 CPU ref::nms 的 std::stable_sort 语义一致), value = 原索引
//   2. thrust::stable_sort_by_key 设备端排序 + gather 出有序 Detection
//   3. pairwise kernel: mask[i * col_blocks + b] 的第 t 位置位 <=>
//      box (b*64+t) 与 box i 同类且 IoU > nms_thr (只算 row_block <= col_block)
//   4. walk kernel: 单 block 按序逐保留框, 把其 mask 行 OR 进 remv 位图,
//      抑制后续同类且 IoU>nms_thr 的框, 输出紧凑保留列表
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/sort.h>

#include <stdexcept>

#include "kernel_launch.h"

#define YOLOOP_CUDA_CHECK(call)                                              \
    do {                                                                     \
        cudaError_t err_ = (call);                                           \
        if (err_ != cudaSuccess)                                             \
            throw std::runtime_error(std::string("CUDA error: ") +           \
                                     cudaGetErrorString(err_) + " at " +     \
                                     __FILE__ + ":" + std::to_string(__LINE__)); \
    } while (0)

namespace yoloop {
namespace cuda {
namespace detail {

__device__ __forceinline__ float iou_dev(const Detection& a, const Detection& b) {
    const float iw = fmaxf(fminf(a.x2, b.x2) - fmaxf(a.x1, b.x1), 0.f);
    const float ih = fmaxf(fminf(a.y2, b.y2) - fmaxf(a.y1, b.y1), 0.f);
    const float inter = iw * ih;
    const float ua = (a.x2 - a.x1) * (a.y2 - a.y1) +
                     (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
    return ua > 0.f ? inter / ua : 0.f;
}

__global__ void fill_keys_kernel(const Detection* __restrict__ dets, int n,
                                 float* __restrict__ keys, int* __restrict__ idx) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        keys[i] = -dets[i].score;  // 升序排 -score == 降序排 score
        idx[i] = i;
    }
}

__global__ void gather_kernel(const Detection* __restrict__ dets,
                              const int* __restrict__ idx, int n,
                              Detection* __restrict__ sorted) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) sorted[i] = dets[idx[i]];
}

__global__ void nms_pairwise_kernel(const Detection* __restrict__ sorted, int n,
                                    float nms_thr,
                                    unsigned long long* __restrict__ mask,
                                    int col_blocks) {
    const int col_start = blockIdx.x;
    const int row_start = blockIdx.y;
    if (row_start > col_start) return;  // 只需上三角(含对角)块
    const int tid = threadIdx.x;        // 64 线程
    __shared__ Detection col_boxes[64];
    const int row_size = min(n - row_start * 64, 64);
    const int col_size = min(n - col_start * 64, 64);
    if (tid < col_size) col_boxes[tid] = sorted[col_start * 64 + tid];
    __syncthreads();
    if (tid < row_size) {
        const int i = row_start * 64 + tid;
        const Detection a = sorted[i];
        unsigned long long m = 0ULL;
        for (int j = 0; j < col_size; ++j) {
            const Detection& b = col_boxes[j];
            if (b.label == a.label && iou_dev(a, b) > nms_thr) m |= 1ULL << j;
        }
        mask[(size_t)i * col_blocks + col_start] = m;
    }
}

__global__ void nms_walk_kernel(const unsigned long long* __restrict__ mask,
                                const Detection* __restrict__ sorted, int n,
                                Detection* __restrict__ keep,
                                int* __restrict__ count) {
    // 单 warp 顺序扫描: lane 0 判定保留, __shfl 广播, warp 并行把该框的掩码行
    // OR 进 remv 位图, 抑制后续同类且 IoU>nms_thr 的框
    extern __shared__ unsigned long long remv[];  // col_blocks 个 uint64
    const int col_blocks = (n + 63) >> 6;
    const int lane = threadIdx.x;  // blockDim.x = 32
    for (int j = lane; j < col_blocks; j += 32) remv[j] = 0ULL;
    __syncwarp();
    int kept = 0;  // 仅 lane 0 维护
    if (col_blocks <= 32) {
        // 快路径: 每 lane 持行掩码的一个 word, 寄存器流水预取下一行掩码与
        // 下一个 Detection, 掩盖逐保留框的全局内存延迟
        unsigned long long cur = 0;
        if (lane < col_blocks) cur = mask[lane];  // 第 0 行, nb=0
        Detection cur_det;
        if (lane == 0) cur_det = sorted[0];
        for (int i = 0; i < n; ++i) {
            const int nb = i >> 6, ib = i & 63;
            int do_keep = 0;
            if (lane == 0) do_keep = ((remv[nb] >> ib) & 1ULL) == 0ULL;
            do_keep = __shfl_sync(0xffffffffu, do_keep, 0);
            unsigned long long nxt = 0;
            Detection nxt_det;
            if (i + 1 < n) {
                const int j1 = ((i + 1) >> 6) + lane;
                if (j1 < col_blocks) nxt = mask[(size_t)(i + 1) * col_blocks + j1];
                if (lane == 0) nxt_det = sorted[i + 1];
            }
            if (do_keep) {
                if (lane == 0) { keep[kept] = cur_det; ++kept; }
                const int j = nb + lane;
                if (j < col_blocks) remv[j] |= cur;
            }
            cur = nxt;
            cur_det = nxt_det;
            __syncwarp();
        }
    } else {
        for (int i = 0; i < n; ++i) {
            const int nb = i >> 6, ib = i & 63;
            const Detection di = sorted[i];  // 无条件加载, 与判定重叠
            int do_keep = 0;
            if (lane == 0) do_keep = ((remv[nb] >> ib) & 1ULL) == 0ULL;
            do_keep = __shfl_sync(0xffffffffu, do_keep, 0);
            if (do_keep) {
                if (lane == 0) { keep[kept] = di; ++kept; }
                const unsigned long long* row = mask + (size_t)i * col_blocks;
                for (int j = nb + lane; j < col_blocks; j += 32) remv[j] |= row[j];
            }
            __syncwarp();
        }
    }
    if (lane == 0) *count = kept;
}

void launch_nms(const Detection* dets_dev, int n, float nms_thr,
                Detection* keep_dev, int* count_dev,
                cudaStream_t stream) {
    if (n <= 0) {
        YOLOOP_CUDA_CHECK(cudaMemsetAsync(count_dev, 0, sizeof(int), stream));
        return;
    }
    const int col_blocks = (n + 63) / 64;
    float* keys = nullptr;
    int* idx = nullptr;
    Detection* sorted = nullptr;
    unsigned long long* mask = nullptr;
    YOLOOP_CUDA_CHECK(cudaMallocAsync(&keys, (size_t)n * sizeof(float), stream));
    YOLOOP_CUDA_CHECK(cudaMallocAsync(&idx, (size_t)n * sizeof(int), stream));
    YOLOOP_CUDA_CHECK(cudaMallocAsync(&sorted, (size_t)n * sizeof(Detection), stream));
    YOLOOP_CUDA_CHECK(cudaMallocAsync(&mask, (size_t)n * col_blocks * sizeof(unsigned long long), stream));

    const int block = 256;
    const int grid = (n + block - 1) / block;
    fill_keys_kernel<<<grid, block, 0, stream>>>(dets_dev, n, keys, idx);
    thrust::stable_sort_by_key(thrust::cuda::par.on(stream), keys, keys + n,
                               idx, thrust::less<float>());
    gather_kernel<<<grid, block, 0, stream>>>(dets_dev, idx, n, sorted);

    YOLOOP_CUDA_CHECK(cudaMemsetAsync(mask, 0, (size_t)n * col_blocks * sizeof(unsigned long long), stream));
    const dim3 pgrid(col_blocks, col_blocks);
    nms_pairwise_kernel<<<pgrid, 64, 0, stream>>>(sorted, n, nms_thr, mask, col_blocks);

    const size_t shmem = (size_t)col_blocks * sizeof(unsigned long long);
    if (shmem > 48 * 1024) {
        YOLOOP_CUDA_CHECK(cudaFuncSetAttribute(nms_walk_kernel,
                          cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem));
    }
    nms_walk_kernel<<<1, 32, shmem, stream>>>(mask, sorted, n, keep_dev, count_dev);

    YOLOOP_CUDA_CHECK(cudaFreeAsync(keys, stream));
    YOLOOP_CUDA_CHECK(cudaFreeAsync(idx, stream));
    YOLOOP_CUDA_CHECK(cudaFreeAsync(sorted, stream));
    YOLOOP_CUDA_CHECK(cudaFreeAsync(mask, stream));
}

}  // namespace detail
}  // namespace cuda
}  // namespace yoloop

#undef YOLOOP_CUDA_CHECK
