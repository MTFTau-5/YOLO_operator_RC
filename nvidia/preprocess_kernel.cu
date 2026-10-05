#include "kernel_launch.h"

namespace yoloop {
namespace cuda {
namespace detail {

__global__ void letterbox_kernel(const uint8_t* __restrict__ src,
                                 int src_w, int src_h,
                                 float* __restrict__ out,
                                 int dst_w, int dst_h,
                                 float scale, int new_w, int new_h,
                                 int pad_x, int pad_y) {
    const int dx = blockIdx.x * blockDim.x + threadIdx.x;
    const int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_w || dy >= dst_h) return;

    const int plane = dst_w * dst_h;
    const size_t o = (size_t)dy * dst_w + dx;
    const int rx = dx - pad_x;  // resized 区域内坐标
    const int ry = dy - pad_y;

    if (rx >= 0 && rx < new_w && ry >= 0 && ry < new_h) {
        const float sy = (ry + 0.5f) / scale - 0.5f;
        int y0 = (int)floorf(sy);
        const float fy = sy - y0;
        int y1 = y0 + 1;
        y0 = max(0, min(y0, src_h - 1));
        y1 = max(0, min(y1, src_h - 1));
        const float sx = (rx + 0.5f) / scale - 0.5f;
        int x0 = (int)floorf(sx);
        const float fx = sx - x0;
        int x1 = x0 + 1;
        x0 = max(0, min(x0, src_w - 1));
        x1 = max(0, min(x1, src_w - 1));

        const uint8_t* p00 = src + ((size_t)y0 * src_w + x0) * 3;
        const uint8_t* p01 = src + ((size_t)y0 * src_w + x1) * 3;
        const uint8_t* p10 = src + ((size_t)y1 * src_w + x0) * 3;
        const uint8_t* p11 = src + ((size_t)y1 * src_w + x1) * 3;
        const float w0 = 1.f - fx, w1 = 1.f - fy;
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
            const float v00 = (float)p00[c], v01 = (float)p01[c];
            const float v10 = (float)p10[c], v11 = (float)p11[c];
            const float v = (v00 * w0 + v01 * fx) * w1 + (v10 * w0 + v11 * fx) * fy;
            out[(size_t)c * plane + o] = v / 255.f;
        }
    } else {
        const float pad_val = 114.f / 255.f;
        out[o] = pad_val;
        out[(size_t)plane + o] = pad_val;
        out[(size_t)2 * plane + o] = pad_val;
    }
}

void launch_letterbox(const uint8_t* src_dev, int src_w, int src_h,
                      float* out_dev, int dst_w, int dst_h,
                      cudaStream_t stream) {
    const float scale = fminf((float)dst_w / src_w, (float)dst_h / src_h);
    const int new_w = (int)roundf(src_w * scale);
    const int new_h = (int)roundf(src_h * scale);
    const int pad_x = (dst_w - new_w) / 2;
    const int pad_y = (dst_h - new_h) / 2;
    const dim3 block(16, 16);
    const dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);
    letterbox_kernel<<<grid, block, 0, stream>>>(src_dev, src_w, src_h, out_dev,
                                                 dst_w, dst_h, scale,
                                                 new_w, new_h, pad_x, pad_y);
}

}  // namespace detail
}  // namespace cuda
}  // namespace yoloop
