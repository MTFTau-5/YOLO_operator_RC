#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "../common/testio.h"
#include "../common/yolo_ref.h"
#include "../x86/decode_avx2.h"
#include "../x86/nms_simd.h"
#include "../x86/preprocess_avx2.h"

using namespace yoloop;

template <typename F>
static double bench_ms(F&& f, int rounds) {
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < rounds; ++i) f();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / rounds;
}

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : "test_data";

    // ---- decode: case2_dense (N=25200, 密集场景) ----
    {
        std::string c = dir + "/case2_dense";
        auto meta = testio::load_meta(c);
        int nc = testio::meta_int(meta, "num_classes");
        int iw = testio::meta_int(meta, "input_w"), ih = testio::meta_int(meta, "input_h");
        float conf = testio::meta_float(meta, "conf_thr");
        auto pred = testio::load_f32(c + "/pred.bin");
        AnchorTable tab = build_anchor_table(default_levels(iw, ih));

        size_t nd = 0;
        double t_ref = bench_ms([&] {
            auto d = ref::decode(pred.data(), tab, nc, conf);
            nd = d.size();
        }, 100);
        double t_x86 = bench_ms([&] { volatile size_t n = x86::decode(pred.data(), tab, nc, conf).size(); (void)n; }, 1000);
        printf("decode      N=%d (%zu 存活)  ref(标量)=%8.3f ms   x86(AVX2)=%7.4f ms   加速 %6.1fx\n",
               tab.n, nd, t_ref, t_x86, t_ref / t_x86);
    }

    // ---- NMS: case2 golden_decode 输出(1930 候选) ----
    {
        std::string c = dir + "/case2_dense";
        auto meta = testio::load_meta(c);
        float nms_thr = testio::meta_float(meta, "nms_thr");
        auto dets = testio::load_dets(c + "/golden_decode.bin");

        double t_ref = bench_ms([&] { volatile size_t n = ref::nms(dets, nms_thr).size(); (void)n; }, 20);
        double t_x86 = bench_ms([&] { volatile size_t n = x86::nms(dets, nms_thr).size(); (void)n; }, 1000);
        printf("nms         %zu 候选       ref(标量)=%8.3f ms   x86(SIMD)=%7.4f ms   加速 %6.1fx\n",
               dets.size(), t_ref, t_x86, t_ref / t_x86);
    }

    // ---- preprocess: case4 1280x720 -> 640x640 ----
    {
        std::string c = dir + "/case4_preprocess";
        auto meta = testio::load_meta(c);
        int sw = testio::meta_int(meta, "img_w"), sh = testio::meta_int(meta, "img_h");
        int dw = testio::meta_int(meta, "dst_w"), dh = testio::meta_int(meta, "dst_h");
        auto img = testio::load_u8(c + "/img.bin");
        std::vector<float> out((size_t)3 * dw * dh);

        double t_ref = bench_ms([&] { ref::letterbox_normalize(img.data(), sw, sh, out.data(), dw, dh); }, 100);
        double t_x86 = bench_ms([&] { x86::letterbox_normalize(img.data(), sw, sh, out.data(), dw, dh); }, 1000);
        printf("preprocess  %dx%d->%dx%d ref(标量)=%8.3f ms   x86(AVX2)=%7.4f ms   加速 %6.1fx\n",
               sw, sh, dw, dh, t_ref, t_x86, t_ref / t_x86);
    }
    return 0;
}
