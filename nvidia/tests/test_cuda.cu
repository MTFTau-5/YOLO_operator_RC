#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../common/testio.h"
#include "../common/yolo_ref.h"
#include "yolo_ops.h"

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err_ = (call);                                           \
        if (err_ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error: %s at %s:%d\n",                     \
                    cudaGetErrorString(err_), __FILE__, __LINE__);           \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

using namespace yoloop;

namespace {

int greedy_match(const std::vector<Detection>& got, const std::vector<Detection>& gold,
                 float coord_tol, float score_tol,
                 float& max_coord_err, float& max_score_err) {
    max_coord_err = 0.f;
    max_score_err = 0.f;
    if (got.size() != gold.size()) return -1;  // 数量不等直接失败
    std::vector<char> used(gold.size(), 0);
    int unmatched = 0;
    for (const auto& g : got) {
        bool found = false;
        for (size_t j = 0; j < gold.size(); ++j) {
            if (used[j] || gold[j].label != g.label) continue;
            const float ce = std::max(std::max(fabsf(g.x1 - gold[j].x1), fabsf(g.y1 - gold[j].y1)),
                                      std::max(fabsf(g.x2 - gold[j].x2), fabsf(g.y2 - gold[j].y2)));
            const float se = fabsf(g.score - gold[j].score);
            if (ce < coord_tol && se < score_tol) {
                used[j] = 1;
                found = true;
                max_coord_err = std::max(max_coord_err, ce);
                max_score_err = std::max(max_score_err, se);
                break;
            }
        }
        if (!found) ++unmatched;
    }
    return unmatched;
}

struct DevBuf {
    void* p = nullptr;
    DevBuf() = default;
    explicit DevBuf(size_t bytes) { CUDA_CHECK(cudaMalloc(&p, bytes)); }
    ~DevBuf() { if (p) cudaFree(p); }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
};

bool run_decode_case(const std::string& root, const char* name) {
    const std::string dir = root + "/" + name;
    const auto meta = testio::load_meta(dir);
    const int N = testio::meta_int(meta, "num_anchors");
    const int nc = testio::meta_int(meta, "num_classes");
    const float conf_thr = testio::meta_float(meta, "conf_thr");
    const float nms_thr = testio::meta_float(meta, "nms_thr");
    AnchorTable tab = build_anchor_table(default_levels(
        testio::meta_int(meta, "input_w"), testio::meta_int(meta, "input_h")));
    if (tab.n != N) {
        printf("  anchor table mismatch: built %d, meta %d\n", tab.n, N);
        return false;
    }
    const std::vector<float> pred = testio::load_f32(dir + "/pred.bin");
    const std::vector<Detection> gold_dec = testio::load_dets(dir + "/golden_decode.bin");
    const std::vector<Detection> gold_nms = testio::load_dets(dir + "/golden_nms.bin");

    bool ok = true;
    float ce, se;

    const std::vector<Detection> ref_dec = ref::decode(pred.data(), tab, nc, conf_thr);
    int un = greedy_match(ref_dec, gold_dec, 1e-2f, 1e-3f, ce, se);
    printf("  ref  decode vs golden: n=%zu gold=%zu %s max_coord_err=%.3e max_score_err=%.3e\n",
           ref_dec.size(), gold_dec.size(), un == 0 ? "match" : "MISMATCH", ce, se);
    ok &= (un == 0);

    DevBuf pred_dev(pred.size() * sizeof(float));
    DevBuf dets_dev((size_t)N * sizeof(Detection));
    DevBuf cnt_dev(sizeof(int));
    CUDA_CHECK(cudaMemcpy(pred_dev.p, pred.data(), pred.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    cuda::CudaAnchorTable dev_tab(tab);
    cuda::decode((const float*)pred_dev.p, dev_tab.get(), nc, conf_thr,
                 (Detection*)dets_dev.p, (int*)cnt_dev.p, N);
    int cnt = 0;
    CUDA_CHECK(cudaMemcpy(&cnt, cnt_dev.p, sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<Detection> cuda_dec((size_t)std::min(cnt, N));
    CUDA_CHECK(cudaMemcpy(cuda_dec.data(), dets_dev.p,
                          cuda_dec.size() * sizeof(Detection), cudaMemcpyDeviceToHost));
    un = greedy_match(cuda_dec, gold_dec, 1e-2f, 1e-3f, ce, se);
    printf("  cuda decode vs golden: n=%d gold=%zu %s max_coord_err=%.3e max_score_err=%.3e\n",
           cnt, gold_dec.size(), un == 0 ? "match" : "MISMATCH", ce, se);
    ok &= (un == 0);

    const std::vector<Detection> ref_nms = ref::nms(ref_dec, nms_thr);
    un = greedy_match(ref_nms, gold_nms, 1e-2f, 1e-3f, ce, se);
    printf("  ref  nms    vs golden: n=%zu gold=%zu %s\n",
           ref_nms.size(), gold_nms.size(), un == 0 ? "match" : "MISMATCH");
    ok &= (un == 0);

    DevBuf keep_dev(cuda_dec.size() * sizeof(Detection));
    cuda::nms((const Detection*)dets_dev.p, (int)cuda_dec.size(), nms_thr,
              (Detection*)keep_dev.p, (int*)cnt_dev.p);
    int kcnt = 0;
    CUDA_CHECK(cudaMemcpy(&kcnt, cnt_dev.p, sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<Detection> cuda_nms((size_t)kcnt);
    CUDA_CHECK(cudaMemcpy(cuda_nms.data(), keep_dev.p,
                          cuda_nms.size() * sizeof(Detection), cudaMemcpyDeviceToHost));
    un = greedy_match(cuda_nms, gold_nms, 1e-2f, 1e-3f, ce, se);
    printf("  cuda nms(chained) vs golden: n=%d gold=%zu %s\n",
           kcnt, gold_nms.size(), un == 0 ? "match" : "MISMATCH");
    ok &= (un == 0);

    DevBuf gin_dev(gold_dec.size() * sizeof(Detection));
    CUDA_CHECK(cudaMemcpy(gin_dev.p, gold_dec.data(),
                          gold_dec.size() * sizeof(Detection), cudaMemcpyHostToDevice));
    cuda::nms((const Detection*)gin_dev.p, (int)gold_dec.size(), nms_thr,
              (Detection*)keep_dev.p, (int*)cnt_dev.p);
    CUDA_CHECK(cudaMemcpy(&kcnt, cnt_dev.p, sizeof(int), cudaMemcpyDeviceToHost));
    cuda_nms.resize((size_t)kcnt);
    CUDA_CHECK(cudaMemcpy(cuda_nms.data(), keep_dev.p,
                          cuda_nms.size() * sizeof(Detection), cudaMemcpyDeviceToHost));
    un = greedy_match(cuda_nms, gold_nms, 1e-2f, 1e-3f, ce, se);
    printf("  cuda nms(golden in) vs golden: n=%d gold=%zu %s\n",
           kcnt, gold_nms.size(), un == 0 ? "match" : "MISMATCH");
    ok &= (un == 0);
    return ok;
}

bool run_preprocess_case(const std::string& root, const char* name) {
    const std::string dir = root + "/" + name;
    const auto meta = testio::load_meta(dir);
    const int sw = testio::meta_int(meta, "img_w"), sh = testio::meta_int(meta, "img_h");
    const int dw = testio::meta_int(meta, "dst_w"), dh = testio::meta_int(meta, "dst_h");
    const std::vector<uint8_t> img = testio::load_u8(dir + "/img.bin");
    const std::vector<float> gold = testio::load_f32(dir + "/golden_pre.bin");

    DevBuf img_dev(img.size());
    DevBuf out_dev(gold.size() * sizeof(float));
    CUDA_CHECK(cudaMemcpy(img_dev.p, img.data(), img.size(), cudaMemcpyHostToDevice));
    cuda::letterbox_normalize((const uint8_t*)img_dev.p, sw, sh,
                              (float*)out_dev.p, dw, dh);
    std::vector<float> out(gold.size());
    CUDA_CHECK(cudaMemcpy(out.data(), out_dev.p, gold.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    float maxerr = 0.f;
    for (size_t i = 0; i < out.size(); ++i) maxerr = std::max(maxerr, fabsf(out[i] - gold[i]));
    printf("  max_abs_err=%.3e (tol 1e-3) %s\n", maxerr, maxerr < 1e-3f ? "" : "FAIL");
    return maxerr < 1e-3f;
}

template <typename F>
float bench(F&& fn, int iters = 100) {
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    for (int i = 0; i < 5; ++i) fn();  // warmup
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < iters; ++i) fn();
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return ms / iters;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <test_data dir>\n", argv[0]);
        return 1;
    }
    const std::string root = argv[1];
    int fails = 0;

    // ---- decode + nms cases ----
    const char* dec_cases[] = {"case1_sparse", "case2_dense", "case3_few_classes"};
    for (const char* name : dec_cases) {
        printf("[%s]\n", name);
        const bool ok = run_decode_case(root, name);
        printf("  => %s\n\n", ok ? "PASS" : "FAIL");
        if (!ok) ++fails;
    }

    const char* pre_cases[] = {"case4_preprocess", "case5_preprocess_odd"};
    for (const char* name : pre_cases) {
        printf("[%s]\n", name);
        const bool ok = run_preprocess_case(root, name);
        printf("  => %s\n\n", ok ? "PASS" : "FAIL");
        if (!ok) ++fails;
    }

    {
        const std::string dir = root + "/case2_dense";
        const auto meta = testio::load_meta(dir);
        const int N = testio::meta_int(meta, "num_anchors");
        const int nc = testio::meta_int(meta, "num_classes");
        const float conf_thr = testio::meta_float(meta, "conf_thr");
        const float nms_thr = testio::meta_float(meta, "nms_thr");
        const std::vector<float> pred = testio::load_f32(dir + "/pred.bin");
        AnchorTable tab = build_anchor_table(default_levels(640, 640));
        cuda::CudaAnchorTable dev_tab(tab);

        DevBuf pred_dev(pred.size() * sizeof(float));
        DevBuf dets_dev((size_t)N * sizeof(Detection));
        DevBuf cnt_dev(sizeof(int));
        CUDA_CHECK(cudaMemcpy(pred_dev.p, pred.data(), pred.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        const float dec_ms = bench([&] {
            cuda::decode((const float*)pred_dev.p, dev_tab.get(), nc, conf_thr,
                         (Detection*)dets_dev.p, (int*)cnt_dev.p, N);
        });
        printf("[benchmark] decode case2 (N=%d): %.4f ms\n", N, dec_ms);

        const std::vector<Detection> cands = testio::load_dets(dir + "/golden_decode.bin");
        DevBuf cand_dev(cands.size() * sizeof(Detection));
        DevBuf keep_dev(cands.size() * sizeof(Detection));
        CUDA_CHECK(cudaMemcpy(cand_dev.p, cands.data(),
                              cands.size() * sizeof(Detection), cudaMemcpyHostToDevice));
        const float nms_ms = bench([&] {
            cuda::nms((const Detection*)cand_dev.p, (int)cands.size(), nms_thr,
                      (Detection*)keep_dev.p, (int*)cnt_dev.p);
        });
        printf("[benchmark] nms (%zu candidates): %.4f ms\n", cands.size(), nms_ms);
    }
    {
        const std::string dir = root + "/case4_preprocess";
        const std::vector<uint8_t> img = testio::load_u8(dir + "/img.bin");
        DevBuf img_dev(img.size());
        DevBuf out_dev((size_t)3 * 640 * 640 * sizeof(float));
        CUDA_CHECK(cudaMemcpy(img_dev.p, img.data(), img.size(), cudaMemcpyHostToDevice));
        const float pre_ms = bench([&] {
            cuda::letterbox_normalize((const uint8_t*)img_dev.p, 1280, 720,
                                      (float*)out_dev.p, 640, 640);
        });
        printf("[benchmark] preprocess 1280x720 -> 640x640: %.4f ms\n", pre_ms);
    }

    printf(fails ? "==== FAIL (%d case(s)) ====\n" : "==== ALL PASS ====\n", fails);
    return fails ? 1 : 0;
}
