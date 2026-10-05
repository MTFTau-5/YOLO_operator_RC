#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "../common/testio.h"
#include "../common/yolo_ref.h"
#include "../decode_avx2.h"
#include "../nms_simd.h"
#include "../preprocess_avx2.h"

namespace {

using yoloop::Detection;

volatile size_t g_sink = 0;

struct MatchResult {
    bool ok = true;
    double max_coord_err = 0.0;
    double max_score_err = 0.0;
    size_t unmatched = 0;
};

double coord_err(const Detection& a, const Detection& b) {
    return std::max(std::max((double)std::fabs(a.x1 - b.x1), (double)std::fabs(a.y1 - b.y1)),
                    std::max((double)std::fabs(a.x2 - b.x2), (double)std::fabs(a.y2 - b.y2)));
}

MatchResult match_sets(const std::vector<Detection>& got,
                       const std::vector<Detection>& golden,
                       double coord_tol, double score_tol) {
    MatchResult r;
    if (got.size() != golden.size()) r.ok = false;
    std::vector<char> used(got.size(), 0);
    for (const auto& g : golden) {
        int best = -1;
        double best_cost = 1e300;
        for (size_t i = 0; i < got.size(); ++i) {
            if (used[i] || got[i].label != g.label) continue;
            double dc = coord_err(got[i], g);
            double ds = (double)std::fabs(got[i].score - g.score);
            if (dc < coord_tol && ds < score_tol && dc + ds < best_cost) {
                best_cost = dc + ds;
                best = (int)i;
            }
        }
        if (best < 0) {
            r.ok = false;
            ++r.unmatched;
        } else {
            used[(size_t)best] = 1;
            r.max_coord_err = std::max(r.max_coord_err, coord_err(got[(size_t)best], g));
            r.max_score_err = std::max(r.max_score_err,
                                       (double)std::fabs(got[(size_t)best].score - g.score));
        }
    }
    return r;
}

template <typename F>
double bench_us(F&& f, int rounds) {
    for (int i = 0; i < 3; ++i) f();  // warmup
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < rounds; ++i) f();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / (double)rounds;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <test_data 目录>\n", argv[0]);
        return 2;
    }
    const std::string root = argv[1];
    bool all_ok = true;
    double bench_decode = 0.0, bench_nms = 0.0, bench_pre = 0.0;

    const char* decode_cases[] = {"case1_sparse", "case2_dense", "case3_few_classes"};
    for (const char* name : decode_cases) {
        const std::string dir = root + "/" + name;
        auto meta = yoloop::testio::load_meta(dir);
        const int N = yoloop::testio::meta_int(meta, "num_anchors");
        const int nc = yoloop::testio::meta_int(meta, "num_classes");
        const float conf = yoloop::testio::meta_float(meta, "conf_thr");
        const float nth = yoloop::testio::meta_float(meta, "nms_thr");
        const int iw = yoloop::testio::meta_int(meta, "input_w");
        const int ih = yoloop::testio::meta_int(meta, "input_h");

        auto pred = yoloop::testio::load_f32(dir + "/pred.bin");
        auto gdec = yoloop::testio::load_dets(dir + "/golden_decode.bin");
        auto gnms = yoloop::testio::load_dets(dir + "/golden_nms.bin");
        if ((int)pred.size() != (5 + nc) * N) {
            std::fprintf(stderr, "[%s] pred.bin 尺寸不符: %zu != %d\n", name, pred.size(), (5 + nc) * N);
            return 1;
        }
        auto tab = yoloop::build_anchor_table(yoloop::default_levels(iw, ih));
        if (tab.n != N) {
            std::fprintf(stderr, "[%s] anchor 表不符: %d != %d\n", name, tab.n, N);
            return 1;
        }

        auto rdec = yoloop::ref::decode(pred.data(), tab, nc, conf);
        auto xdec = yoloop::x86::decode(pred.data(), tab, nc, conf);
        auto m_ref = match_sets(rdec, gdec, 1e-2, 1e-3);
        auto m_x86 = match_sets(xdec, gdec, 1e-2, 1e-3);
        bool ok = m_ref.ok && m_x86.ok;
        all_ok = all_ok && ok;
        std::printf("[%s] decode %s | golden=%zu ref=%zu x86=%zu | ref_err coord=%.3g score=%.3g"
                    " | x86_err coord=%.3g score=%.3g%s\n",
                    name, ok ? "PASS" : "FAIL", gdec.size(), rdec.size(), xdec.size(),
                    m_ref.max_coord_err, m_ref.max_score_err,
                    m_x86.max_coord_err, m_x86.max_score_err,
                    m_x86.unmatched ? " (unmatched!)" : "");

        auto rnms = yoloop::ref::nms(gdec, nth);
        auto xnms = yoloop::x86::nms(gdec, nth, /*topk=*/0);
        auto n_ref = match_sets(rnms, gnms, 1e-2, 1e-3);
        auto n_x86 = match_sets(xnms, gnms, 1e-2, 1e-3);
        bool nok = n_ref.ok && n_x86.ok;
        all_ok = all_ok && nok;
        std::printf("[%s] nms    %s | golden=%zu ref=%zu x86=%zu%s\n",
                    name, nok ? "PASS" : "FAIL", gnms.size(), rnms.size(), xnms.size(),
                    n_x86.unmatched ? " (unmatched!)" : "");

        if (std::string(name) == "case2_dense") {
            bench_decode = bench_us([&] {
                auto v = yoloop::x86::decode(pred.data(), tab, nc, conf);
                g_sink = v.size();
            }, 1000);
            bench_nms = bench_us([&] {
                auto v = yoloop::x86::nms(gdec, nth, /*topk=*/300);
                g_sink = v.size();
            }, 1000);
        }
    }

    const char* pre_cases[] = {"case4_preprocess", "case5_preprocess_odd"};
    for (const char* name : pre_cases) {
        const std::string dir = root + "/" + name;
        auto meta = yoloop::testio::load_meta(dir);
        const int iw = yoloop::testio::meta_int(meta, "img_w");
        const int ih = yoloop::testio::meta_int(meta, "img_h");
        const int dw = yoloop::testio::meta_int(meta, "dst_w");
        const int dh = yoloop::testio::meta_int(meta, "dst_h");

        auto img = yoloop::testio::load_u8(dir + "/img.bin");
        auto golden = yoloop::testio::load_f32(dir + "/golden_pre.bin");
        if (img.size() != (size_t)iw * (size_t)ih * 3u ||
            golden.size() != (size_t)3u * (size_t)dw * (size_t)dh) {
            std::fprintf(stderr, "[%s] 输入/golden 尺寸不符\n", name);
            return 1;
        }

        std::vector<float> out(golden.size());
        yoloop::x86::letterbox_normalize(img.data(), iw, ih, out.data(), dw, dh);
        double err_x86 = 0.0;
        for (size_t i = 0; i < out.size(); ++i)
            err_x86 = std::max(err_x86, (double)std::fabs(out[i] - golden[i]));

        std::vector<float> rout(golden.size());
        yoloop::ref::letterbox_normalize(img.data(), iw, ih, rout.data(), dw, dh);
        double err_ref = 0.0;
        for (size_t i = 0; i < rout.size(); ++i)
            err_ref = std::max(err_ref, (double)std::fabs(rout[i] - golden[i]));

        bool ok = err_x86 < 1e-3 && err_ref < 1e-3;
        all_ok = all_ok && ok;
        std::printf("[%s] preprocess %s | x86 max_abs_err=%.3g | ref max_abs_err=%.3g\n",
                    name, ok ? "PASS" : "FAIL", err_x86, err_ref);

        if (std::string(name) == "case4_preprocess") {
            bench_pre = bench_us([&] {
                yoloop::x86::letterbox_normalize(img.data(), iw, ih, out.data(), dw, dh);
                g_sink = (size_t)out[0];
            }, 100);
        }
    }

    std::printf("\n== 基准（平均耗时） ==\n");
    std::printf("decode      case2_dense N=25200     : %9.1f us  (1000 rounds)\n", bench_decode);
    std::printf("nms         case2_dense topk=300    : %9.1f us  (1000 rounds)\n", bench_nms);
    std::printf("preprocess  1280x720 -> 640x640     : %9.1f us  (100 rounds)\n", bench_pre);
    std::printf("\n%s\n", all_ok ? "ALL PASS" : "FAILED");
    return all_ok ? 0 : 1;
}
