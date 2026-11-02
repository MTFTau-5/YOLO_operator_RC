// 真实 ncnn raw 模型（tools/export_af_raw.py 产物）多尺寸动态验收。
//
// 对 case 目录（tools/accept_raw_dynamic.py 生成）里的每个尺寸：
//   同一 param/bin → NcnnInfer::load（N 由输出形状自动发现）→ infer
//   → 与 torch 逐层 raw golden 比对（布局 + 数值，fp 容差）
//   → decode_af_raw + nms → 与 python 参考 golden 框比对（数量 + 坐标容差）
// 用法: test_raw_model --case DIR --param P --model B [--heads out0,out1,out2]
//        [--gpu -1]
// case 目录缺失时打印 SKIP 并返回 0（验收数据由 accept_raw_dynamic.py 一键再生成）。

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "decode_af_raw.h"
#include "head_types.h"
#include "ncnn_infer.h"
#include "nms_simd.h"
#include "testio.h"

namespace {

using namespace yoloop;

int g_fail = 0;
#define CHECK(cond, msg)                                     \
    do {                                                     \
        if (!(cond)) {                                       \
            std::fprintf(stderr, "[FAIL] %s\n", msg);        \
            ++g_fail;                                        \
        }                                                    \
    } while (0)

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    size_t p = 0;
    while (true) {
        size_t q = s.find(sep, p);
        out.push_back(s.substr(p, q == std::string::npos ? q : q - p));
        if (q == std::string::npos) break;
        p = q + 1;
    }
    return out;
}

void run_size(const std::string& dir, const std::string& param, const std::string& model,
              const std::vector<std::string>& heads, int gpu, int w, int h,
              const std::map<std::string, std::string>& meta) {
    const int nc = testio::meta_int(meta, "num_classes");
    const int rm = testio::meta_int(meta, "reg_max");
    const float conf = testio::meta_float(meta, "conf_thr");
    const float nms_thr = testio::meta_float(meta, "nms_thr");
    const float tol_raw = testio::meta_float(meta, "tol_raw");
    const float tol_px = testio::meta_float(meta, "tol_px");
    const float tol_score = testio::meta_float(meta, "tol_score");
    const float margin = meta.count("margin") ? testio::meta_float(meta, "margin") : 0.05f;
    const std::string tag = std::to_string(w) + "x" + std::to_string(h);

    // golden：逐层 raw [C, hw]（torch flatten），列向拼接顺序即 pred 布局
    const int C = 4 * rm + nc;
    std::vector<float> golden_raw;
    int n_golden = 0;
    for (int l = 0; l < 3; ++l) {
        auto v = testio::load_f32(dir + "/raw_" + tag + "_l" + std::to_string(l) + ".bin");
        CHECK((int)v.size() % C == 0, "raw golden 通道不整除");
        n_golden += (int)v.size() / C;
        golden_raw.insert(golden_raw.end(), v.begin(), v.end());
    }
    auto blob = testio::load_f32(dir + "/blob_" + tag + ".bin");
    CHECK((int)blob.size() == 3 * w * h, "blob 尺寸不符");
    auto golden_dets = testio::load_dets(dir + "/golden_" + tag + ".bin");

    InferConfig cfg;
    cfg.param_path = param;
    cfg.model_path = model;
    cfg.head_names = heads;
    cfg.in_w = w;
    cfg.in_h = h;
    cfg.head = HeadType::AfRaw;
    cfg.num_classes = nc;
    cfg.reg_max = rm;
    cfg.gpu_index = gpu;

    NcnnInfer infer;
    CHECK(infer.load(cfg, 0), "load 失败");
    if (!infer.pred_n()) return;
    CHECK(infer.pred_n() == n_golden, "N 自动发现与 golden 网格不一致");
    CHECK(infer.c_total() == C, "c_total 不符");
    const AnchorTable& g = infer.grid();
    CHECK(g.n == n_golden, "网格表 N 不符");

    std::vector<float> pred((size_t)C * infer.pred_n(), -1.f);
    CHECK(infer.infer(blob.data(), pred.data()), "infer 失败");

    // raw 张量比对：golden 逐层 [C, hw_l]，pred 为 [C, N] 列向拼接；
    // 通道主序与列序一致，重组 golden 为 [C, N] 后逐元素比较。
    // 层边界取网格表 stride 值变化处（8→16→32 升序）。
    double max_d = 0.0;
    const int Cn = infer.pred_n();
    std::vector<int> bounds{0};
    for (int i = 1; i < g.n; ++i)
        if (g.stride[i] != g.stride[i - 1]) bounds.push_back(i);
    bounds.push_back(g.n);
    CHECK((int)bounds.size() == 4, "网格表层数 != 3");
    {
        std::vector<float> golden_cn((size_t)C * n_golden);
        size_t off_l = 0;
        for (int l = 0; l < 3; ++l) {
            const int base = bounds[l], n = bounds[l + 1] - bounds[l];
            for (int c = 0; c < C; ++c)
                std::memcpy(golden_cn.data() + (size_t)c * n_golden + base,
                            golden_raw.data() + off_l + (size_t)c * n,
                            (size_t)n * sizeof(float));
            off_l += (size_t)C * n;
        }
        for (size_t i = 0; i < (size_t)C * n_golden; ++i)
            max_d = std::max(max_d, (double)std::fabs(pred[i] - golden_cn[i]));
    }
    std::printf("[test_raw_model] %s: N=%d, raw max|Δ|=%.5f (tol %.5f)\n", tag.c_str(),
                infer.pred_n(), max_d, tol_raw);
    CHECK(max_d <= tol_raw, "raw 张量与 torch 不一致");

    auto dets = x86::decode_af_raw(pred.data(), g, nc, rm, conf);
    auto kept = x86::nms(std::move(dets), nms_thr, 300);
    // fp16/图融合会让贴阈值的边缘框抖动出没（score ∈ [conf, conf+margin)），
    // 硬性要求只对 score ≥ conf+margin 的框：双向都必须匹配；带内差异只报告。
    auto is_strict = [&](float s) { return s >= conf + margin; };
    std::vector<char> used(kept.size(), 0);
    double max_px = 0.0;
    int n_strict = 0, n_marginal = 0, n_matched = 0;
    bool det_ok = true;
    for (const auto& gd : golden_dets) {
        const bool strict = is_strict(gd.score);
        n_strict += strict;
        n_marginal += !strict;
        double best_px = 1e30;
        int best = -1;
        for (size_t i = 0; i < kept.size(); ++i) {
            if (used[i] || kept[i].label != gd.label) continue;
            double d = std::max(
                {std::fabs(kept[i].x1 - gd.x1), std::fabs(kept[i].y1 - gd.y1),
                 std::fabs(kept[i].x2 - gd.x2), std::fabs(kept[i].y2 - gd.y2)});
            if (d < best_px) { best_px = d; best = (int)i; }
        }
        if (best >= 0 && best_px <= tol_px &&
            std::fabs(kept[best].score - gd.score) <= tol_score) {
            used[best] = 1;
            ++n_matched;
            max_px = std::max(max_px, best_px);
        } else if (strict) {
            det_ok = false;
        }
    }
    int n_unmatched_strict = 0, n_unmatched_marginal = 0;
    for (size_t i = 0; i < kept.size(); ++i)
        if (!used[i]) (is_strict(kept[i].score) ? n_unmatched_strict
                                               : n_unmatched_marginal)++;
    if (n_unmatched_strict) det_ok = false;
    std::printf("[test_raw_model] %s: 框 kept=%zu golden=%zu（严格 %d，边缘带 %d）,"
                " 匹配 %d, 坐标 max|Δ|=%.4fpx (tol %.4f)%s\n",
                tag.c_str(), kept.size(), golden_dets.size(), n_strict, n_marginal,
                n_matched, max_px, tol_px, det_ok ? " OK" : " MISMATCH");
    CHECK(det_ok, "decode+nms 与 golden 不一致（严格框）");
}

}  // namespace

int main(int argc, char** argv) {
    std::string dir, param, model, only;
    std::vector<std::string> heads = {"out0", "out1", "out2"};
    int gpu = -1;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto val = [&](const char* what) {
            if (++i >= argc) { fprintf(stderr, "%s 缺参数\n", what); exit(2); }
            return std::string(argv[i]);
        };
        if (k == "--case") dir = val("--case");
        else if (k == "--param") param = val("--param");
        else if (k == "--model") model = val("--model");
        else if (k == "--heads") heads = split(val("--heads"), ',');
        else if (k == "--gpu") gpu = atoi(val("--gpu").c_str());
        else if (k == "--only-size") only = val("--only-size");  // 只跑该尺寸（逐尺寸导出模式）
        else { fprintf(stderr, "未知参数: %s\n", k.c_str()); return 2; }
    }
    if (dir.empty() || param.empty() || model.empty()) {
        fprintf(stderr, "用法: test_raw_model --case DIR --param P --model B "
                        "[--heads out0,out1,out2] [--gpu -1] [--only-size WxH]\n");
        return 2;
    }
    std::map<std::string, std::string> meta;
    try {
        meta = testio::load_meta(dir);
    } catch (...) {
        std::printf("[test_raw_model] SKIP: %s 不存在（先运行 "
                    "tools/accept_raw_dynamic.py 生成验收数据）\n", dir.c_str());
        return 0;
    }
    for (const auto& s : split(meta.at("sizes"), ',')) {
        if (!only.empty() && s != only) continue;
        int w = 0, h = 0;
        if (sscanf(s.c_str(), "%dx%d", &w, &h) != 2) {
            fprintf(stderr, "meta sizes 项无法解析: %s\n", s.c_str());
            return 2;
        }
        run_size(dir, param, model, heads, gpu, w, h, meta);
    }
    std::printf("[test_raw_model] %s\n", g_fail ? "FAILED" : "ALL PASS");
    return g_fail ? 1 : 0;
}
