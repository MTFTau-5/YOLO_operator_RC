// 用合成 ncnn 图（MemoryData 常量输出）端到端校验 NcnnInfer 的形状探测与拷贝：
//   1) E2E box-major [N, 6] → 转置为 channel-major [6, N] → decode_e2e 结果正确
//   2) AfRaw 逐层 [4*reg_max+nc, h, w] × 3 → 网格表推导（stride 由 in-size/grid 推断）
//      + 逐 channel 拼接拷贝布局正确
//   3) AnchorRaw 逐层 [na*(5+nc), h, w] × 3（anchor-major）→ na 推导 +
//      [5+nc, N] 重排布局全量比对 + 反例（expected_n 不符 / 通道非整数倍）
//   4) AnchorRaw planted anchor → decode 坐标/类别正确
// 不需要真实模型文件；param/bin 在临时目录生成。
// 所有 pattern 值取 0.125 的倍数（fp16 精确表示），规避 ncnn fp16 存储的舍入。

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "decode_af_raw.h"
#include "decode_avx2.h"
#include "decode_e2e.h"
#include "head_types.h"
#include "ncnn_infer.h"

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

bool write_text(const std::string& path, const std::string& s) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    std::fwrite(s.data(), 1, s.size(), f);
    std::fclose(f);
    return true;
}

bool write_bin(const std::string& path, const std::vector<float>& v) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    std::fwrite(v.data(), sizeof(float), v.size(), f);
    std::fclose(f);
    return true;
}

// ---- 1) E2E：单 MemoryData 输出 [1, 300, 6]（box-major，须转置）----
void test_e2e(const std::string& dir) {
    const int n = 300;
    std::vector<float> data(n * 6);
    for (int j = 0; j < n; ++j) {
        float* b = data.data() + j * 6;
        b[0] = (float)(j % 64);          // x1
        b[1] = (float)((j * 3) % 64);    // y1
        b[2] = b[0] + 8.f;               // x2
        b[3] = b[1] + 8.f;               // y2
        b[4] = (j % 10 == 0) ? 0.875f : 0.125f;  // score：j 为 10 的倍数时过阈
        b[5] = (float)(j % 3);           // 类别
    }
    write_text(dir + "/e2e.param",
               "7767517\n2 3\n"
               "Input in0 0 1 in0\n"
               "MemoryData out0 0 1 out0 0=6 1=300 2=1\n");
    write_bin(dir + "/e2e.bin", data);

    InferConfig cfg;
    cfg.param_path = dir + "/e2e.param";
    cfg.model_path = dir + "/e2e.bin";
    cfg.head_names = {"out0"};
    cfg.head = HeadType::E2E;
    cfg.in_w = cfg.in_h = 64;  // 图不消费输入，尺寸任意
    cfg.gpu_index = -1;        // 纯 CPU，结果确定性

    NcnnInfer infer;
    CHECK(infer.load(cfg, 0), "e2e: load 失败");
    CHECK(infer.pred_n() == n, "e2e: pred_n != 300");
    CHECK(infer.c_total() == 6, "e2e: c_total != 6");

    std::vector<float> blob(64 * 64 * 3, 0.f);
    std::vector<float> pred((size_t)6 * n, -1.f);
    CHECK(infer.infer(blob.data(), pred.data()), "e2e: infer 失败");
    // 转置正确性：pred[c*N + j] == data[j*6 + c]
    bool layout_ok = true;
    for (int j = 0; j < n && layout_ok; ++j)
        for (int c = 0; c < 6; ++c)
            if (pred[(size_t)c * n + j] != data[j * 6 + c]) { layout_ok = false; break; }
    CHECK(layout_ok, "e2e: box-major 转置布局错误");

    auto dets = x86::decode_e2e(pred.data(), n, 0.25f);
    CHECK(dets.size() == 30, "e2e: decode 数量 != 30");
    bool det_ok = dets.size() == 30;
    for (size_t k = 0; k < dets.size() && det_ok; ++k) {
        int j = (int)k * 10;
        const float* b = data.data() + j * 6;
        det_ok = dets[k].x1 == b[0] && dets[k].y1 == b[1] && dets[k].x2 == b[2] &&
                 dets[k].y2 == b[3] && dets[k].score == b[4] && dets[k].label == j % 3;
    }
    CHECK(det_ok, "e2e: decode 结果与 golden 不符");
    std::printf("[test_head_layouts] e2e box-major %s\n", g_fail ? "FAIL" : "PASS");
}

// ---- 2) AfRaw：三层 MemoryData [67, 80/40/20, 80/40/20]，stride 推导 8/16/32 ----
void test_afraw(const std::string& dir) {
    const int nc = 3, rm = 16, C = 4 * rm + nc;
    const int gw[3] = {80, 40, 20};
    std::vector<float> all;
    size_t off[3] = {0, 0, 0};
    for (int l = 0; l < 3; ++l) {
        off[l] = all.size();
        size_t sz = (size_t)C * gw[l] * gw[l];
        for (size_t i = 0; i < sz; ++i)
            all.push_back((float)((int)((i * 7 + l * 13) % 16)) * 0.125f - 1.f);
    }
    write_text(dir + "/afraw.param",
               "7767517\n4 5\n"
               "Input in0 0 1 in0\n"
               "MemoryData m0 0 1 m0 0=80 1=80 2=67\n"
               "MemoryData m1 0 1 m1 0=40 1=40 2=67\n"
               "MemoryData m2 0 1 m2 0=20 1=20 2=67\n");
    write_bin(dir + "/afraw.bin", all);

    InferConfig cfg;
    cfg.param_path = dir + "/afraw.param";
    cfg.model_path = dir + "/afraw.bin";
    cfg.head_names = {"m0", "m1", "m2"};
    cfg.head = HeadType::AfRaw;
    cfg.num_classes = nc;
    cfg.reg_max = rm;
    cfg.in_w = cfg.in_h = 640;
    cfg.gpu_index = -1;

    NcnnInfer infer;
    CHECK(infer.load(cfg, 0), "afraw: load 失败");
    CHECK(infer.pred_n() == 8400, "afraw: pred_n != 8400");
    CHECK(infer.c_total() == C, "afraw: c_total != 67");

    // 网格表：stride 8/16/32 由 in-size/grid 推断，中心点 x+0.5
    const AnchorTable& g = infer.grid();
    CHECK(g.n == 8400, "afraw: grid.n != 8400");
    bool grid_ok = g.n == 8400 && g.stride[0] == 8.f && g.stride[6400] == 16.f &&
                   g.stride[8000] == 32.f && g.gx[1] == 1.5f && g.gy[80] == 1.5f;
    CHECK(grid_ok, "afraw: 网格表推导错误");

    std::vector<float> blob(640 * 640 * 3, 0.f);
    std::vector<float> pred((size_t)C * 8400, -1.f);
    CHECK(infer.infer(blob.data(), pred.data()), "afraw: infer 失败");
    // 逐层 channel-major 拼接：pred[c*8400 + 层内偏移 + i] == all[off[l] + c*gw*gw + i]
    bool layout_ok = true;
    int base = 0;
    for (int l = 0; l < 3 && layout_ok; ++l) {
        int n = gw[l] * gw[l];
        for (int c = 0; c < C && layout_ok; c += C - 1)  // 抽查首末通道
            for (int i = 0; i < n; i += 997)
                if (pred[(size_t)c * 8400 + base + i] != all[off[l] + (size_t)c * n + i]) {
                    layout_ok = false;
                    break;
                }
        base += n;
    }
    CHECK(layout_ok, "afraw: 逐层拼接布局错误");

    // decode 冒烟：网格表 + 数据自洽（数值 golden 由 case9 覆盖）
    auto dets = x86::decode_af_raw(pred.data(), g, nc, rm, 0.25f);
    CHECK(!dets.empty(), "afraw: decode 无输出（预期高 logit 占 7/16）");
    std::printf("[test_head_layouts] afraw 逐层探测 %s\n", g_fail ? "FAIL" : "PASS");
}

// ---- 3) AnchorRaw：三层 MemoryData [na*(5+nc), 80/40/20, 80/40/20]（anchor-major）----
// na=3, nc=2 → C=21/层，640 输入 → N=25200。校验：na 由通道数推导、
// 重排为 [5+nc, N] 且列序 (y,x,a) 与 build_anchor_table 一致、expected_n 与
// 通道整数倍校验能拦住错误配置。
void test_anchorraw(const std::string& dir) {
    const int na = 3, nc = 2, A = 5 + nc, C = na * A;  // 21
    const int gw[3] = {80, 40, 20};
    const int N = na * (80 * 80 + 40 * 40 + 20 * 20);  // 25200

    // pattern(l,cc,j) 对 (l,cc) 唯一；0.125 倍数且 |v*8|≤2048，fp16 精确表示
    auto pattern = [](int l, int cc, int j) {
        return (float)(l * 97 + cc * 4 - 40) + (float)(j % 32) * 0.125f;
    };
    std::vector<float> all;
    for (int l = 0; l < 3; ++l)
        for (int cc = 0; cc < C; ++cc)
            for (int j = 0; j < gw[l] * gw[l]; ++j) all.push_back(pattern(l, cc, j));
    write_text(dir + "/v5.param",
               "7767517\n4 5\n"
               "Input in0 0 1 in0\n"
               "MemoryData m0 0 1 m0 0=80 1=80 2=21\n"
               "MemoryData m1 0 1 m1 0=40 1=40 2=21\n"
               "MemoryData m2 0 1 m2 0=20 1=20 2=21\n");
    write_bin(dir + "/v5.bin", all);

    InferConfig cfg;
    cfg.param_path = dir + "/v5.param";
    cfg.model_path = dir + "/v5.bin";
    cfg.head_names = {"m0", "m1", "m2"};
    cfg.head = HeadType::AnchorRaw;
    cfg.num_classes = nc;
    cfg.in_w = cfg.in_h = 640;
    cfg.gpu_index = -1;

    NcnnInfer infer;
    CHECK(infer.load(cfg, N), "v5: load 失败");
    CHECK(infer.pred_n() == N, "v5: pred_n != 25200");
    CHECK(infer.c_total() == A, "v5: c_total != 7");

    {  // 反例：expected_n 不符 → 拒绝
        NcnnInfer bad;
        CHECK(!bad.load(cfg, N + 1), "v5: expected_n 不符未拦截");
    }
    {  // 反例：blob 通道 21 不是 (5+nc)=8 的整数倍（nc=3 配错）→ 拒绝
        NcnnInfer bad;
        InferConfig c2 = cfg;
        c2.num_classes = 3;
        CHECK(!bad.load(c2, 0), "v5: 通道非 na*(5+nc) 整数倍未拦截");
    }

    std::vector<float> blob(640 * 640 * 3, 0.f);
    std::vector<float> pred((size_t)A * N, -1.f);
    CHECK(infer.infer(blob.data(), pred.data()), "v5: infer 失败");

    // 重排布局全量校验：pred[k*N + base + j*na + a] == pattern(l, a*A+k, j)
    bool layout_ok = true;
    int base = 0;
    const float* src = all.data();
    for (int l = 0; l < 3 && layout_ok; ++l) {
        const int hw = gw[l] * gw[l];
        for (int cc = 0; cc < C && layout_ok; ++cc) {
            const int a = cc / A, k = cc % A;
            for (int j = 0; j < hw; ++j) {
                if (pred[(size_t)k * N + base + (size_t)j * na + a] != *src++) {
                    layout_ok = false;
                    break;
                }
            }
        }
        base += hw * na;
    }
    CHECK(layout_ok, "v5: anchor 维重排布局错误");
    std::printf("[test_head_layouts] anchorraw 重排 %s\n", g_fail ? "FAIL" : "PASS");
}

// ---- 4) AnchorRaw decode 冒烟：planted anchor 经重排后 decode 坐标/类别正确 ----
void test_anchorraw_decode(const std::string& dir) {
    const int na = 3, nc = 2, A = 5 + nc, C = na * A;
    const int gw[3] = {80, 40, 20};
    const int hw_l[3] = {6400, 1600, 400};
    const int N = na * (hw_l[0] + hw_l[1] + hw_l[2]);

    // 全 -8（obj/cls 低 logit）；planted 点 tx..th=0、obj=8、cls[label]=8
    std::vector<float> all((size_t)C * (hw_l[0] + hw_l[1] + hw_l[2]), -8.f);
    const size_t off[3] = {0, (size_t)C * hw_l[0], (size_t)C * (hw_l[0] + hw_l[1])};
    struct Plant { int l, y, x, a, label; };
    const Plant plants[3] = {{0, 10, 20, 1, 0}, {1, 5, 30, 2, 1}, {2, 3, 7, 0, 1}};
    for (const auto& p : plants) {
        const int j = p.y * gw[p.l] + p.x;
        auto set = [&](int k, float v) {
            all[off[p.l] + (size_t)(p.a * A + k) * hw_l[p.l] + j] = v;
        };
        set(0, 0.f); set(1, 0.f); set(2, 0.f); set(3, 0.f);  // tx,ty,tw,th
        set(4, 8.f);                                          // obj
        set(5 + p.label, 8.f);                                // cls
    }
    write_text(dir + "/v5d.param",
               "7767517\n4 5\n"
               "Input in0 0 1 in0\n"
               "MemoryData m0 0 1 m0 0=80 1=80 2=21\n"
               "MemoryData m1 0 1 m1 0=40 1=40 2=21\n"
               "MemoryData m2 0 1 m2 0=20 1=20 2=21\n");
    write_bin(dir + "/v5d.bin", all);

    InferConfig cfg;
    cfg.param_path = dir + "/v5d.param";
    cfg.model_path = dir + "/v5d.bin";
    cfg.head_names = {"m0", "m1", "m2"};
    cfg.head = HeadType::AnchorRaw;
    cfg.num_classes = nc;
    cfg.in_w = cfg.in_h = 640;
    cfg.gpu_index = -1;

    NcnnInfer infer;
    CHECK(infer.load(cfg, N), "v5d: load 失败");
    std::vector<float> blob(640 * 640 * 3, 0.f);
    std::vector<float> pred((size_t)A * N, -1.f);
    CHECK(infer.infer(blob.data(), pred.data()), "v5d: infer 失败");

    AnchorTable tab = build_anchor_table(default_levels(640, 640));
    auto dets = x86::decode(pred.data(), tab, nc, 0.25f);
    CHECK(dets.size() == 3, "v5d: decode 数量 != 3");
    // 列 col = base + (y*gw+x)*na + a；σ(0)=0.5 → cx=(0.5+gx)*s，w=(0.5*2)²*aw=aw
    bool det_ok = dets.size() == 3;
    int base = 0;
    for (const auto& p : plants) {
        const int col = base + (p.y * gw[p.l] + p.x) * na + p.a;
        const float cx = (0.5f + tab.gx[col]) * tab.stride[col];
        const float cy = (0.5f + tab.gy[col]) * tab.stride[col];
        const float w = tab.aw[col], h = tab.ah[col];
        bool found = false;
        for (const auto& d : dets)
            if (d.label == p.label && d.score > 0.9f &&
                std::fabs(d.x1 - (cx - w * 0.5f)) < 1e-3f &&
                std::fabs(d.y1 - (cy - h * 0.5f)) < 1e-3f &&
                std::fabs(d.x2 - (cx + w * 0.5f)) < 1e-3f &&
                std::fabs(d.y2 - (cy + h * 0.5f)) < 1e-3f)
                found = true;
        det_ok = det_ok && found;
        base += hw_l[p.l] * na;
    }
    CHECK(det_ok, "v5d: decode 结果与 planted anchor 不符");
    std::printf("[test_head_layouts] anchorraw decode 冒烟 %s\n", g_fail ? "FAIL" : "PASS");
}

}  // namespace

int main() {
    char tmpl[] = "/tmp/yoloop_head_layouts_XXXXXX";
    if (!mkdtemp(tmpl)) {
        std::fprintf(stderr, "mkdtemp 失败\n");
        return 1;
    }
    const std::string dir = tmpl;
    test_e2e(dir);
    test_afraw(dir);
    test_anchorraw(dir);
    test_anchorraw_decode(dir);
    std::printf("%s\n", g_fail ? "FAILED" : "ALL PASS");
    return g_fail ? 1 : 0;
}
