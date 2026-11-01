// 用合成 ncnn 图（MemoryData 常量输出）端到端校验 NcnnInfer 的形状探测与拷贝：
//   1) E2E box-major [N, 6] → 转置为 channel-major [6, N] → decode_e2e 结果正确
//   2) AfRaw 逐层 [4*reg_max+nc, h, w] × 3 → 网格表推导（stride 由 in-size/grid 推断）
//      + 逐 channel 拼接拷贝布局正确
// 不需要真实模型文件；param/bin 在临时目录生成。
// 所有 pattern 值取 0.125 的倍数（fp16 精确表示），规避 ncnn fp16 存储的舍入。

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "decode_af_raw.h"
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
    std::printf("%s\n", g_fail ? "FAILED" : "ALL PASS");
    return g_fail ? 1 : 0;
}
