// test_tracker —— Tracker 对 RK3588 BoxKalman/门控逻辑的数值对齐。
// 回放 tools/gen_tracker_golden.py 生成的 test_data/case6_tracker/：
//   dets.csv   每帧检测框（空帧无记录）
//   golden.csv 每帧 predict 框 / update 后输出框 / KF 内部状态（Python float64）
// 容差：坐标与状态误差 < 1e-3；并断言事件序列（INITIAL_LOCK/LOST/REACQUIRED）。
//
// 用法: ./test_tracker <test_data_dir>（参数为 test_data 根，内部进 case6_tracker）

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "tracker.h"

using namespace yoloop;

namespace {

constexpr int IMG_W = 1280, IMG_H = 720;
constexpr double TOL = 1e-3;

std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    for (char ch : line) {
        if (ch == ',') {
            out.push_back(cur);
            cur.clear();
        } else {
            cur += ch;
        }
    }
    out.push_back(cur);
    return out;
}

double fnum(const std::string& s) { return strtod(s.c_str(), nullptr); }

int g_fail = 0;
void check(bool ok, const char* what, int frame) {
    if (!ok) {
        printf("FAIL frame=%d: %s\n", frame, what);
        g_fail++;
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "用法: %s <test_data_dir>\n", argv[0]);
        return 1;
    }
    std::string dir = std::string(argv[1]) + "/case6_tracker";

    // dets.csv → frame → 检测列表
    std::map<int, std::vector<Detection>> det_map;
    {
        std::ifstream f(dir + "/dets.csv");
        std::string line;
        std::getline(f, line);  // 表头
        while (std::getline(f, line)) {
            auto v = split_csv(line);
            if (v.size() < 6) continue;
            Detection d;
            d.x1 = (float)fnum(v[1]);
            d.y1 = (float)fnum(v[2]);
            d.x2 = (float)fnum(v[3]);
            d.y2 = (float)fnum(v[4]);
            d.score = (float)fnum(v[5]);
            d.label = 0;
            det_map[atoi(v[0].c_str())].push_back(d);
        }
    }

    std::ifstream fg(dir + "/golden.csv");
    if (!fg) {
        fprintf(stderr, "找不到 %s/golden.csv，先跑 tools/gen_tracker_golden.py\n",
                dir.c_str());
        return 1;
    }

    Tracker tracker;  // 默认参数 = track3.py v2 默认
    std::vector<std::string> events;
    std::string line;
    std::getline(fg, line);  // 表头
    int frames = 0;
    while (std::getline(fg, line)) {
        auto v = split_csv(line);
        if (v.size() < 17) continue;
        int frame = atoi(v[0].c_str());
        double g_pred[4] = {fnum(v[1]), fnum(v[2]), fnum(v[3]), fnum(v[4])};
        double g_out[4] = {fnum(v[5]), fnum(v[6]), fnum(v[7]), fnum(v[8])};
        double g_state[8];
        for (int i = 0; i < 8; ++i) g_state[i] = fnum(v[9 + i]);

        auto it = det_map.find(frame);
        std::vector<Detection> dets = (it != det_map.end()) ? it->second
                                                            : std::vector<Detection>{};

        // predict：kal 活跃 ↔ golden pred 非 NaN
        const Track* t = tracker.predict();
        bool py_has_kal = !std::isnan(g_pred[0]);
        check((t != nullptr && tracker.kalman() != nullptr) == py_has_kal,
              "predict 存在性与 golden 不一致", frame);
        if (py_has_kal && t) {
            for (int i = 0; i < 4; ++i)
                check(fabs(t->box[i] - g_pred[i]) < TOL, "predict 框不符", frame);
        }

        t = tracker.update(dets, IMG_W, IMG_H, frame);
        if (t->event[0]) events.push_back(t->event);

        if (!std::isnan(g_out[0])) {
            for (int i = 0; i < 4; ++i)
                check(fabs(t->box[i] - g_out[i]) < TOL, "update 输出框不符", frame);
        }
        bool py_kal_after = !std::isnan(g_state[0]);
        check((tracker.kalman() != nullptr) == py_kal_after, "KF 存活状态不符", frame);
        if (py_kal_after && tracker.kalman()) {
            const double* x = tracker.kalman()->state();
            for (int i = 0; i < 8; ++i)
                check(fabs(x[i] - g_state[i]) < TOL, "KF 内部状态不符", frame);
        }
        frames++;
    }

    // 事件序列：初始锁定 → (段4 丢 8 帧超 coast) LOST → 重新锁定
    std::string ev;
    for (auto& e : events) ev += std::string(e) + ",";
    printf("事件序列: %s\n", ev.c_str());
    check(ev == "INITIAL_LOCK,LOST,REACQUIRED,", "事件序列不符", -1);

    printf("%s: %d 帧回放, %d 处不符\n", g_fail ? "FAIL" : "PASS", frames, g_fail);
    return g_fail ? 1 : 0;
}
