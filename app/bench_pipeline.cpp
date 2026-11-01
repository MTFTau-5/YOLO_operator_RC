// bench_pipeline —— 流水线实测基准（不进 yolo_rt，不影响其相机接口设计）。
//
// 用 OpenCV 从视频文件循环取帧，跑完整链路的两种模式：
//   --serial   单线程串行：逐阶段计时（letterbox / infer / decode+nms / tracker / 端到端）
//   默认       两线程流水线（PIPELINE.md §4）：吞吐 FPS、端到端延迟 P50/P99、丢帧数
//
// 用法: bench_pipeline --param P --model M --model-type v8-pose --heads out0 \
//          --num-classes 1 --num-kpts 4 --video xxx.mp4 [--frames 500] [--serial]

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <pthread.h>
#include <sched.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include <opencv2/opencv.hpp>

#include "decode_af_raw.h"
#include "decode_avx2.h"
#include "decode_e2e.h"
#include "decode_v8.h"
#include "ncnn_infer.h"
#include "nms_simd.h"
#include "pipeline.h"
#include "preprocess_avx2.h"
#include "tracker.h"

using namespace yoloop;

namespace {

struct BArgs {
    std::string param, model, video;
    std::string in0 = "in0";
    std::vector<std::string> heads;
    int in_w = 640, in_h = 640;
    HeadType head = HeadType::PoseFused;
    int reg_max = 16;
    int num_classes = 1;
    int num_kpts = 4;
    int gpu = 0;
    int threads = 4;
    float conf = 0.25f, nms = 0.45f;
    int topk = 100;
    int frames = 500;
    bool serial = false;
    std::string cores_a = "0-3", cores_b = "4-7";
};

BArgs parse(int argc, char** argv) {
    BArgs a;
    auto val = [&](int& i) { return std::string(argv[++i]); };
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if (k == "--param") a.param = val(i);
        else if (k == "--model") a.model = val(i);
        else if (k == "--video") a.video = val(i);
        else if (k == "--in0") a.in0 = val(i);
        else if (k == "--heads") {
            std::string v = val(i);
            size_t p = 0;
            while (true) {
                size_t q = v.find(',', p);
                a.heads.push_back(v.substr(p, q == std::string::npos ? q : q - p));
                if (q == std::string::npos) break;
                p = q + 1;
            }
        }
        else if (k == "--in-size") sscanf(val(i).c_str(), "%dx%d", &a.in_w, &a.in_h);
        else if (k == "--model-type") {
            std::string v = val(i);
            if (!parse_head_type(v, a.head)) {
                fprintf(stderr, "未知 model-type: %s\n", v.c_str());
                exit(1);
            }
        }
        else if (k == "--reg-max") a.reg_max = atoi(val(i).c_str());
        else if (k == "--num-classes") a.num_classes = atoi(val(i).c_str());
        else if (k == "--num-kpts") a.num_kpts = atoi(val(i).c_str());
        else if (k == "--gpu") a.gpu = atoi(val(i).c_str());
        else if (k == "--threads") a.threads = atoi(val(i).c_str());
        else if (k == "--conf") a.conf = atof(val(i).c_str());
        else if (k == "--nms") a.nms = atof(val(i).c_str());
        else if (k == "--topk") a.topk = atoi(val(i).c_str());
        else if (k == "--frames") a.frames = atoi(val(i).c_str());
        else if (k == "--serial") a.serial = true;
        else if (k == "--cores-a") a.cores_a = val(i);
        else if (k == "--cores-b") a.cores_b = val(i);
    }
    return a;
}

void pin(const std::string& spec) {
    cpu_set_t set;
    CPU_ZERO(&set);
    int nproc = (int)sysconf(_SC_NPROCESSORS_ONLN);
    int a, b;
    if (sscanf(spec.c_str(), "%d-%d", &a, &b) == 2)
        for (int c = a; c <= b && c < nproc; ++c) CPU_SET(c, &set);
    else if (sscanf(spec.c_str(), "%d", &a) == 1 && a < nproc)
        CPU_SET(a, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

class VideoSource {  // bench 专用帧源：视频循环 + 尺寸适配
public:
    bool open(const std::string& path, int max_w, int max_h) {
        cap_.open(path);
        if (!cap_.isOpened()) return false;
        w_ = (int)cap_.get(cv::CAP_PROP_FRAME_WIDTH);
        h_ = (int)cap_.get(cv::CAP_PROP_FRAME_HEIGHT);
        double s = std::min((double)max_w / w_, (double)max_h / h_);
        if (s < 1.0) {
            w_ = (int)(w_ * s);
            h_ = (int)(h_ * s);
            resize_ = true;
        }
        return true;
    }
    bool grab(uint8_t* dst, int& w, int& h) {
        cv::Mat f;
        if (!cap_.read(f)) {  // 循环
            cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
            if (!cap_.read(f)) return false;
        }
        if (resize_) cv::resize(f, f, {w_, h_});
        cv::cvtColor(f, f, cv::COLOR_BGR2RGB);
        memcpy(dst, f.data, (size_t)w_ * h_ * 3);
        w = w_;
        h = h_;
        return true;
    }
    int w() const { return w_; }
    int h() const { return h_; }

private:
    cv::VideoCapture cap_;
    int w_ = 0, h_ = 0;
    bool resize_ = false;
};

struct Bench {
    std::vector<double> letterbox, infer, decode_nms, track, e2e;  // us
    static void report(const char* name, std::vector<double> v) {
        if (v.empty()) return;
        std::sort(v.begin(), v.end());
        auto pct = [&](double q) { return v[(size_t)(q * (v.size() - 1))] / 1000.0; };
        double sum = 0;
        for (double x : v) sum += x;
        printf("  %-12s mean=%6.3fms  P50=%6.3fms  P99=%6.3fms\n", name,
               sum / v.size() / 1000.0, pct(0.5), pct(0.99));
    }
};

}  // namespace

int main(int argc, char** argv) {
    BArgs a = parse(argc, argv);
    VideoSource cam;
    if (!cam.open(a.video, 1920, 1080)) {
        fprintf(stderr, "打不开视频: %s\n", a.video.c_str());
        return 1;
    }
    NcnnInfer infer;
    InferConfig ic;
    ic.param_path = a.param;
    ic.model_path = a.model;
    ic.input_name = a.in0;
    ic.head_names = a.heads;
    ic.in_w = a.in_w;
    ic.in_h = a.in_h;
    ic.head = a.head;
    ic.num_classes = a.num_classes;
    ic.num_kpts = a.num_kpts;
    ic.reg_max = a.reg_max;
    ic.gpu_index = a.gpu;
    ic.cpu_threads = a.threads;
    if (!infer.load(ic, 0)) return 1;
    fprintf(stderr, "[bench] 视频 %dx%d, 帧数 %d, %s 模式\n", cam.w(), cam.h(), a.frames,
            a.serial ? "串行" : "两线程");

    Pipeline pipe;
    pipe.init(a.in_w, a.in_h, infer.c_total(), infer.pred_n(), 1920, 1080);
    Tracker tracker;
    Bench bm;

    // AnchorRaw 需要 anchor 表（无配置入口，用 YOLOv5s P5 默认）；AfRaw 用探测推导的网格表
    AnchorTable tab;
    if (a.head == HeadType::AnchorRaw)
        tab = build_anchor_table(default_levels(a.in_w, a.in_h));
    else if (a.head == HeadType::AfRaw)
        tab = infer.grid();

    auto decode_track = [&](Slot* s, Bench& b) {
        auto t0 = std::chrono::steady_clock::now();
        std::vector<Detection> kept;
        std::vector<float> kpts_raw;
        switch (a.head) {
            case HeadType::PoseFused: {
                x86::decode_v8_pose(s->pred.data(), a.num_classes, a.num_kpts,
                                    infer.pred_n(), a.conf, s->dets, kpts_raw);
                auto idx = x86::nms_indices(s->dets, a.nms, a.topk);
                kept.reserve(idx.size());
                for (int32_t i : idx) kept.push_back(s->dets[(size_t)i]);
                break;
            }
            case HeadType::AnchorRaw:
                kept = x86::nms(x86::decode(s->pred.data(), tab, a.num_classes, a.conf),
                                a.nms, a.topk);
                break;
            case HeadType::AfRaw:
                kept = x86::nms(x86::decode_af_raw(s->pred.data(), tab, a.num_classes,
                                                   a.reg_max, a.conf),
                                a.nms, a.topk);
                break;
            case HeadType::E2E:
                kept = x86::decode_e2e(s->pred.data(), infer.pred_n(), a.conf);
                break;
            case HeadType::AfFused:
                kept = x86::nms(x86::decode_v8(s->pred.data(), a.num_classes,
                                               infer.pred_n(), a.conf),
                                a.nms, a.topk);
                break;
        }
        auto t1 = std::chrono::steady_clock::now();
        tracker.predict();
        tracker.update(kept, s->img_w, s->img_h, 0);
        auto t2 = std::chrono::steady_clock::now();
        b.decode_nms.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        b.track.push_back(std::chrono::duration<double, std::micro>(t2 - t1).count());
    };

    if (a.serial) {
        // ---- 串行：逐阶段计时 ----
        for (int f = 0; f < a.frames; ++f) {
            Slot* s = pipe.acquire_slot();
            auto t0 = std::chrono::steady_clock::now();
            cam.grab(s->img.data(), s->img_w, s->img_h);
            x86::letterbox_normalize(s->img.data(), s->img_w, s->img_h, s->blob.data(),
                                     a.in_w, a.in_h);
            auto t1 = std::chrono::steady_clock::now();
            infer.infer(s->blob.data(), s->pred.data());
            auto t2 = std::chrono::steady_clock::now();
            decode_track(s, bm);
            auto t3 = std::chrono::steady_clock::now();
            bm.letterbox.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
            bm.infer.push_back(std::chrono::duration<double, std::micro>(t2 - t1).count());
            bm.e2e.push_back(std::chrono::duration<double, std::micro>(t3 - t0).count());
            pipe.q_free.push(s);
        }
        printf("== 串行模式逐阶段（%d 帧）==\n", a.frames);
        Bench::report("letterbox", bm.letterbox);
        Bench::report("infer(GPU)", bm.infer);
        Bench::report("decode+nms", bm.decode_nms);
        Bench::report("tracker", bm.track);
        Bench::report("端到端", bm.e2e);
        double sum = 0;
        for (double x : bm.e2e) sum += x;
        printf("  理论吞吐上限 ≈ %.1f FPS\n", a.frames / (sum * 1e-6));
    } else {
        // ---- 两线程流水线（PIPELINE.md §4） ----
        std::vector<int64_t> lat;
        std::atomic<uint64_t> done_cnt{0};
        std::thread tb([&] {
            pin(a.cores_b);
            Slot* s;
            while (pipe.q_gpu.pop(s)) {
                auto t0 = std::chrono::steady_clock::now();
                s->error = !infer.infer(s->blob.data(), s->pred.data());
                auto t1 = std::chrono::steady_clock::now();
                bm.infer.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
                pipe.q_back.push(s);
            }
        });
        pin(a.cores_a);
        int produced = 0;
        auto wall0 = std::chrono::steady_clock::now();
        // 终止条件：已产出帧全部有着落（处理完 done_cnt + 被丢 dropped）。
        // 注意 acquire_slot 会从 q_gpu 偷最旧帧（dropped++），只数 done_cnt 会死循环。
        while (done_cnt + pipe.dropped.load() < (uint64_t)a.frames) {
            Slot* s = nullptr;
            if (produced < a.frames) {
                s = pipe.acquire_slot();
                if (s) {
                    if (cam.grab(s->img.data(), s->img_w, s->img_h)) {
                        s->t_capture = now_us();
                        x86::letterbox_normalize(s->img.data(), s->img_w, s->img_h,
                                                 s->blob.data(), a.in_w, a.in_h);
                        pipe.q_gpu.push(s);
                        produced++;
                    } else {
                        pipe.q_free.push(s);
                        s = nullptr;
                    }
                }
            }
            Slot* done = nullptr;
            // 无槽可生产（或已产完）时阻塞等结果，避免空转
            bool got = (produced >= a.frames || !s) ? pipe.q_back.pop(done)
                                                    : pipe.q_back.try_pop(done);
            if (got && done) {
                decode_track(done, bm);
                lat.push_back(now_us() - done->t_capture);
                done_cnt++;
                pipe.q_free.push(done);
            }
        }
        pipe.shutdown();
        tb.join();
        double wall = std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - wall0)
                          .count();
        std::sort(lat.begin(), lat.end());
        auto pct = [&](double q) { return lat[(size_t)(q * (lat.size() - 1))] / 1000.0; };
        printf("== 两线程流水线（%d 帧）==\n", a.frames);
        Bench::report("infer(GPU)", bm.infer);
        Bench::report("decode+nms", bm.decode_nms);
        Bench::report("tracker", bm.track);
        if (!lat.empty())
            printf("  端到端延迟  P50=%.3fms  P99=%.3fms\n", pct(0.5), pct(0.99));
        printf("  吞吐 = %.1f FPS（处理 %llu 帧, wall %.2fs）\n",
               done_cnt.load() / wall, (unsigned long long)done_cnt.load(), wall);
        printf("  丢帧 dropped = %llu\n", (unsigned long long)pipe.dropped.load());
    }
    return 0;
}
