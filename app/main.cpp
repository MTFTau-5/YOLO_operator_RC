// yolo_rt —— 实时检测跟踪流水线（PIPELINE.md：两线程 + 三槽位 + 丢旧帧）
//
//   线程A（CPU，默认绑核 0-3）：采集 → letterbox → 推 q_gpu → 取回上帧 →
//                                decode → NMS → Tracker(Kalman) → UDP 控制
//   线程B（GPU 提交，默认绑核 4-7）：q_gpu.pop → ncnn extract() → q_back.push
//
// --serial 退化为 M1/M3 串行模式（单线程跑通全链路，用于数值对齐与验收）。

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <string>
#include <thread>
#include <vector>

#include "camera_source.h"
#include "config.h"
#include "control_tx.h"
#include "decode_avx2.h"
#include "decode_v8.h"
#include "ncnn_infer.h"
#include "nms_simd.h"
#include "pipeline.h"
#include "preprocess_avx2.h"
#include "stats.h"
#include "tracker.h"

namespace {

using namespace yoloop;

struct Args {
    // 模型
    std::string param, model;
    std::string in0 = "in0";
    std::vector<std::string> heads;
    int in_w = 640, in_h = 640;
    int model_type = 0;  // 0=v5 契约 [4+1+nc,N]；1=v8/v11 检测 [4+nc,N]；2=v8 pose
    int num_classes = 2;
    int num_kpts = 0;    // model_type=2 时有效
    int gpu = 0;
    int cpu_threads = 1;
    bool no_model = false;   // 冒烟：跳过 ncnn，线程B 假装推理
    int fake_gpu_ms = 10;    // --no-model 时线程B 每帧睡眠毫秒数
    // 检测阈值
    float conf = 0.3f;
    float nms = 0.6f;
    int topk = 300;
    // 运行模式
    bool serial = false;
    std::string cores_a = "0-3";
    std::string cores_b = "4-7";
    int max_w = 1920, max_h = 1080;  // 采集缓冲上限
    int max_frames = 0;              // 0 = 不限
    // anchor 表（空 = YOLOv5s P5 默认）；换模型时在配置文件里给 anchors-s8/s16/s32
    std::vector<float> anchors_s8, anchors_s16, anchors_s32;
    // 控制输出
    std::string udp;  // host:port，空 = 不发
    // 跟踪参数（track3.py v2 默认）
    TrackerParams tk;
    // 配置文件
    std::string config;
    bool dump_config = false;
};

volatile sig_atomic_t g_sigint = 0;
void on_sigint(int) { g_sigint = 1; }

std::vector<std::string> split(const std::string& s, char c) {
    std::vector<std::string> out;
    size_t p = 0;
    while (true) {
        size_t q = s.find(c, p);
        out.push_back(s.substr(p, q == std::string::npos ? q : q - p));
        if (q == std::string::npos) break;
        p = q + 1;
    }
    return out;
}

void parse_size(const std::string& s, int& w, int& h) {
    if (sscanf(s.c_str(), "%dx%d", &w, &h) != 2) {
        fprintf(stderr, "尺寸格式应为 WxH: %s\n", s.c_str());
        exit(1);
    }
}

// 绑核（"0-3" 或 "0,2"），核数不足自动收缩并告警
void pin_thread(const std::string& spec, const char* name) {
    int nproc = (int)sysconf(_SC_NPROCESSORS_ONLN);
    cpu_set_t set;
    CPU_ZERO(&set);
    bool any = false;
    for (const auto& tok : split(spec, ',')) {
        int a, b;
        if (sscanf(tok.c_str(), "%d-%d", &a, &b) == 2) {
            for (int c = a; c <= b; ++c)
                if (c < nproc) {
                    CPU_SET(c, &set);
                    any = true;
                }
        } else if (sscanf(tok.c_str(), "%d", &a) == 1 && a < nproc) {
            CPU_SET(a, &set);
            any = true;
        }
    }
    if (!any) return;
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0)
        fprintf(stderr, "[warn] %s 绑核 %s 失败\n", name, spec.c_str());
}

void usage(const char* argv0) {
    fprintf(stderr,
            "用法: %s [--config app.yaml] [覆盖选项...]\n"
            "  配置: --config 路径（扁平 YAML，key 与下列长选项同名）；CLI 覆盖配置文件\n"
            "        --dump-config 打印最终生效配置后退出\n"
            "  模型: --param P --model M --heads b0,b1,b2 [--in0 NAME] [--in-size WxH]\n"
            "        [--num-classes N] [--gpu N] [--cpu-threads N]\n"
            "        [--anchors-s8 w,h,w,h,... --anchors-s16 ... --anchors-s32 ...]\n"
            "        [--no-model --fake-gpu-ms N]   (无模型冒烟)\n"
            "  检测: [--conf 0.3] [--nms 0.6] [--topk 300]\n"
            "  运行: [--serial] [--cores-a 0-3] [--cores-b 4-7] [--max-w W --max-h H]\n"
            "        [--max-frames N]\n"
            "  控制: [--udp host:port]\n"
            "  跟踪: [--gate-px 120] [--gate-scale 2] [--size-ratio 3] [--coast 5]\n",
            argv0);
}

std::vector<float> parse_floats(std::string v) {
    if (!v.empty() && v.front() == '[' && v.back() == ']') v = v.substr(1, v.size() - 2);
    std::vector<float> out;
    for (auto& s : split(v, ',')) out.push_back((float)atof(s.c_str()));
    return out;
}

// 统一的 key=value 应用入口：CLI 与配置文件共用（key 不带前导 --）
void apply_kv(Args& a, const std::string& k, const std::string& v) {
    if (k == "param") a.param = v;
    else if (k == "model") a.model = v;
    else if (k == "in0") a.in0 = v;
    else if (k == "heads") {
        std::string t = v;
        if (!t.empty() && t.front() == '[' && t.back() == ']') t = t.substr(1, t.size() - 2);
        a.heads = split(t, ',');
        for (auto& h : a.heads) {
            size_t b = h.find_first_not_of(" \t"), e = h.find_last_not_of(" \t");
            h = h.substr(b, e - b + 1);
        }
    }
    else if (k == "in-size") parse_size(v, a.in_w, a.in_h);
    else if (k == "model-type") {
        if (v == "v5") a.model_type = 0;
        else if (v == "v8" || v == "v11") a.model_type = 1;
        else if (v == "v8-pose" || v == "pose" || v == "v11-pose") a.model_type = 2;
        else {
            fprintf(stderr, "model-type 仅支持 v5 / v8 / v8-pose: %s\n", v.c_str());
            exit(1);
        }
    }
    else if (k == "num-kpts") a.num_kpts = atoi(v.c_str());
    else if (k == "num-classes") a.num_classes = atoi(v.c_str());
    else if (k == "gpu") a.gpu = atoi(v.c_str());
    else if (k == "cpu-threads") a.cpu_threads = atoi(v.c_str());
    else if (k == "no-model") a.no_model = (v == "true" || v == "1" || v == "yes");
    else if (k == "fake-gpu-ms") a.fake_gpu_ms = atoi(v.c_str());
    else if (k == "conf") a.conf = atof(v.c_str());
    else if (k == "nms") a.nms = atof(v.c_str());
    else if (k == "topk") a.topk = atoi(v.c_str());
    else if (k == "serial") a.serial = (v == "true" || v == "1" || v == "yes");
    else if (k == "cores-a") a.cores_a = v;
    else if (k == "cores-b") a.cores_b = v;
    else if (k == "max-w") a.max_w = atoi(v.c_str());
    else if (k == "max-h") a.max_h = atoi(v.c_str());
    else if (k == "max-frames") a.max_frames = atoi(v.c_str());
    else if (k == "udp") a.udp = v;
    else if (k == "gate-px") a.tk.association_gate_px = atof(v.c_str());
    else if (k == "gate-scale") a.tk.association_gate_scale = atof(v.c_str());
    else if (k == "size-ratio") a.tk.size_ratio_limit = atof(v.c_str());
    else if (k == "size-memory") a.tk.size_memory_frames = atoi(v.c_str());
    else if (k == "coast") a.tk.coast_frames = atoi(v.c_str());
    else if (k == "max-misses") a.tk.max_misses = atoi(v.c_str());
    else if (k == "anchors-s8") a.anchors_s8 = parse_floats(v);
    else if (k == "anchors-s16") a.anchors_s16 = parse_floats(v);
    else if (k == "anchors-s32") a.anchors_s32 = parse_floats(v);
    else if (k == "config" || k == "dump-config") {}  // 已处理
    else fprintf(stderr, "[warn] 未知配置项: %s\n", k.c_str());
}

// 合并顺序：默认值 < --config 文件 < CLI
Args parse_args(int argc, char** argv) {
    Args a;
    // 第一遍：找 --config
    for (int i = 1; i + 1 < argc; ++i)
        if (std::string(argv[i]) == "--config") a.config = argv[i + 1];
    if (!a.config.empty()) {
        Config cfg;
        std::string err;
        if (!cfg.load(a.config, err)) {
            fprintf(stderr, "[config] %s\n", err.c_str());
            exit(1);
        }
        for (const auto& kv : cfg.all()) apply_kv(a, kv.first, kv.second);
    }
    // 第二遍：CLI 覆盖
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if (k.rfind("--", 0) != 0) {
            usage(argv[0]);
            exit(1);
        }
        k = k.substr(2);
        if (k == "serial" || k == "no-model" || k == "dump-config") {
            if (k == "dump-config") a.dump_config = true;
            else apply_kv(a, k, "true");
            continue;
        }
        if (++i >= argc) {
            fprintf(stderr, "--%s 缺参数\n", k.c_str());
            exit(1);
        }
        apply_kv(a, k, argv[i]);
    }
    if (!a.no_model && (a.param.empty() || a.model.empty() || a.heads.empty())) {
        usage(argv[0]);
        exit(1);
    }
    return a;
}

void dump_config(const Args& a) {
    fprintf(stderr,
            "# 最终生效配置\n"
            "param: %s\nmodel: %s\nin0: %s\nheads: [%s]\nin-size: %dx%d\n"
            "model-type: %s\nnum-classes: %d\nnum-kpts: %d\ngpu: %d\ncpu-threads: %d\n"
            "no-model: %s\nfake-gpu-ms: %d\n"
            "conf: %.3f\nnms: %.3f\ntopk: %d\nserial: %s\ncores-a: %s\ncores-b: %s\n"
            "max-w: %d\nmax-h: %d\nmax-frames: %d\nudp: %s\n"
            "gate-px: %.1f\ngate-scale: %.2f\nsize-ratio: %.2f\nsize-memory: %d\n"
            "coast: %d\nmax-misses: %d\nanchors-s8: %zu 个\nanchors-s16: %zu 个\n"
            "anchors-s32: %zu 个\n",
            a.param.c_str(), a.model.c_str(), a.in0.c_str(),
            [&] {
                std::string s;
                for (size_t i = 0; i < a.heads.size(); ++i)
                    s += (i ? ", " : "") + a.heads[i];
                return s;
            }()
                .c_str(),
            a.in_w, a.in_h,
            a.model_type == 0 ? "v5" : (a.model_type == 1 ? "v8" : "v8-pose"),
            a.num_classes, a.num_kpts, a.gpu, a.cpu_threads,
            a.no_model ? "true" : "false", a.fake_gpu_ms, a.conf, a.nms, a.topk,
            a.serial ? "true" : "false", a.cores_a.c_str(), a.cores_b.c_str(), a.max_w,
            a.max_h, a.max_frames, a.udp.c_str(), a.tk.association_gate_px,
            a.tk.association_gate_scale, a.tk.size_ratio_limit, a.tk.size_memory_frames,
            a.tk.coast_frames, a.tk.max_misses, a.anchors_s8.size() / 2,
            a.anchors_s16.size() / 2, a.anchors_s32.size() / 2);
}

// 把 decode 出的网络输入尺度框映射回相机原图坐标（letterbox 逆变换）
void unletterbox(std::vector<Detection>& dets, int img_w, int img_h, int in_w, int in_h) {
    float scale = std::min((float)in_w / img_w, (float)in_h / img_h);
    float pad_x = (in_w - roundf(img_w * scale)) * 0.5f;
    float pad_y = (in_h - roundf(img_h * scale)) * 0.5f;
    for (auto& d : dets) {
        d.x1 = (d.x1 - pad_x) / scale;
        d.y1 = (d.y1 - pad_y) / scale;
        d.x2 = (d.x2 - pad_x) / scale;
        d.y2 = (d.y2 - pad_y) / scale;
    }
}

// pose 关键点的 letterbox 逆变换（每点 x,y,conf；conf 不变换）
void unletterbox_kpts(std::vector<float>& kpts, int num_kpts, int img_w, int img_h,
                      int in_w, int in_h) {
    float scale = std::min((float)in_w / img_w, (float)in_h / img_h);
    float pad_x = (in_w - roundf(img_w * scale)) * 0.5f;
    float pad_y = (in_h - roundf(img_h * scale)) * 0.5f;
    for (size_t i = 0; i + 2 < kpts.size(); i += 3) {
        kpts[i] = (kpts[i] - pad_x) / scale;
        kpts[i + 1] = (kpts[i + 1] - pad_y) / scale;
    }
    (void)num_kpts;
}

// 流水线上下文（线程A独占 tracker/stats，天然无锁 —— PIPELINE.md §1）
struct Ctx {
    Args args;
    Pipeline pipe;
    AnchorTable tab;      // 仅 model_type=0（v5 契约）使用
    int c_total = 0;      // pred 通道数（由 model_type 推导）
    int pred_n = 0;       // 输出总列数（v5=anchor 表 N；v8=模型发现）
    NcnnInfer infer;
    NullSource cam;  // 采集接口占位：接真实相机时替换为其实现
    Tracker tracker;
    ControlTx tx;
    Stats stats;
    uint32_t frame_id = 0;
};

// §6 GPU 出错：丢该帧，KF 纯 predict 外推一帧
void consume(Ctx& c, Slot* done) {
    const Track* t = c.tracker.predict();
    if (!done->error) {
        if (c.args.model_type == 0) {
            done->dets = x86::decode(done->pred.data(), c.tab, c.args.num_classes,
                                     c.args.conf);
            done->kept = x86::nms(std::move(done->dets), c.args.nms, c.args.topk);
        } else if (c.args.model_type == 1) {
            done->dets = x86::decode_v8(done->pred.data(), c.args.num_classes, c.pred_n,
                                        c.args.conf);
            done->kept = x86::nms(std::move(done->dets), c.args.nms, c.args.topk);
        } else {  // v8 pose：关键点随框过 NMS
            const int kstride = 3 * c.args.num_kpts;
            std::vector<float> kpts_raw;
            x86::decode_v8_pose(done->pred.data(), c.args.num_classes,
                                c.args.num_kpts, c.pred_n, c.args.conf, done->dets,
                                kpts_raw);
            auto idx = x86::nms_indices(done->dets, c.args.nms, c.args.topk);
            done->kept.clear();
            done->kpts.clear();
            for (int32_t i : idx) {
                done->kept.push_back(done->dets[(size_t)i]);
                done->kpts.insert(done->kpts.end(), kpts_raw.begin() + (size_t)i * kstride,
                                  kpts_raw.begin() + (size_t)(i + 1) * kstride);
            }
        }
        unletterbox(done->kept, done->img_w, done->img_h, c.args.in_w, c.args.in_h);
        if (c.args.model_type == 2)
            unletterbox_kpts(done->kpts, c.args.num_kpts, done->img_w, done->img_h,
                             c.args.in_w, c.args.in_h);
        t = c.tracker.update(done->kept, done->img_w, done->img_h, c.frame_id);
    } else {
        t = c.tracker.update({}, done->img_w, done->img_h, c.frame_id);
    }
    // §4：只发确认的轨迹
    if (t && t->hits >= 2 && t->misses == 0 && c.tx.valid()) {
        ControlPacket pkt;
        pkt.frame_id = c.frame_id;
        pkt.state = (uint32_t)t->state;
        pkt.cx = t->box[0] + t->box[2] * 0.5f;
        pkt.cy = t->box[1] + t->box[3] * 0.5f;
        pkt.w = t->box[2];
        pkt.h = t->box[3];
        pkt.vx = t->vel[0];
        pkt.vy = t->vel[1];
        pkt.vw = t->vel[2];
        pkt.vh = t->vel[3];
        pkt.score = t->score;
        // pose：附本帧命中框的关键点（coast/丢失帧不带）
        int lm = c.tracker.last_match();
        if (c.args.model_type == 2 && lm >= 0 && !done->error) {
            int nk = c.args.num_kpts;
            pkt.kpt_count = (uint32_t)std::min(nk, (int)ControlPacket::MAX_KPTS);
            const float* src = done->kpts.data() + (size_t)lm * 3 * nk;
            std::memcpy(pkt.kpts, src, (size_t)pkt.kpt_count * 3 * sizeof(float));
        }
        c.tx.send(pkt);
    }
    c.frame_id++;
    int64_t lat = now_us() - done->t_capture;
    c.pipe.latency_us = lat;
    c.stats.add_latency(lat);
    c.stats.set_dropped(c.pipe.dropped);
    c.pipe.q_free.push(done);
}

// 线程B：GPU 提交（§1 拓扑）
void thread_gpu(Ctx& c) {
    pin_thread(c.args.cores_b, "thread_gpu");
    Slot* s = nullptr;
    while (c.pipe.q_gpu.pop(s)) {
        if (c.args.no_model) {
            std::this_thread::sleep_for(std::chrono::milliseconds(c.args.fake_gpu_ms));
            std::memset(s->pred.data(), 0, s->pred.size() * sizeof(float));
            s->error = false;
        } else {
            s->error = !c.infer.infer(s->blob.data(), s->pred.data());
        }
        c.pipe.q_back.push(s);
    }
}

// 采集失败公共路径：归还槽位 + §6 相机掉线 1s 重连循环（不空转）
void camera_offline(Ctx& c, Slot* s) {
    c.pipe.q_free.push(s);
    if (!c.cam.reopen()) {
        static int64_t last_log = 0;
        int64_t t = now_us();
        if (t - last_log > 1000000) {
            fprintf(stderr, "[camera] 无新帧/掉线，1s 重连中...\n");
            last_log = t;
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

int run_serial(Ctx& c) {
    pin_thread(c.args.cores_a, "thread_cpu");
    int64_t last_stat = now_us();
    while (c.pipe.run && !g_sigint) {
        if (c.args.max_frames && c.frame_id >= (uint32_t)c.args.max_frames) break;
        Slot* s = c.pipe.acquire_slot();
        if (!s) continue;
        if (!c.cam.grab(s->img.data(), s->img_w, s->img_h)) {
            camera_offline(c, s);
            continue;
        }
        s->t_capture = now_us();
        x86::letterbox_normalize(s->img.data(), s->img_w, s->img_h, s->blob.data(),
                                 c.args.in_w, c.args.in_h);
        if (c.args.no_model) {
            std::this_thread::sleep_for(std::chrono::milliseconds(c.args.fake_gpu_ms));
            std::memset(s->pred.data(), 0, s->pred.size() * sizeof(float));
            s->error = false;
        } else {
            s->error = !c.infer.infer(s->blob.data(), s->pred.data());
        }
        consume(c, s);
        if (now_us() - last_stat > 2000000) {
            c.stats.tick(now_us());
            last_stat = now_us();
        }
    }
    return 0;
}

int run_pipeline(Ctx& c) {
    std::thread b(thread_gpu, std::ref(c));
    pin_thread(c.args.cores_a, "thread_cpu");
    int64_t last_stat = now_us();
    // §4 线程A完整循环（修正：decode 作用于取回的 done 槽位，不是新推的 s）
    while (c.pipe.run && !g_sigint) {
        if (c.args.max_frames && c.frame_id >= (uint32_t)c.args.max_frames) break;
        Slot* s = c.pipe.acquire_slot();
        if (!s) {  // free 与 q_gpu 都空：B 正在推理，阻塞等结果
            Slot* done = nullptr;
            if (c.pipe.q_back.pop(done)) consume(c, done);
            continue;
        }
        if (!c.cam.grab(s->img.data(), s->img_w, s->img_h)) {
            // 已推给 GPU 的在途帧仍要取回
            Slot* done = nullptr;
            if (c.pipe.q_back.try_pop(done)) consume(c, done);
            camera_offline(c, s);
            continue;
        }
        s->t_capture = now_us();
        x86::letterbox_normalize(s->img.data(), s->img_w, s->img_h, s->blob.data(),
                                 c.args.in_w, c.args.in_h);
        c.pipe.q_gpu.push(s);  // 交给 GPU，不等待

        Slot* done = nullptr;  // 取上一帧结果（通常已完成）
        if (c.pipe.q_back.try_pop(done)) consume(c, done);
        if (now_us() - last_stat > 2000000) {
            c.stats.tick(now_us());
            last_stat = now_us();
        }
    }
    // §6 关机：关队列 → B 退出 → A 排空 q_back → join，不杀线程
    c.pipe.shutdown();
    b.join();
    c.pipe.drain_back();
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    Ctx c;
    c.args = parse_args(argc, argv);
    if (c.args.dump_config) {
        dump_config(c.args);
        return 0;
    }
    signal(SIGINT, on_sigint);
    signal(SIGTERM, on_sigint);

    // v8 pose 必须给关键点数
    if (c.args.model_type == 2 && c.args.num_kpts <= 0) {
        fprintf(stderr, "[yolo_rt] model-type=v8-pose 需要 num-kpts > 0\n");
        return 1;
    }

    // v5 契约才需要 anchor 表：配置给了三级 anchor 就用配置的，否则 YOLOv5s P5 默认。
    // v8/v11 anchor-free，N 由模型输出自动发现。
    int expected_n = 0;
    if (c.args.model_type == 0) {
        if (!c.args.anchors_s8.empty() && !c.args.anchors_s16.empty() &&
            !c.args.anchors_s32.empty()) {
            auto mk = [](int stride, int gw, int gh, const std::vector<float>& fl) {
                LevelSpec lv;
                lv.stride = stride;
                lv.grid_w = gw;
                lv.grid_h = gh;
                for (size_t i = 0; i + 1 < fl.size(); i += 2)
                    lv.anchors.push_back({fl[i], fl[i + 1]});
                return lv;
            };
            c.tab = build_anchor_table(
                {mk(8, c.args.in_w / 8, c.args.in_h / 8, c.args.anchors_s8),
                 mk(16, c.args.in_w / 16, c.args.in_h / 16, c.args.anchors_s16),
                 mk(32, c.args.in_w / 32, c.args.in_h / 32, c.args.anchors_s32)});
        } else {
            c.tab = build_anchor_table(default_levels(c.args.in_w, c.args.in_h));
        }
        expected_n = c.tab.n;
    }

    if (!c.args.no_model) {
        InferConfig icfg;
        icfg.param_path = c.args.param;
        icfg.model_path = c.args.model;
        icfg.input_name = c.args.in0;
        icfg.head_names = c.args.heads;
        icfg.in_w = c.args.in_w;
        icfg.in_h = c.args.in_h;
        icfg.model_type = c.args.model_type;
        icfg.num_classes = c.args.num_classes;
        icfg.num_kpts = c.args.num_kpts;
        icfg.gpu_index = c.args.gpu;
        icfg.cpu_threads = c.args.cpu_threads;
        // 加载时试跑校验：v5 校验通道数与 anchor 总数；v8 校验通道数并发现 N。
        // 尺寸不匹配启动即报错，不会带病运行。
        if (!c.infer.load(icfg, expected_n)) return 1;
        c.c_total = c.infer.c_total();
        c.pred_n = c.infer.pred_n();
    } else {
        fprintf(stderr, "[yolo_rt] --no-model 冒烟模式（线程B 假推理 %dms/帧）\n",
                c.args.fake_gpu_ms);
        c.c_total = c.args.model_type == 2
                        ? 4 + c.args.num_classes + 3 * c.args.num_kpts
                        : (c.args.model_type == 1 ? 4 + c.args.num_classes
                                                  : 5 + c.args.num_classes);
        c.pred_n = expected_n > 0 ? expected_n : 8400;
    }
    c.pipe.init(c.args.in_w, c.args.in_h, c.c_total, c.pred_n, c.args.max_w,
                c.args.max_h);

    if (!c.args.udp.empty()) {
        auto hp = split(c.args.udp, ':');
        if (hp.size() != 2 || !c.tx.open(hp[0], (uint16_t)atoi(hp[1].c_str()))) {
            fprintf(stderr, "[control] --udp 格式 host:port\n");
            return 1;
        }
    }

    c.tracker = Tracker(c.args.tk);
    fprintf(stderr, "[yolo_rt] %s 模式, in=%dx%d, N=%d, 槽位=%d\n",
            c.args.serial ? "串行" : "两线程", c.args.in_w, c.args.in_h, c.pred_n,
            NUM_SLOTS);

    int rc = c.args.serial ? run_serial(c) : run_pipeline(c);
    fprintf(stderr, "[yolo_rt] 退出, 处理 %u 帧, 丢帧 %llu\n", c.frame_id,
            (unsigned long long)c.pipe.dropped.load());
    return rc;
}
