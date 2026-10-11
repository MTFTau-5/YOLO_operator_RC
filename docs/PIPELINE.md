# 流水线设计规格 — 两线程 + 三槽位 + 丢旧帧

本文档是视觉系统流水线的实现规范。前置阅读：`README.md`（项目总览）。

## 1. 拓扑

```
线程A（CPU, 绑核 0-3）:  采集 → letterbox → [推给GPU] → 取回结果 → decode → NMS → 卡尔曼 → 发控制
线程B（GPU 提交, 绑核 4-7）:  从队列取 → ncnn extract() → 推回
                                  │
                              780M 核显实际计算在这里异步进行
```

**为什么是 2 线程而不是 3**：CPU 全部工作合计 ~2.4 ms，GPU 段 ~10 ms，瓶颈唯一。
CPU 内部再拆线程只省 ~0.2 ms 量级，却多两套队列与同步成本；把 CPU 压成一个线程，
卡尔曼/轨迹/时序等全部 CPU 侧状态**天然无锁**。

## 2. 槽位状态机

```
                 线程A                    线程B
                   │                       │
   FREE ──采集成──▶ IMG_READY ──push──────▶│
   ▲                                       ▶ ncnn 推理
   │                                       │
   └───归还──── PRED_READY ◀──push─────────┘
        ◀─ decode / KF / 消费
```

槽位状态由其所在队列表示，无独立状态字段：

| 队列 | 含义 | 写者 | 读者 |
|---|---|---|---|
| `q_free` | 可复用 | A（归还） | A（取去采集） |
| `q_gpu` | 等待 GPU | A | B |
| `q_back` | 推理完成 | B | A |

**不变式（实现时用于自查）**

1. 一个槽任意时刻只在一个队列里
2. 图像内存只有 A 写、blob 只有 A 写、pred 只有 B 写——数据面零锁，锁只保护队列
3. 槽总数恒为 3，系统内在途帧 ≤ 3

## 3. 队列语义与丢旧帧

`q_gpu` / `q_back` 为普通阻塞队列；`q_free` 特殊——空时不阻塞，从 `q_gpu`
偷最老槽位（其帧已过气），保证**处理最新帧**：

```cpp
Slot* acquire_slot() {
    Slot* s = nullptr;
    if (q_free.pop_oldest(s)) return s;   // 有空槽, 直接用
    if (q_gpu.pop_oldest(s)) { g_dropped++; return s; }  // 丢 GPU 队列里最老的帧
    return nullptr;                        // GPU 也空, 本轮放弃
}
```

由此得到实时性核心性质：**在途帧数恒 ≤ 3，端到端延迟上界 = 3 × 帧周期 ≈ 30 ms，
任何情况下不恶化。**

## 4. 线程A完整循环

```cpp
void thread_cpu() {
    while (g_run) {
        Slot* s = acquire_slot();
        if (!s) continue;

        if (!camera_grab(s->img.data(), s->img_w, s->img_h)) {  // 无新帧则跳过
            q_free.push(s);
            continue;
        }
        s->t_capture = now_us();               // 延迟统计时间戳

        x86::letterbox_normalize(s->img.data(), s->img_w, s->img_h,
                                 s->blob.data(), IN_W, IN_H);
        q_gpu.push(s);                          // 交给 GPU, 不等待

        Slot* done = q_back.try_pop();          // 取上一帧结果（通常已完成）
        if (done) {
            s->dets = x86::decode(s->pred.data(), g_tab, 2, CONF_THR);
            s->kept = x86::nms(std::move(s->dets), NMS_THR, TOPK);
            auto trks = g_tracker.predict();
            g_tracker.update(to_dets(s->kept));
            for (auto& t : trks)
                if (t.hits >= 2 && t.misses == 0) send_to_control(t);
            g_latency_us = now_us() - s->t_capture;
            q_free.push(s);
        }
    }
}
```

要点：A 把帧 N 推给 GPU 后**立即**处理帧 N-1 的结果，CPU 的 2.4 ms 工作塞进
GPU 10 ms 的空隙。若取回时帧 N-1 未完成，A 短暂阻塞——此机制吸收 GPU 抖动，
卡尔曼 predict 步外推兜底。

## 5. 稳态时间线

```
时刻(ms)   0      2.2    2.3        12.3      12.4   14.6   ...
线程A:  采集N 预处理N 推q_gpu   取回N-1   解码N-1 采集N+1 ...
线程B/GPU:              [==== 推理帧N-1 ====]
                         [====== 推理帧N ======]
```

- 单帧延迟（采集→出框）≈ 12 ms，与串行相同
- 吞吐由瓶颈段决定 = GPU 段 ≈ 10.3 ms/帧 ≈ 97 FPS

## 6. 边界与异常处理

| 情况 | 行为 |
|---|---|
| 相机无新帧 | 归还槽位，`continue`，不空转 |
| GPU 推理出错 | B 标记 error，A 丢弃该帧，卡尔曼纯 predict 外推一帧 |
| `q_back` 积压 | 取最老处理，丢新的（正常情况不该发生） |
| 关机 | `g_run=false` → B 退出 → A 排空 `q_back` → 干净退出，不杀线程 |
| 相机掉线 | A 进入 1 s 重连循环；轨迹 `misses` 递增，控制端可判视觉失效 |

## 7. 槽位数 = 3 的理由

延迟上界 = `NUM_SLOTS × GPU段耗时`。2 槽上界 ~20 ms 但无抖动余量；
3 槽上界 ~31 ms 且能吸收采集/GPU 正常抖动；再多每槽 +10 ms 延迟纯亏。
**写死为 3，不做配置项。**

## 8. 落地里程碑

```
M1  串行版        单线程跑通全链路, 打印延迟
M2  数值对齐      test_x86 golden 对齐全 PASS
M3  接入卡尔曼    串行版内接 Tracker, 验证跟踪连续性
M4  拆两线程      引入槽位/队列, 只改调度不改算法
M5  丢旧帧+统计  压测 P50/P99、吞吐、丢帧率
M6  实机调优      taskset / governor / UMA, 数字写回 README
```

M1→M4 每步仅将已验证代码换调度方式，出错易二分定位。

## 9. 验收指标（M5）

- 端到端延迟 P50 ≈ 12 ms，P99 < 25 ms
- 稳态吞吐 ≥ 95 FPS（120 fps 相机下丢帧率 < 20%）
- 静止场景 `g_dropped` ≈ 0（GPU 跟得上就不该丢）
- CPU 单线程占用 < 50%
