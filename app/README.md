# yolo_rt —— 实时检测跟踪流水线应用

实现 [PIPELNE](../docs/PIPELINE.md)：两线程 + 三槽位 + 丢旧帧。本目录是对算子库的**集成层**，
不改动 `common/ x86/ nvidia/ vulkan/` 任何代码。

## 拓扑（PIPELINE.md §1/§2）

```
线程A（CPU, 默认绑核 0-3）:  采集 → letterbox → [推 q_gpu] → 取回上帧 → decode → NMS → Tracker(Kalman) → UDP 控制
线程B（GPU 提交, 默认绑核 4-7）:  q_gpu.pop → ncnn extract()（Vulkan fp16）→ q_back.push
```

- 槽位写死 3 个（`slot.h: NUM_SLOTS`），状态由所在队列表示，在途帧 ≤ 3；
- `acquire_slot()` 丢旧帧：free 空时偷 `q_gpu` 最老槽（`pipeline.h`），端到端延迟上界 ≈ 3×GPU 段耗时；
- 线程A 独占 Tracker/Stats，CPU 侧状态天然无锁，锁只护队列（`block_queue.h`）。

## 构建

根目录 CMake 用 FetchContent 拉 ncnn（Vulkan ON，`-DNCNN_GIT_TAG=` 可换版本）：

```bash
cmake -S . -B build -DYOLOOP_BUILD_CUDA=OFF -DYOLOOP_BUILD_TRT_PLUGIN=OFF \
      -DYOLOOP_BUILD_PYTHON=OFF -DYOLOOP_X86_ARCH=znver4   # 目标机 7840HS；本机开发用 native
cmake --build build -j$(nproc)          # 产物: build/app/yolo_rt
```

## 运行

所有参数集中在 YAML 配置文件（扁平 `key: value`，key 与 CLI 长选项同名）：

```bash
cp app/pipeline.example.yaml app/pipeline.yaml    # 按你的模型改
./build/app/yolo_rt --config app/pipeline.yaml
./build/app/yolo_rt --config app/pipeline.yaml --conf 0.35   # CLI 覆盖配置文件
./build/app/yolo_rt --config app/pipeline.yaml --dump-config # 打印最终生效配置
```

覆盖顺序：默认值 < 配置文件 < CLI。配置项分组（详见 `app/pipeline.example.yaml`
注释）：模型（`param/model/in0/heads/in-size/num-classes/gpu`）、anchor 表
（`anchors-s8/s16/s32`，不给 = YOLOv5s P5 默认）、检测阈值（`conf/nms/topk`）、
跟踪（`gate-px/gate-scale/size-ratio/size-memory/coast/max-misses`）、运行
（`serial/cores-a/cores-b/max-w/max-h`）、控制（`udp`）。

**模型类型（`model-type`）**：

| 值 | 输出契约 | decode 算子 |
|---|---|---|
| `v5` | 切头导出，3 个 head `[4+1+nc, N]`（README §3），需 anchor 表 | `x86::decode`（logit 快筛） |
| `v8` | Detect 头烘在图内，单输出 `[4+nc, N]`（xywh 像素 + 类别概率） | `x86::decode_v8` |
| `v8-pose` | 同上 + `3×num-kpts` 关键点通道（x,y,conf 像素），点数任意 | `x86::decode_v8_pose` |

pose 模式下关键点随框过 NMS（`x86::nms_indices` 索引重载），UDP 包附带被跟踪框
命中时的关键点（≤32 点；coast/丢失帧 `kpt_count=0`）。

**防大小不匹配**：v5 由 `in-size`/`num-classes`/anchor 表推导 N；v8 系 N 由模型输出
自动发现。`NcnnInfer::load` 加载时会试跑一帧校验通道数（与总数）——模型与配置
不匹配时启动即报错，不会带病运行。

- `--heads`：三个输出 blob 名，**按 stride 8→16→32 顺序**；输出须为 README §3 契约的
  原始卷积输出。
- `serial: true`：M1/M3 串行模式（单线程全链路），用于数值对齐与验收基线。
- `no-model: true` + `fake-gpu-ms`：无模型冒烟（线程B 假推理），验证调度/关机路径。
- 采集源目前只有接口（`camera_source.h`）+ `NullSource`：恒无帧 → 走 PIPELINE.md §6
  "相机掉线 1s 重连"路径。接真实相机时实现 `CameraSource` 替换 `main.cpp` 里的 `NullSource` 即可。

## 跟踪器（`tracker.h/.cpp`）

从 `RK3588-AntiUAV-Tracker` 提取的卡尔曼逻辑，C++ 精确移植（内部 double，与 Python
float64 对齐）：

- `BoxKalman` ← `tracker/run.py:39-70`：8D 等速 `[cx,cy,w,h,vx,vy,vw,vh]`，dt=1 帧，
  Joseph form 更新，内部状态不做图像裁剪（出界可外推）；
- `Tracker` ← `tracker/track3.py:102-168`（YoloKalman）：尺寸记忆过滤
  （`--size-ratio 3`, 120 帧记忆）、最近中心关联 + misses 自适应门限
  （`--gate-px 120 --gate-scale 2`）、`--coast 5` 帧后 LOST、SEARCH/TRACK/COAST/LOST
  状态机，事件 INITIAL_LOCK/REACQUIRED/LOST；
- 数值对齐：`python3 tools/gen_tracker_golden.py` 生成 golden（直接 import 原版
  BoxKalman），`./build/app/test_tracker test_data` 回放比对（误差 <1e-3）。

## 控制输出（`control_tx.h/.cpp`）

UDP 非阻塞发送，仅发确认轨迹（`hits>=2 && misses==0`）。包格式（小端 POD）：

```
u32 magic('YT') | u32 frame_id | u32 state(0 SEARCH/1 TRACK/2 COAST/3 LOST) |
f32 cx cy w h | f32 vx vy vw vh | f32 score |
u32 kpt_count | f32 kpts[32*3]    # pose：命中框关键点 (x,y,conf)，非 pose/无命中 = 0
```

## 速度基准（`bench_pipeline`）

离线视频全链路测速（无需相机），串行逐阶段 / 两线程流水线两种模式：

```bash
./build/app/bench_pipeline --param ../data/best.ncnn.param --model ../data/best.ncnn.bin \
    --model-type v8-pose --heads out0 --num-classes 1 --num-kpts 4 \
    --video ../data/xxx.webm --frames 500 [--serial] [--gpu 1] [--threads 8]
```

- `--gpu N` 选 Vulkan 设备（启动日志有设备表），`--gpu -1` = 纯 CPU（配合 `--threads`）。
  **注意：Vulkan 设备枚举顺序不稳定**（同一台机两次运行 [0] 可能是核显也可能是
  独显），选卡前务必看启动日志里的设备表再定 N；
- `--serial` 逐阶段计时 letterbox / infer / decode+nms / tracker；默认两线程模式报
  分阶段耗时 + 端到端延迟 P50/P99 + 吞吐 FPS + 丢帧数。

参考数字（开发机 9700X + RTX 3090Ti，**非目标机 7840HS/780M**；best.pt 自训 pose，
cube_face 1 类 4 关键点，输入 1447×1080→640，500 帧）：

| 后端 | infer P50 | 端到端 P50 | 吞吐 |
|---|---|---|---|
| Vulkan / RTX 3090Ti（串行） | 9.5ms | 15.3ms | 上限 ≈65 FPS |
| Vulkan / RTX 3090Ti（两线程） | 10.6ms | 14.2ms | 91.4 FPS，丢帧 364/500* |
| Vulkan / 核显 RADV 2CU（串行） | 77.7ms | 83.9ms | 上限 ≈12 FPS |
| Vulkan / 核显 RADV 2CU（两线程） | 83.6ms | 88.3ms | 11.5 FPS，丢帧 288/300* |
| 纯 CPU 8 线程（串行） | 49.2ms | 56.7ms | 上限 ≈18 FPS |
| 纯 CPU 8 线程（两线程） | 83.9ms | 89.9ms | 12.1 FPS，丢帧 285/300* |

*视频文件解码远快于推理，丢旧帧策略主动丢弃积压帧（符合 PIPELINE.md §4 设计）；
接真实相机时采集速率即输入速率，不会这样丢。

CPU 两线程注意：infer 从串行的 49ms 涨到 84ms——线程 A 的 webm 解码（VP9）
+ letterbox 与 ncnn 的 8 线程推理抢核/带宽，吞吐反而低于串行上限（12 vs 18 FPS）。
`--threads 4` 几乎相同（12.6 FPS），说明已瓶颈在带宽/争抢而非线程数。真实相机
源线程 A 负载轻得多，劣化会小；**CPU 后端部署时建议直接用 serial 模式**。

decode+nms ≈0.01ms、tracker ≈0.002ms，纯算子开销可忽略，瓶颈全在推理。

官方 `yolo11n-pose.pt`（1 类 17 关键点，C=56=4+1+3×17）同链路验证通过——只改
`num-kpts` 即可，点数任意：串行 infer P50 6.8ms、端到端 12.3ms，上限 ≈80 FPS；
两线程 infer P50 7.5ms、端到端 P50 10.8ms、吞吐 125 FPS（3090Ti）。

## 里程碑对应（PIPELINE.md §8）

| 里程碑 | 落地 |
|---|---|
| M1 串行版 | `--serial` 模式 |
| M2 数值对齐 | `x86/test_x86` + `app/test_tracker`（golden） |
| M3 接入卡尔曼 | `Tracker`（串行模式内验证连续性） |
| M4 拆两线程 | 默认模式，`--no-model --fake-gpu-ms` 可离线验证调度 |
| M5 丢旧帧+统计 | `app/test_queue`（不变式/丢旧帧）+ `[stats]` 周期打印 P50/P99/吞吐/丢帧 |
| M6 实机调优 | 不在本仓库：taskset/governor/UMA 到 7840HS 目标机执行 |

## 遗留项

- 真实相机采集实现（接口已就位）；
- 真实模型（best.pt pose → ncnn）已经 bench_pipeline 视频链路验证；相机端到端待采
  集实现；
- M5 验收指标（P50≈12ms / P99<25ms / ≥95FPS）需在目标机+真模型上复测。
