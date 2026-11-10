# yolo_rt —— 实时检测跟踪流水线应用

实现 [PIPELNE](../docs/PIPELINE.md)：两线程 + 三槽位 + 丢旧帧。本目录是对算子库的**集成层**，
通过 `common/cpu_ops.h` 选择 `x86/` 或 `riscv/` 后端，模型推理由 ncnn 或
SpaceMIT ONNX Runtime（`backend: spacemit`）完成。

## 拓扑（PIPELINE.md §1/§2）

```
线程A（CPU, 默认绑核 0-3）:  采集 → letterbox → [推 q_gpu] → 取回上帧 → decode → NMS → Tracker(Kalman) → UDP 控制
线程B（推理, 默认绑核 4-7）:  q_gpu.pop → ncnn/SpaceMIT ORT → q_back.push
```

- 槽位写死 3 个（`slot.h: NUM_SLOTS`），状态由所在队列表示，在途帧 ≤ 3；
- `acquire_slot()` 丢旧帧：free 空时偷 `q_gpu` 最老槽（`pipeline.h`），端到端延迟上界 ≈ 3×GPU 段耗时；
- 线程A 独占 Tracker/Stats，CPU 侧状态天然无锁，锁只护队列（`block_queue.h`）。

## 构建

根目录 CMake 用 FetchContent 拉 ncnn（`YOLOOP_NCNN_VULKAN` 控制 Vulkan，
`-DNCNN_GIT_TAG=` 可换版本）：

```bash
cmake -S . -B build -DYOLOOP_BUILD_CUDA=OFF -DYOLOOP_BUILD_TRT_PLUGIN=OFF \
      -DYOLOOP_BUILD_PYTHON=OFF -DYOLOOP_X86_ARCH=znver4   # 目标机 7840HS；本机开发用 native
cmake --build build -j$(nproc)          # 产物: build/app/yolo_rt
```

K3 原生构建直接使用 `./build_k3.sh`。安装 `spacemit-onnxruntime` 后会自动启用
官方 SpaceMIT EP；否则只保留 ncnn CPU 后端。也可通过 `SPACEMIT_ORT_ROOT` 指定
解压后的 SDK 根目录。

随包 pose 模型无需改配置：

```bash
./run_k3_video.sh                     # SpaceMIT EP + 修复版 FP16 ONNX
BACKEND=ncnn ./run_k3_video.sh        # ncnn + best.ncnn.param/.bin
./build-k3/app/yolo_rt --config app/pipeline_k3_ort.yaml
```

ONNX 模型名称为 `images/output0`，ncnn 模型名称为 `in0/out0`，两套 YAML 已分别配置。

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

**模型类型（`model-type`）**：六种 head 契约（`common/head_types.h`）。

| 值 | 输出契约 | decode 算子 |
|---|---|---|
| `v5` | 切头导出，逐层 `[na*(5+nc), h, w]`（通道 anchor-major），`ncnn_infer` 重排为 `[4+1+nc, N]`（README §3），需 anchor 表 | `cpu::decode`（logit 快筛） |
| `v8` / `v9` / `v11` | Detect 头烘在图内，单输出 `[4+nc, N]`（xywh 像素 + 类别概率）；v9/v11 检测头与 v8 同构 | `cpu::decode_v8` |
| `v8-pose` | 同上 + `3×num-kpts` 关键点通道（x,y,conf 像素），点数任意 | `cpu::decode_v8_pose` |
| `v8-raw` / `v9-raw` / `v11-raw` | Detect 切在 DFL 前，逐层 `[4*reg_max+nc, h, w]`，网格表加载时推导（改输入尺寸只改 `in-size` 重新加载、免重新导出；前提是导出图内无固定空间尺寸的 reshape / 常量 grid 等静态 shape 算子） | `cpu::decode_af_raw`（DFL+dist2bbox） |
| `v8-dist` / `v9-dist` / `v11-dist` | DFL 期望留图内，逐层 `[4+nc, h, w]`（4 条边投影距离 + cls logit），网格表推导同 raw；grid+dist2bbox 在 C++（实验路径，见「DFL 切分点三路径对比」） | `cpu::decode_af_dist` |
| `v10` | 端到端 NMS-free `[N, 6]`（x1,y1,x2,y2,score,cls），box-major 自动转置，**不做 NMS** | `cpu::decode_e2e` |

`v8-raw` 系的 `reg-max`（默认 16）与 `num-classes` 决定校验通道数 `4*reg_max+nc`。

pose 模式下关键点随框过 NMS（`cpu::nms_indices` 索引重载），UDP 包附带被跟踪框
命中时的关键点（≤32 点；coast/丢失帧 `kpt_count=0`）。

**防大小不匹配**：v5 由 `in-size`/`num-classes`/anchor 表推导 N（另校验 blob 通道为
`na*(5+nc)` 整数倍，na 由通道数推导）；v8 系 N 由模型输出自动发现。`NcnnInfer::load`
加载时试跑一帧校验通道数与总数——把布局错误尽量前移到加载期，形状/通道数不匹配
启动即报错；但试跑不能验证 head 顺序、box/class 通道语义、anchor 排列、logit 是否
已 sigmoid，这些仍由模型契约约束。

- `--heads`：输出 blob 名列表。逐层契约（v5 / v8-raw 系）给三个，**按 stride 8→16→32
  顺序**，输出须为对应契约的原始卷积输出；fused / v10 通常单个拼接输出。
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

### 实测补充：7940HX + RTX 4060 Laptop（2026-10）

第二台实测机：Ryzen 9 7940HX（16C/32T，双 CCD）+ RTX 4060 Laptop + 核显 610M。
模型 `best.pt`（YOLO11s-pose，cube_face 1 类 4 关键点，ultralytics `format='ncnn'`
fp16 导出），视频 1447×1080，500 帧：

| 后端 | 模式 | infer P50 | 端到端 P50 | 吞吐 |
|---|---|---|---|---|
| Vulkan / RTX 4060 Laptop | 串行 | 11.2ms | 20.6ms | 上限 ≈49 FPS |
| Vulkan / RTX 4060 Laptop | 两线程 | 12.0ms | 16.2ms（P99 22.4） | **77.1 FPS** |
| 纯 CPU 16 线程 | 串行 | 50.6ms | 59.9ms | 上限 ≈16.7 FPS |
| 纯 CPU 8 线程 | 串行 | 47.3ms | 56.9ms | 上限 ≈17.3 FPS |
| 纯 CPU 16 线程 | 两线程（默认绑核 0-3/4-7） | 134.0ms | 144.4ms（P99 186.8） | 7.4 FPS |
| 纯 CPU 16 线程 | 两线程（`--cores-b 4-31`） | 87~90ms | — | 11.2 FPS |
| 纯 CPU 8 线程 | 两线程（默认绑核） | 111ms | — | 8.5 FPS |
| 纯 CPU 16 线程 | 两线程（完全不绑核 0-31/0-31） | 127ms | — | 7.7 FPS |

**纯 CPU 两线程异常排查结论**（7940HX 与 9700X 行为一致，非代码缺陷）：

- **元凶是默认 `cores-b: 4-7`**：该默认值是给 Vulkan 模式设计的（线程B 只提交
  GPU 命令）。纯 CPU 模式下推理本体就在线程B 执行（ncnn 走 libgomp，每个并行区
  的 master 即线程B），16 线程挤 4 个物理核 → infer 从 51ms 恶化到 134ms。
  纯 CPU 两线程必须放宽绑核：`--cores-b 4-31`（87ms，与 9700X 同型配置 84ms 一致）。
- **完全不绑核同样糟糕**（127ms）：线程A 的 OpenCV 解码线程池与 ncnn 的 OpenMP
  worker 满核互抢——隔离比数量重要。
- 即使绑核放宽，两线程 CPU 推理仍比串行慢 ~1.7×（51→87ms），与 9700X 的
  49→84ms 同型：OpenMP 静态切分下每个并行区都被最慢 worker 拖住。
  `OMP_WAIT_POLICY=passive` 无效，不是自旋问题。
- 串行模式 8 线程略快于 16 线程（47.3 vs 50.6ms）：7940HX 双 CCD + SMT，
  超过物理核半数后只剩争抢。**纯 CPU 部署用 `--serial --threads 8`。**

### 矩形推理（640×480）：610M 核显原型实测（2026-10-30）

模型侧优化（《Operator_Suggestion.md》附录 A ★★）：源画面为 4:3 / 16:9 时，方形
640×640 letterbox 的 padding 是纯浪费。按长边 640、短边对齐 32 重导矩形输入即可，
**app 零代码改动**（letterbox / unletterbox / v8 N 自动发现均已参数化），只需新模型
+ yaml 改 `in-size`。注意 DFL 网格烘在图内（anchor 网格是图里的常量），**fused
路径换输入尺寸必须重新导出模型**，运行时改不了；`v8-raw` 系（C++ 侧 DFL）的网格
表按输出形状加载时推导，输入尺寸不受此限制。

重导命令（imgsz 为 [H,W]）：

```python
YOLO('best.pt').export(format='ncnn', imgsz=[480, 640], half=True)  # 4:3 → 640×480
# 16:9（1920×1080 相机）：imgsz=[384, 640]，省 ~40% 像素
```

配置见 `app/pipeline_pose_rect.yaml`（与 `pipeline_pose.yaml` 仅模型路径与 `in-size` 之差）。

实测：7940HX + **Radeon 610M**（RADV，本机 ncnn 枚举 `--gpu 1`；枚举顺序不稳定，
以启动日志设备表为准），视频 1447×1080（bench 的 1920×1080 上限内缩放后），500 帧，
best.pt（YOLO11s-pose，fp16）：

| 输入 | 模式 | infer P50 | 端到端 P50 | 吞吐 |
|---|---|---|---|---|
| 640×640 | 串行 | 71.5ms | 81.0ms | 上限 12.4 FPS |
| 640×480 | 串行 | 53.6ms | 62.2ms | 上限 16.1 FPS |
| 640×640 | 两线程 | 78.0ms | 82.3ms（P99 164.0） | 12.0 FPS |
| 640×480 | 两线程 | 58.1ms | 62.6ms（P99 150.3） | 16.1 FPS |

结论：

- **infer 耗时比 = 0.75（串行）/ 0.75（两线程）**——与像素量比 640×480/640×640 = 0.75
  完全一致。conv 主导的网络耗时随输入像素线性缩放，矩形推理的收益可按像素比直接估算
  （16:9 导 640×384 时像素比 0.6，infer 预期 ~0.6×）。
- 数值对齐：矩形 ncnn 与 `best.pt` 同帧对拍（同 `imgsz=[480,640]`、`conf=0.25`），
  框数一致、坐标/关键点 max|Δ| ≤ 0.21px（fp16 容差内）。矩形下 letterbox 缩放系数与
  方形相同（长边主导），跟踪门控参数不用改。
- 吞吐 12.0 → 16.1 FPS（**+34%**）。610M 仅 2CU，绝对帧率只作原型参考；该相对收益
  在 780M 上同样成立（瓶颈同为大分辨率 conv，见《Operator_Suggestion.md》§12）。

### SPPF 并行化：数值逐位等价，610M 上实测性能中性（2026-10-30）

《Operator_Suggestion.md》§0.2.2 / 阶段 0：串行 3×MaxPool5（s=1）数学上等价于并行
MaxPool 5/9/13 + concat（max 复合无舍入）。已工具化为
`tools/export_ncnn_sppf.py`（替换前逐模块做 exact-0 断言，不等价即中止）：

```bash
python tools/export_ncnn_sppf.py best.pt --imgsz 480 640 \
    --out-dir ../data/best640x480_sppf_ncnn_model
```

验证（同视频 3 帧，conf=0.25）：并行 vs 串行矩形模型 **框/关键点 Δ = 0.000px**
（逐位一致）；并行 vs best.pt Δ ≤ 0.21px（fp16 容差）。yolo_rt 冒烟通过
（`in=640x480, N=6300`）。

性能（610M，同视频 500 帧，与上表矩形行对照）：

| 输入 | 模式 | infer P50 | 端到端 P50 | 吞吐 |
|---|---|---|---|---|
| 640×480 串行SPPF | 串行 | 53.6ms | 62.2ms | 上限 16.1 FPS |
| 640×480 并行SPPF | 串行 | 53.8ms | 62.3ms | 上限 16.0 FPS |
| 640×480 串行SPPF | 两线程 | 58.1ms | 62.6ms（P99 150.3） | 16.1 FPS |
| 640×480 并行SPPF | 两线程 | 58.4ms | 63.5ms（P99 138.0） | 16.0 FPS |

结论：**数值上严格等价、无回归，但 610M + ncnn Vulkan 上提速 <0.5%（淹没在
运行间噪声内）**。原因：SPPF 在 stride-32（20×15×256）上，三个串行小 pool 的绝对
耗时本就可忽略，dispatch 串行化在 53ms 的整体推理里调度开销占比太小。该改写的
价值在于图更干净（并联无依赖）且等价性有断言背书，对 NPU/编译器后端可能更有
意义；在 ncnn/Vulkan 路线上不必为其单独重导。**矩形推理才是真正的收益来源。**
配置：`app/pipeline_pose_rect_sppf.yaml`。

### 真实模型 raw 导出与多尺寸动态验收（2026-11）

工具链：`tools/export_af_raw.py`（v8/v9/v11 detect 头 pre-dfl / post-dfl 两种切分
导出 ncnn；**必须先过 onnxsim**，未化简的图会让 pnnx 段错误——yolo11 实测复现；
pnnx 把多输出 blob 重命名为 `out0..out2`）+ `tools/accept_raw_dynamic.py`
（导出 → golden 生成 → C++ `test_raw_model` 对拍 → ultralytics predict 交叉核对，
一键复现）+ `app/tests/test_raw_model.cpp`（多尺寸 `NcnnInfer::load` N 自动发现、
raw 张量对拍、`decode_af_raw`+NMS 对拍；贴阈值边缘带 score∈[conf, conf+0.05)
的框允许 fp16/图融合抖动出没，硬性对拍只约束严格框）。

实测（yolov8n / yolo11n，bus.jpg，conf=0.25，fp32 导出 + fp16 存储推理）：

- **v8（纯 conv 骨干）**：同一 param/bin 在 640×640 / 640×480 / 992×544 下
  N 自动发现正确（8400 / 6300 / 11067），raw 张量 max|Δ| ≤ 0.165（fp16 存储
  级别），严格框坐标 max|Δ| ≤ 0.09px，与原始 PyTorch Detect 交叉核对一致。
  **raw 契约的"改尺寸免重导"在 v8 上已验证成立；v9 待验证（见下）。**
- **v11（C2PSA attention，结论绑定当前工具链）**：YOLO11 经当前
  PyTorch → ONNX → onnxsim → pnnx → ncnn 链路导出时，attention 内
  `view(B,C,H,W)` 把空间尺寸烘为常量（ncnn param 中 `Reshape 0=400/0=20`
  系列），同一模型跨尺寸推理数值错误（raw max|Δ| 达 22.8，框数错），因此
  **当前工具链下需逐尺寸导出**。这不是 attention 在数学上不能动态——
  动态恢复路径（dynamo 符号 shape 导出 + pnnx `inputshape2`）已实测不可行：
  pnnx 对命名符号维 `stoi` 崩溃。逐尺寸导出、逐尺寸对拍全部通过
  （坐标 max|Δ| ≤ 0.09px；attention 使 raw 张量误差偏大，≤0.69，语义层面框一致）。
- v9：检测头与 v8 同构、GELAN 骨干纯 conv 无 attention，**跨尺寸能力待验证**
  （未实测，无官方权重验收；不在 v8 已验证结论内）。

**支持矩阵**（验收状态以本表为准）：

| 路径 | 契约测试（合成） | 真实固定尺寸 | 真实跨尺寸（同 param/bin） |
|---|---|---|---|
| v5 raw | 已通过（重排全量比对 + planted decode + 反例） | 待验收（暂无 v5 导出工具） | 暂不属于本次范围 |
| v8 raw | 已通过 | 已通过（≤0.09px） | 已通过（640×640 / 640×480 / 992×544） |
| v9 raw | 可复用 v8 契约 | 待验收 | 待验证 |
| v11 raw | 已通过 | 已通过（逐尺寸导出，≤0.09px） | 当前工具链下失败，需逐尺寸导出 |
| AfDist | 已通过（case12 golden） | 已通过（数值抽查 + bench 全链路） | 当前导出不具备该优势（见下） |

**验收环境（已固定，偏离需重验）**：torch 2.14.1 / ultralytics 8.4.178 /
onnx 1.23.2 / onnxsim 0.8.1 / onnxscript 0.7.2 / pnnx+ncnn 1.0.20260526
（完整清单与注释见 `requirements.txt`）。导出器版本是验收环境的
一部分：onnxsim 化简顺序影响 pnnx 是否段错误、命名符号维让 pnnx inputshape2
路径崩溃，均为本版本组合下的实测行为（脚本运行时打印 `[toolchain]` 实际版本）。

### DFL 切分点三路径对比（同权重 yolov8n，610M 核显）

第三条切分路径 `v8-dist`（`x86::decode_af_dist`）：DFL 期望留图内、逐层
`[4+nc, h, w]`，C++ 只做 grid+dist2bbox。输出体量：

| 路径 | 逐层通道 | N=8400 fp32 输出 | 换尺寸 |
|---|---|---|---|
| fused（图内全 decode） | —（单输出 [4+nc, N]） | 84×8400×4B ≈ 2.82MB | 必须重导 |
| raw（pre-dfl） | 4×16+80=144 | 144×8400×4B ≈ 4.84MB（nc=80 时 1.7×；nc=3 时 67/7 ≈ 9.6×） | v8/v9 免重导 |
| dist（post-dfl） | 4+80=84 | ≈ 2.82MB | 必须重导（图内 reshape 烘尺寸） |

610M + Vulkan fp16、1447×1080 视频 300 帧串行（第二轮实测，运行间噪声 ±10%）：

| 路径 | infer(GPU, 含 extract+拷贝) | decode+nms | 端到端 P50 近似 |
|---|---|---|---|
| fused | 6.03ms | 0.316ms | 9.60ms（≈104 FPS 上限） |
| raw | 7.71ms | 0.100ms | 11.20ms（≈89 FPS 上限） |
| dist | 7.08ms | 0.098ms | 10.58ms（≈95 FPS 上限） |

结论：

- decode 微基准（case9/case12，N=8400）：`decode_af_raw` 110.6µs vs
  `decode_af_dist` 5.0µs——**DFL 是 raw 侧 decode 的绝对大头**；但在端到端里
  它被 infer/extract/拷贝差异淹没（raw 与 dist 的 decode+nms 都 ~0.1ms，
  差距不及 infer 段的 1/10）。
- **本机上 DFL 应放图内**：fused 端到端最快；raw 慢 ~17%（144 通道的提取+拷贝
  是主因）；dist 居中。**"dist 两头不占"仅限当前实现**：当前 post-dfl 导出把
  H/W 烘为常量（同 v11 的工具链限制），换尺寸须重导，而性能上又不占优，故保留
  为实验路径。理论上 shape-safe 的逐层 DFL 投影可以只输出 `[4+nc, h, w]` 且把
  grid 留在图外，同时保住小输出与动态性；既然本机已无性能收益，除非以后特别
  需要这条动态路径，不再为此投入。
- 该结论与「核显共享内存下拷贝代价敏感」一致；独显、低阈值、密集场景（候选框
  上千、NMS/decode 占比上升）结论可能反转——按 `bench_pipeline` 同法复测，
  不要只看 decode 微基准。

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
- 覆盖补全（不影响现有结论可信度）：v5 raw 真实模型对拍（需先补 v5 导出工具，
  同 export_af_raw.py 模式）、v9 真实模型固定尺寸 + 跨尺寸验收（当前标"待验证"）；
- M5 验收指标（P50≈12ms / P99<25ms / ≥95FPS）需在目标机+真模型上复测。
