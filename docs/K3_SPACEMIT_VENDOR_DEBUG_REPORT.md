# 致 SpaceMIT：K3 EP 故障问题单

更新：2026-11-12。详细映射、版本和实验见 [根因报告](K3_ROOT_CAUSE_ANALYSIS_2026-11-12.md)，原始证据见 [压缩包](evidence/k3-2026-11-12.tar.gz)。

## 环境

K3 CoM260 IFX，Bianbu 4.0.8 / Linux 6.18.3，SpaceMIT ORT 2.0.7+2、Spine 0.6.3、TCM 3.0.1。模型输入 `[1,3,640,640]`，输出 `[1,17,8400]`。

## 问题与诉求

1. **Split 属性兼容**：opset 12 完整模型的 `[32,32,64]` 在 EP 子图被推导为 `[43,43,42]`，导致 `Reshape_139` 失败。转 opset 13 第二输入形式后正确。请检查旧版 split 属性进入 shape inference 的路径。
2. **FP16 TCM 越界读**：`ime2::spe_direct_conv2d+402`，库内 `0x3fb78e` 的 `vle16.v v3,(a3)` 读取至 384 KiB TCM 块末端。入口布局与 288 KiB 权重 +96 KiB 输入吻合；请核查 M=20/N=32/K=512 分块、padding/halo、stride 与容量规划。

系统 OpenCV + OpenBLAS 8 线程自然故障 6/10；厂商 OpenCV 可规避，但加 TCM 保护页后两套 OpenCV 各 3/3 崩溃，且厂商配置停在同一指令。因此初始化顺序和单线程是规避，不是根因修复。

请提供对应源码修复或含符号的配套 ORT/EP/Spine 包；验收同时检查保护页和输出数值，不能只统计冷启动成功。

## 单文件复现

无需项目业务二进制、OpenCV 或视频：

```bash
c++ -std=c++17 -O2 -pthread -Wl,--export-dynamic tools/reproduce_k3_tcm.cpp \
  -o /tmp/reproduce_k3_tcm -lonnxruntime -lspacemit_ep -ldl
/tmp/reproduce_k3_tcm --model ../data/best.spacemit.fp16.onnx --runs 3
```

程序先执行 baseline，再执行带保护页的进程。比较 `FAULT address` 与 `TCM_GUARD end`；程序汇总“复现成功”不表示推理成功。源码中 `--help` 列出只运行保护页等选项。

## 其他日志入口

```bash
SPACEMIT_EP_DUMP_SUBGRAPHS=1 FRAMES=1 ./run_k3_video.sh
SPACEMIT_EP_DEBUG_PROFILE=debug_ FRAMES=10 ./run_k3_video.sh
SPACEMIT_EP_DUMP_TENSORS=dump FRAMES=1 ./run_k3_video.sh
sudo dmesg | tail -80
```

`--load-only` 脱离视频检查，`--video-first` 恢复旧初始化顺序。`OrtInfer` 是项目适配层，故障首次推理是普通官方 `session->Run`，发生在业务后处理前。

历史仅替换 OpenBLAS 的 60 份试验实际为 `gotoblas` 符号缺失，未进入推理，应与本卷积越界问题分别处理。
