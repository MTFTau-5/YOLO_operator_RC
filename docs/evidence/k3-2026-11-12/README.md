# 2026-11-12 K3 复核证据

本目录为设备临时目录 `/tmp/k3-rootcause-20261112` 中的日志/诊断源码副本，以及本地生成的 EP 子图摘要和数值检查结果。不包含密码、完整模型、系统库或编译后的诊断二进制。原部署没有修改。

主要入口：

- `matrix.json` / `matrix.py`：6 组各 10 次冷启动对照，返回值 -11 为 SIGSEGV。
- `entry-1.txt`：在卷积函数第一条指令设置断点，记录入口参数和自然崩溃。
- `guard-vendor-gdb.txt`：厂商 OpenCV + 保护页，停在同一卷积读指令。
- `guard-results.json`：6 次保护页实验返回码；对应 `guard-*.txt` 为进程输出。
- `tcm_guard.c`：仅为本次诊断构建的 mmap 保护页探针，非部署修复。
- `tcm_trace.c` / `maptrace-*.txt`：仅记录 TCM 映射的探针及运行输出。
- `trace-entry.py`：GDB 入口记录器；应在 `start` 完成、EP 共享库已载入后 source。
- `full-opset12.log` / `full-opset13.log`：同权重完整 FP32 模型在 K3 的结果。
- `SpaceMIT*.onnx`：EP 导出的图结构快照，不是独立可运行的完整模型；外部权重未打包。
- `subgraph-facts.json`：从这两份快照提取的 Split 属性和输出形状。
- `opset-numerical-check.json`：本地 CPU ORT 的一个固定随机样本对照。

`split-*.onnx` / `split_probe.cpp` 是额外的最小图检查，未生成 EP 子图，不能用它们替代完整模型的 EP 对照。

## 保护页复现

在 K3 上使用同一版 EP/Spine/TCM 和原模型，编译探针：

```bash
gcc -shared -fPIC -O2 tcm_guard.c -o /tmp/k3-tcm-guard.so -ldl
```

先不设置 LD_PRELOAD 运行基线，再只给目标进程设置探针。以下命令从设备项目目录执行；它会主动暴露越界，预期可能 SIGSEGV。不要把此 LD_PRELOAD 加入部署脚本或 shell 全局配置。

```bash
OPENBLAS_NUM_THREADS=1 LD_PRELOAD=/tmp/k3-tcm-guard.so   taskset -c 0-7 ./build-k3/app/bench_pipeline   --backend spacemit --model ../data/best.spacemit.fp16.onnx   --in0 images --heads output0 --model-type v8-pose   --num-classes 1 --num-kpts 4   --video '../data/Screencast from 2026-10-13 20-00-08.webm'   --frames 1 --serial --gpu -1 --threads 8 --cores-a 0-7 --video-first
```

探针保留每个 0x60000 字节 TCM 映射的有效范围和 fd/offset，只在两侧增加不可访问页；保留的额外页随测试进程结束自动回收。该实现只用于短时定位，不适合长期运行。

完整结论见 `../../K3_ROOT_CAUSE_ANALYSIS_2026-11-12.md`。日志中的时间戳保留设备/工具原始时区，不代表已转换为用户时区。
