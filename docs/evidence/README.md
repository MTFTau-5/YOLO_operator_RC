# K3 验证证据

- `k3-2026-11-12.tar.gz`：原始根因核查的 GDB、TCM 探针、依赖、矩阵实验与子图。
- `k3-a100-validation.tar.gz`：A100 全量 golden、worker/并发测试、完整应用构建、边界用例及 EP/ncnn 视频日志。
- `k3-cleanup-validation.tar.gz`：本次精简后的 K3 单/八线程 golden 日志、单文件复现日志和边界测试源码。

本次 K3 两组 golden 均 `ALL PASS`；边界测试按标量参考逐项比较数量、标签、坐标、score 和候选顺序，通过 9 个长度 ×2 个阈值的 v5/raw/dist 检查。本机标量回退在启用/关闭 OpenMP 时均通过已有 golden。

复现日志显示 baseline 输出 `[1,17,8400]`，保护页进程退出 139，故障地址等于 TCM 映射末端。该厂商故障仍存在。

测试在临时目录编译执行。Python requirements 已检查语法；本轮未新建并安装完整导出环境，也未重做 CUDA/TRT 构建。依赖版本保留原有导出链路约束。

A100 迁移验收：全部 golden、9 长度×2 阈值保序检查、worker VLEN=1024、同步/嵌套/并发/异常传播、Tracker/队列/head 布局测试通过。EP 串行 100 帧、两线程提交 30 帧（处理 15、丢弃 15）及 ncnn 串行 5 帧完成。使用完整 CMake 构建的程序另跑 10 帧通过；构建复用了设备已有的 ncnn 静态库。
