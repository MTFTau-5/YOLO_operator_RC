# K3 SSH 与部署移交

准备设备 IP、用户名 `epoch`、单独交付的密码，以及保持同级结构的 `YOLO_operator_RC/` 和 `data/`。密码不要写进脚本、文档或仓库。

```bash
ssh epoch@<K3_IP>
# 首次连接核对指纹，按提示输入密码。
```

推荐用 `ssh-keygen -t ed25519` 生成密钥，再执行 `ssh-copy-id epoch@<K3_IP>`。私钥留在本机。

```bash
scp K3_YOLO_test.zip epoch@<K3_IP>:/home/epoch/work/
ssh epoch@<K3_IP>
cd /home/epoch/work
unzip K3_YOLO_test.zip
# 按实际解压目录进入项目：
cd YOLO_operator_RC
./install_k3_dependencies.sh
./build_k3.sh
./run_k3_video.sh
BACKEND=ncnn ./run_k3_video.sh
```

部署目录若多了一层 `k3/`，按实际路径进入；程序依赖同级 `data/`。默认库、模型和参数见 [README](../README.md)。

连接失败先核对 IP、网络和 SSH 服务，再用 `ssh -vvv epoch@<K3_IP>` 检查；主机密钥变化应先核实设备。退出远程会话用 `exit`。

K3 当前 EP 的越界问题及可交付复现见 [调试摘要](K3_DEBUG_SUMMARY.md)。冷启动成功不代表厂商问题已修复。
