# K3 SSH/SCP 移交指南

## 1. 准备信息

移交时准备以下信息，**不要把密码写入脚本、ZIP 或 GitHub**：

- K3 IP：`<K3_IP>`
- 用户名：`epoch`
- 密码：向设备负责人获取
- 本地文件：`K3_YOLO_test.zip`

K3 与 PC 应能通过网络互通，并开放 SSH 服务。

## 2. 连接 K3

```bash
ssh epoch@<K3_IP>
```

首次连接时确认主机指纹，然后输入设备密码。看到类似以下提示符即表示连接成功：

```text
epoch@...:~$
```

退出：

```bash
exit
```

### 推荐：配置免密登录

在 PC 上生成密钥（已有密钥则跳过）：

```bash
ssh-keygen -t ed25519
```

把公钥安装到 K3：

```bash
ssh-copy-id epoch@<K3_IP>
```

之后可直接：

```bash
ssh epoch@<K3_IP>
```

不要把私钥复制到 K3 或提交到仓库。

## 3. 上传部署包

在 PC 上执行：

```bash
scp /home/mtftau-5/yolo-RGBT/K3_YOLO_test.zip \
  epoch@<K3_IP>:/home/epoch/work/
```

然后登录 K3 并解压：

```bash
ssh epoch@<K3_IP>
cd /home/epoch/work
unzip K3_YOLO_test.zip
```

目录应保持：

```text
work/
├── YOLO_operator_RC/
└── data/
```

权重和视频路径都依赖这个相对结构，不要随意移动 `data/` 中的文件。

## 4. 安装依赖并运行

在 K3 上执行：

```bash
cd /home/epoch/work/YOLO_operator_RC
chmod +x build_k3.sh run_k3_video.sh install_k3_dependencies.sh
./install_k3_dependencies.sh
./build_k3.sh
./run_k3_video.sh
```

默认使用：

- SpaceMIT ONNX Runtime；
- `/opt/opencv-spacemit` 的 RVV OpenCV；
- `/opt/openblas-spacemit` 的 RVV OpenBLAS；
- `best.spacemit.fp16.onnx`。

ncnn 基线对照：

```bash
BACKEND=ncnn ./run_k3_video.sh
```

## 5. 连接失败排查

检查网络：

```bash
ping <K3_IP>
```

查看 SSH 详细过程：

```bash
ssh -vvv epoch@<K3_IP>
```

常见原因：

- IP 或用户名错误；
- K3 的 SSH 服务未启动；
- PC 与 K3 不在同一网络或防火墙阻止 22 端口；
- 密码错误；
- 主机密钥发生变化，需要核对指纹。

如果只需执行一条远程命令，不必先登录：

```bash
ssh epoch@<K3_IP> 'cd /home/epoch/work/YOLO_operator_RC && ./run_k3_video.sh'
```
