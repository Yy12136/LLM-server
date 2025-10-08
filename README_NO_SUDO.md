# LLM大模型部署指南 (无sudo权限版本)

## 🚀 快速开始

### 1. 上传项目文件
```bash
# 将项目文件上传到服务器
scp -r . user@server:/path/to/project/
```

### 2. 登录服务器
```bash
ssh user@server
cd /path/to/project
```

### 3. 运行部署脚本
```bash
chmod +x remote_deploy_no_sudo.sh
./remote_deploy_no_sudo.sh
```

## 📋 部署流程

### 自动部署 (推荐)
运行 `remote_deploy_no_sudo.sh` 脚本，它会自动完成：

1. ✅ 检查系统环境和硬件要求
2. ✅ 创建Python虚拟环境
3. ✅ 安装所有依赖包
4. ✅ 设置HuggingFace认证
5. ✅ 下载模型到用户目录
6. ✅ 创建启动脚本
7. ✅ 测试服务启动

### 手动部署
如果自动脚本遇到问题，可以手动执行：

```bash
# 1. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 2. 安装依赖
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 3. 创建目录
mkdir -p ~/models ~/logs

# 4. HuggingFace认证
huggingface-cli login

# 5. 下载模型
python model_download.py download --model qwen2.5-32b --base-dir ~/models

# 6. 启动服务
python vllm_server.py
```

## 🔧 服务管理

### 启动服务
```bash
# 前台启动
./start_server.sh

# 后台启动
./start_server_daemon.sh
```

### 停止服务
```bash
./stop_server.sh
```

### 查看日志
```bash
tail -f ~/logs/server.log
```

## 🌐 API使用

### 健康检查
```bash
curl http://your-server-ip:8000/health
```

### 聊天接口
```bash
curl -X POST http://your-server-ip:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "你好，请介绍一下你自己"}
    ],
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

### Python客户端
```python
import requests

response = requests.post("http://your-server-ip:8000/chat", json={
    "messages": [{"role": "user", "content": "什么是人工智能？"}],
    "max_tokens": 300
})

print(response.json()["response"])
```

## 📁 目录结构

```
项目目录/
├── venv/                    # Python虚拟环境
├── ~/models/               # 模型文件目录
├── ~/logs/                 # 日志文件目录
├── start_server.sh         # 启动脚本
├── start_server_daemon.sh  # 后台启动脚本
├── stop_server.sh          # 停止脚本
└── config_no_sudo.py       # 配置文件
```

## ⚠️ 重要注意事项

### 1. 权限限制
- 无法创建系统服务，需要手动启动
- 无法修改防火墙，需要联系管理员开放8000端口
- 模型和日志存储在用户目录下

### 2. 资源要求
- **GPU内存**: 建议20GB+ (32B模型需要约64GB)
- **系统内存**: 建议16GB+
- **磁盘空间**: 建议100GB+ (模型文件约64GB)

### 3. 网络要求
- 需要稳定的网络连接下载模型
- 需要联系管理员开放8000端口供外部访问

### 4. 模型选择
```bash
# 根据资源情况选择模型
qwen2.5-7b    # ~14GB, 适合资源受限环境
qwen2.5-14b   # ~28GB, 平衡选择
qwen2.5-32b   # ~64GB, 推荐A800服务器
```

## 🔍 故障排除

### 1. 模型下载失败
```bash
# 检查网络连接
ping huggingface.co

# 重新认证
huggingface-cli login

# 检查磁盘空间
df -h ~
```

### 2. 服务启动失败
```bash
# 检查GPU状态
nvidia-smi

# 检查端口占用
netstat -tlnp | grep 8000

# 查看详细日志
tail -f ~/logs/server.log
```

### 3. 内存不足
```bash
# 检查内存使用
free -h
nvidia-smi

# 使用更小的模型
# 修改config_no_sudo.py中的MODEL_NAME
```

### 4. 网络访问问题
```bash
# 检查服务是否启动
curl http://localhost:8000/health

# 检查防火墙 (需要管理员权限)
# 联系管理员开放8000端口
```

## 🚀 使用screen保持服务稳定

```bash
# 安装screen (如果没有)
# 联系管理员安装: sudo apt install screen

# 创建screen会话
screen -S llm-server

# 在screen中启动服务
./start_server.sh

# 分离会话 (Ctrl+A, D)
# 重新连接会话
screen -r llm-server
```

## 📞 获取帮助

如果遇到问题：

1. **查看日志**: `tail -f ~/logs/server.log`
2. **检查进程**: `ps aux | grep vllm_server`
3. **检查端口**: `netstat -tlnp | grep 8000`
4. **联系管理员**: 需要系统级权限时

---

**注意**: 本指南适用于没有sudo权限的普通用户，所有操作都在用户权限范围内进行。
