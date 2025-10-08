# LLM大模型部署指南

本项目提供了在A800服务器上部署32B大语言模型的完整解决方案，基于Qwen2.5-32B-Instruct模型构建高性能的API服务。

## 🚀 项目特性

- **双引擎支持**: 提供Transformers和VLLM两种推理引擎
- **高性能推理**: VLLM引擎提供2-5倍推理加速
- **RESTful API**: 提供标准化的HTTP API接口
- **流式输出**: 支持实时流式响应
- **自动部署**: 一键部署脚本，自动处理环境配置
- **监控管理**: 内置健康检查和系统监控
- **模型管理**: 支持多种Qwen系列模型下载和切换
- **生产就绪**: 支持systemd服务管理和日志记录

## 📋 系统要求

### 硬件要求
- **GPU**: NVIDIA A800 (推荐) 或其他24GB+显存的GPU
- **内存**: 32GB+ 系统内存
- **存储**: 100GB+ 可用空间（用于模型文件）

### 软件要求
- **操作系统**: Ubuntu 20.04+ / CentOS 8+ / RHEL 8+
- **Python**: 3.8+
- **CUDA**: 11.8+ / 12.1+
- **NVIDIA驱动**: 525+

## 🛠️ 快速部署

### 1. 克隆项目
```bash
git clone <your-repo-url>
cd llm-deployment
```

### 2. 下载模型
```bash
# 查看支持的模型
python model_download.py list

# 下载推荐的32B模型
python model_download.py download --model qwen2.5-32b

# 或者下载更轻量的14B模型
python model_download.py download --model qwen2.5-14b
```

### 3. 选择部署方案

#### 方案A: VLLM高性能部署 (推荐)
```bash
# 安装依赖
pip install -r requirements.txt

# 启动VLLM服务器
python vllm_server.py
```

#### 方案B: Transformers标准部署
```bash
# 安装依赖
pip install -r requirements.txt

# 启动Transformers服务器
python model_server.py
```

### 4. 配置系统服务
```bash
# 运行部署脚本
chmod +x deploy.sh
./deploy.sh
```

部署脚本将自动完成以下操作：
- 检查系统要求和依赖
- 创建Python虚拟环境
- 安装所需依赖包
- 配置systemd服务
- 设置防火墙规则
- 测试服务启动

### 5. 启动服务
```bash
# 启动服务
sudo systemctl start llm-server

# 设置开机自启
sudo systemctl enable llm-server

# 查看服务状态
sudo systemctl status llm-server

# 查看实时日志
sudo journalctl -u llm-server -f
```

## 🔧 手动部署

如果自动部署脚本遇到问题，可以按以下步骤手动部署：

### 1. 环境准备
```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装Python和pip
sudo apt install python3 python3-pip python3-venv -y

# 验证CUDA安装
nvidia-smi
```

### 2. 创建虚拟环境
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### 3. 安装依赖
```bash
# 安装PyTorch (CUDA版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install -r requirements.txt
```

### 4. 下载模型
```bash
# 创建模型目录
sudo mkdir -p /data/models
sudo chown $USER:$USER /data/models

# 下载模型 (约64GB)
huggingface-cli download Qwen/Qwen2.5-32B-Instruct --local-dir /data/models/Qwen2.5-32B-Instruct
```

### 5. 启动服务
```bash
python model_server.py
```

## 🤖 模型选择指南

### 模型来源
所有模型均从 **Hugging Face Hub** 下载，这是目前最权威和可靠的开源模型仓库：

- **官方地址**: https://huggingface.co/Qwen
- **模型系列**: Qwen2.5系列（最新版本）
- **许可证**: Apache 2.0（商业友好）

### 模型对比

| 模型 | 参数量 | 显存需求 | 性能等级 | 推荐场景 |
|------|--------|----------|----------|----------|
| Qwen2.5-7B | 7B | ~14GB | ⭐⭐⭐ | 资源受限环境 |
| Qwen2.5-14B | 14B | ~28GB | ⭐⭐⭐⭐ | 平衡选择 |
| **Qwen2.5-32B** | **32B** | **~64GB** | **⭐⭐⭐⭐⭐** | **A800推荐** |
| Qwen2.5-72B | 72B | ~144GB | ⭐⭐⭐⭐⭐ | 顶级性能 |

### 为什么选择Qwen2.5系列？

1. **最新技术**: Qwen2.5是2024年最新发布的模型系列
2. **中文优化**: 对中文理解和生成能力显著提升
3. **多语言支持**: 支持中、英、日、韩等多种语言
4. **指令遵循**: 优秀的指令理解和执行能力
5. **安全可靠**: 经过安全对齐，减少有害输出

### 关于Qwen3
目前Qwen3尚未正式发布，Qwen2.5是目前最新的稳定版本。当Qwen3发布后，我们会第一时间更新支持。

## 📚 API使用说明

### 基础信息
- **服务地址**: `http://your-server-ip:8000`
- **API文档**: `http://your-server-ip:8000/docs`
- **健康检查**: `http://your-server-ip:8000/health`

### 主要接口

#### 1. 健康检查
```bash
curl http://localhost:8000/health
```

响应示例：
```json
{
  "status": "healthy",
  "model_loaded": true,
  "gpu_memory": {
    "gpu_0": {
      "used": 20480,
      "total": 40960,
      "utilization": 50.0
    }
  },
  "system_memory": 45.2
}
```

#### 2. 聊天接口
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "你好，请介绍一下你自己"}
    ],
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

响应示例：
```json
{
  "response": "你好！我是一个基于Qwen2.5-32B-Instruct的AI助手...",
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 150,
    "total_tokens": 175
  },
  "model": "Qwen/Qwen2.5-32B-Instruct",
  "processing_time": 2.34
}
```

#### 3. 多轮对话
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "什么是机器学习？"},
      {"role": "assistant", "content": "机器学习是人工智能的一个分支..."},
      {"role": "user", "content": "请详细解释深度学习的原理"}
    ],
    "max_tokens": 300
  }'
```

#### 4. 流式输出 (仅VLLM支持)
```bash
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "请写一首关于春天的诗"}
    ],
    "stream": true
  }'
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| messages | List[Dict] | 必填 | 对话历史列表 |
| max_tokens | int | 2048 | 最大生成token数 |
| temperature | float | 0.7 | 控制随机性 (0-2) |
| top_p | float | 0.9 | 核采样参数 (0-1) |
| stream | bool | false | 是否流式输出 |

## 🧪 测试工具

项目提供了完整的测试客户端：

```bash
# 运行测试
python test_client.py
```

测试包括：
- 健康检查测试
- 基本对话功能测试
- 性能基准测试
- 交互式聊天界面

## ⚙️ 配置说明

### 模型配置 (`config.py`)

```python
class ModelConfig:
    # 模型选择
    MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"
    
    # 推理参数
    MAX_TOKENS = 2048
    TEMPERATURE = 0.7
    TOP_P = 0.9
    
    # 量化设置
    USE_QUANTIZATION = True
    QUANTIZATION_BITS = 4
    
    # 服务器设置
    HOST = "0.0.0.0"
    PORT = 8000
```

### 性能优化建议

1. **内存优化**:
   - 启用4-bit量化减少显存占用
   - 调整`MAX_MEMORY_USAGE`参数
   - 使用`device_map="auto"`自动分配设备

2. **推理加速**:
   - 使用VLLM引擎（可选）
   - 调整批处理大小
   - 优化输入长度

3. **并发处理**:
   - 增加worker进程数
   - 使用负载均衡器
   - 实现请求队列管理

## 🔍 监控和维护

### 日志管理
```bash
# 查看服务日志
sudo journalctl -u llm-server -f

# 查看应用日志
tail -f /var/log/llm_deployment.log
```

### 性能监控
- GPU使用率: `nvidia-smi`
- 内存使用: `htop`
- API响应时间: 通过健康检查接口

### 常见问题

#### 1. 模型加载失败
```bash
# 检查模型文件
ls -la /data/models/Qwen2.5-32B-Instruct

# 检查权限
sudo chown -R $USER:$USER /data/models
```

#### 2. GPU内存不足
```bash
# 启用更激进的量化
# 修改config.py中的量化参数
USE_QUANTIZATION = True
QUANTIZATION_BITS = 4
```

#### 3. 服务启动缓慢
- 首次启动需要加载模型，可能需要5-10分钟
- 检查磁盘I/O性能
- 考虑使用SSD存储模型文件

## 🔒 安全建议

1. **网络安全**:
   - 配置防火墙规则
   - 使用HTTPS（生产环境）
   - 限制API访问IP

2. **认证授权**:
   - 添加API密钥验证
   - 实现用户权限管理
   - 记录访问日志

3. **数据安全**:
   - 定期备份配置
   - 加密敏感数据
   - 监控异常访问

## 📈 扩展功能

### 1. 模型切换
支持动态切换不同模型：
```python
# 修改config.py
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"  # 切换到14B模型
```

### 2. 负载均衡
使用Nginx实现负载均衡：
```nginx
upstream llm_backend {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
}

server {
    listen 80;
    location / {
        proxy_pass http://llm_backend;
    }
}
```

### 3. 缓存机制
添加Redis缓存提高响应速度：
```python
import redis
redis_client = redis.Redis(host='localhost', port=6379, db=0)
```

## 🤝 贡献指南

1. Fork本项目
2. 创建特性分支
3. 提交更改
4. 发起Pull Request

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 📞 技术支持

如有问题，请通过以下方式联系：
- 提交Issue
- 发送邮件至: support@example.com
- 加入技术交流群

---

**注意**: 本部署方案针对A800服务器优化，在其他硬件环境下可能需要调整配置参数。
