# LLM大模型部署故障排除指南

## 🔍 常见问题及解决方案

### 1. 模型下载问题

#### 1.1 HuggingFace认证失败
**问题**: `401 Unauthorized` 或 `Authentication required`

**解决方案**:
```bash
# 方法1: 使用huggingface-cli登录
huggingface-cli login
# 输入你的token: hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# 方法2: 设置环境变量
export HUGGINGFACE_HUB_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# 方法3: 在代码中直接使用token
python model_download.py download --model qwen2.5-32b --token YOUR_TOKEN

# 验证登录状态
huggingface-cli whoami
```

#### 1.2 网络连接问题
**问题**: 下载速度慢或连接超时

**解决方案**:
```bash
# 检查网络连接
ping huggingface.co

# 使用代理（如果有）
export https_proxy=http://your-proxy:port
export http_proxy=http://your-proxy:port

# 使用镜像站点
export HF_ENDPOINT=https://hf-mirror.com

# 断点续传
python model_download.py download --model qwen2.5-32b --resume
```

#### 1.3 磁盘空间不足
**问题**: `No space left on device`

**解决方案**:
```bash
# 检查磁盘空间
df -h

# 清理临时文件
sudo apt autoremove
sudo apt autoclean

# 移动模型到其他分区
sudo mkdir -p /mnt/large_disk/models
sudo chown $USER:$USER /mnt/large_disk/models
# 修改config.py中的MODEL_PATH
```

### 2. 服务启动问题

#### 2.1 GPU内存不足
**问题**: `CUDA out of memory` 或 `RuntimeError: CUDA error: out of memory`

**解决方案**:
```python
# 修改config.py，启用更激进的量化
USE_QUANTIZATION = True
QUANTIZATION_BITS = 4
GPU_MEMORY_FRACTION = 0.8  # 降低GPU内存使用率

# 或者使用更小的模型
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"  # 从32B改为14B
```

#### 2.2 模型文件损坏
**问题**: `FileNotFoundError` 或模型加载失败

**解决方案**:
```bash
# 检查模型文件完整性
python model_download.py verify --model qwen2.5-32b

# 重新下载模型
rm -rf /data/models/Qwen2.5-32B-Instruct
python model_download.py download --model qwen2.5-32b

# 检查文件权限
sudo chown -R $USER:$USER /data/models
```

#### 2.3 端口被占用
**问题**: `Address already in use` 或 `Port 8000 is already in use`

**解决方案**:
```bash
# 查找占用端口的进程
sudo netstat -tlnp | grep 8000
sudo lsof -i :8000

# 杀死占用进程
sudo kill -9 <PID>

# 或者修改端口
# 编辑config.py
PORT = 8001  # 改为其他端口
```

#### 2.4 依赖包版本冲突
**问题**: `ImportError` 或版本不兼容

**解决方案**:
```bash
# 重新创建虚拟环境
rm -rf venv
python3 -m venv venv
source venv/bin/activate

# 重新安装依赖
pip install --upgrade pip
pip install -r requirements.txt

# 如果仍有问题，尝试固定版本
pip install torch==2.1.2 transformers==4.44.2
```

### 3. 运行时问题

#### 3.1 推理速度慢
**问题**: 响应时间过长

**解决方案**:
```python
# 优化配置
# config.py
MAX_TOKENS = 1024  # 减少最大token数
BATCH_SIZE = 1     # 减少批处理大小
USE_QUANTIZATION = True  # 启用量化

# 使用VLLM引擎（如果可用）
python vllm_server.py  # 而不是model_server.py
```

#### 3.2 内存泄漏
**问题**: 长时间运行后内存使用持续增长

**解决方案**:
```bash
# 监控内存使用
watch -n 1 'free -h && nvidia-smi'

# 定期重启服务
sudo systemctl restart llm-server

# 设置内存限制
# 在systemd服务文件中添加
Environment=MALLOC_TRIM_THRESHOLD_=100000
```

#### 3.3 API请求超时
**问题**: 客户端请求超时

**解决方案**:
```python
# 增加客户端超时时间
client = LLMAPIClient("http://server:8000", timeout=600)  # 10分钟

# 或者使用流式输出
for chunk in client.chat(messages, stream=True):
    print(chunk, end="", flush=True)
```

### 4. 系统级问题

#### 4.1 CUDA版本不匹配
**问题**: `CUDA runtime error` 或版本不兼容

**解决方案**:
```bash
# 检查CUDA版本
nvidia-smi
nvcc --version

# 安装匹配的PyTorch版本
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 4.2 系统资源不足
**问题**: 系统响应缓慢或服务不稳定

**解决方案**:
```bash
# 检查系统资源
htop
iostat -x 1
nvidia-smi -l 1

# 优化系统配置
# 增加swap空间
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 调整系统参数
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
echo 'vm.dirty_ratio=15' | sudo tee -a /etc/sysctl.conf
```

### 5. 网络和安全问题

#### 5.1 防火墙阻止访问
**问题**: 无法从外部访问API

**解决方案**:
```bash
# Ubuntu/Debian
sudo ufw allow 8000/tcp
sudo ufw status

# CentOS/RHEL
sudo firewall-cmd --permanent --add-port=8000/tcp
sudo firewall-cmd --reload
sudo firewall-cmd --list-ports

# 检查服务绑定
netstat -tlnp | grep 8000
# 应该显示 0.0.0.0:8000 而不是 127.0.0.1:8000
```

#### 5.2 SSL/TLS证书问题
**问题**: HTTPS访问失败

**解决方案**:
```bash
# 生成自签名证书（测试用）
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# 配置Nginx反向代理
# /etc/nginx/sites-available/llm-api
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 6. 日志和调试

#### 6.1 查看详细日志
```bash
# 系统服务日志
sudo journalctl -u llm-server -f

# 应用日志
tail -f /var/log/llm_deployment.log

# 实时监控
watch -n 1 'tail -n 20 /var/log/llm_deployment.log'
```

#### 6.2 启用调试模式
```python
# config.py
LOG_LEVEL = "DEBUG"  # 改为DEBUG级别

# 或者在代码中添加调试信息
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### 6.3 性能分析
```bash
# GPU使用情况
nvidia-smi -l 1

# CPU和内存使用
htop

# 网络连接
netstat -an | grep 8000

# 磁盘I/O
iostat -x 1
```

### 7. 常见错误代码

| 错误代码 | 含义 | 解决方案 |
|---------|------|----------|
| 401 | 认证失败 | 检查HuggingFace token |
| 403 | 权限不足 | 检查文件权限和防火墙 |
| 404 | 服务未找到 | 检查服务是否启动 |
| 500 | 服务器内部错误 | 查看服务器日志 |
| 503 | 服务不可用 | 检查模型是否加载完成 |
| 504 | 网关超时 | 增加超时时间或优化模型 |

### 8. 预防措施

#### 8.1 定期维护
```bash
# 创建维护脚本
#!/bin/bash
# maintenance.sh

# 清理日志文件
find /var/log -name "*.log" -size +100M -delete

# 重启服务
sudo systemctl restart llm-server

# 检查磁盘空间
df -h

# 更新系统
sudo apt update && sudo apt upgrade -y
```

#### 8.2 监控脚本
```bash
#!/bin/bash
# monitor.sh

while true; do
    # 检查服务状态
    if ! curl -s http://localhost:8000/health > /dev/null; then
        echo "$(date): 服务异常，正在重启..."
        sudo systemctl restart llm-server
    fi
    
    # 检查GPU状态
    gpu_usage=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    if [ "$gpu_usage" -gt 95 ]; then
        echo "$(date): GPU使用率过高: ${gpu_usage}%"
    fi
    
    sleep 60
done
```

#### 8.3 备份策略
```bash
#!/bin/bash
# backup.sh

backup_dir="/backup/llm-deployment-$(date +%Y%m%d)"
mkdir -p $backup_dir

# 备份配置文件
cp config.py $backup_dir/
cp requirements.txt $backup_dir/
cp *.py $backup_dir/

# 备份服务配置
sudo cp /etc/systemd/system/llm-server.service $backup_dir/

# 压缩备份
tar -czf $backup_dir.tar.gz $backup_dir
rm -rf $backup_dir

echo "备份完成: $backup_dir.tar.gz"
```

### 9. 获取帮助

如果以上解决方案都无法解决问题，请：

1. **收集信息**:
   ```bash
   # 系统信息
   uname -a
   nvidia-smi
   python --version
   pip list | grep torch
   
   # 服务状态
   sudo systemctl status llm-server
   sudo journalctl -u llm-server --no-pager -l
   ```

2. **检查日志**:
   - 系统日志: `/var/log/syslog`
   - 应用日志: `/var/log/llm_deployment.log`
   - 服务日志: `sudo journalctl -u llm-server`

3. **社区支持**:
   - GitHub Issues
   - 技术论坛
   - 官方文档

---

**注意**: 本指南基于常见问题整理，具体问题可能需要根据实际情况调整解决方案。
