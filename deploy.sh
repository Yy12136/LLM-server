#!/bin/bash

# LLM大模型部署脚本
# 适用于A800服务器

set -e  # 遇到错误立即退出

echo "=========================================="
echo "LLM大模型部署脚本启动"
echo "=========================================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查系统要求
check_system_requirements() {
    log_info "检查系统要求..."
    
    # 检查Python版本
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 未安装"
        exit 1
    fi
    
    python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    if [[ $(echo "$python_version < 3.8" | bc -l) -eq 1 ]]; then
        log_error "Python版本过低，需要3.8+，当前版本: $python_version"
        exit 1
    fi
    log_info "Python版本检查通过: $python_version"
    
    # 检查CUDA
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "NVIDIA驱动未安装或nvidia-smi不可用"
        exit 1
    fi
    
    cuda_version=$(nvidia-smi | grep "CUDA Version" | cut -d' ' -f9)
    log_info "CUDA版本: $cuda_version"
    
    # 检查GPU内存
    gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    if [[ $gpu_memory -lt 24000 ]]; then
        log_warn "GPU内存可能不足，建议24GB+，当前: ${gpu_memory}MB"
    else
        log_info "GPU内存检查通过: ${gpu_memory}MB"
    fi
    
    # 检查系统内存
    total_memory=$(free -g | awk '/^Mem:/{print $2}')
    if [[ $total_memory -lt 32 ]]; then
        log_warn "系统内存可能不足，建议32GB+，当前: ${total_memory}GB"
    else
        log_info "系统内存检查通过: ${total_memory}GB"
    fi
}

# 创建虚拟环境
setup_virtual_environment() {
    log_info "设置Python虚拟环境..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        log_info "虚拟环境创建完成"
    else
        log_info "虚拟环境已存在"
    fi
    
    source venv/bin/activate
    log_info "虚拟环境已激活"
    
    # 升级pip
    pip install --upgrade pip
    log_info "pip已升级"
}

# 安装依赖
install_dependencies() {
    log_info "安装Python依赖..."
    
    # 安装PyTorch (CUDA版本)
    log_info "安装PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    # 安装其他依赖
    pip install -r requirements.txt
    log_info "依赖安装完成"
}

# 创建目录结构
create_directories() {
    log_info "创建目录结构..."
    
    mkdir -p /data/models
    mkdir -p /var/log
    mkdir -p logs
    
    # 设置权限
    sudo chown -R $USER:$USER /data/models 2>/dev/null || true
    sudo chown -R $USER:$USER /var/log 2>/dev/null || true
    
    log_info "目录结构创建完成"
}

# 下载模型（可选）
download_model() {
    log_info "准备模型下载..."
    
    read -p "是否现在下载Qwen2.5-32B-Instruct模型？(y/N): " download_choice
    
    if [[ $download_choice =~ ^[Yy]$ ]]; then
        log_info "开始下载模型..."
        
        # 使用huggingface-cli下载
        pip install huggingface_hub
        huggingface-cli download Qwen/Qwen2.5-32B-Instruct --local-dir /data/models/Qwen2.5-32B-Instruct
        
        log_info "模型下载完成"
    else
        log_warn "跳过模型下载，请确保模型文件在 /data/models/Qwen2.5-32B-Instruct 目录下"
    fi
}

# 创建systemd服务文件
create_systemd_service() {
    log_info "创建systemd服务文件..."
    
    current_dir=$(pwd)
    
    cat > llm-server.service << EOF
[Unit]
Description=LLM Model Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$current_dir
Environment=PATH=$current_dir/venv/bin
ExecStart=$current_dir/venv/bin/python model_server.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
    
    sudo mv llm-server.service /etc/systemd/system/
    sudo systemctl daemon-reload
    
    log_info "systemd服务文件创建完成"
}

# 配置防火墙
configure_firewall() {
    log_info "配置防火墙..."
    
    # 检查防火墙状态
    if command -v ufw &> /dev/null; then
        sudo ufw allow 8000/tcp
        log_info "UFW防火墙规则已添加"
    elif command -v firewall-cmd &> /dev/null; then
        sudo firewall-cmd --permanent --add-port=8000/tcp
        sudo firewall-cmd --reload
        log_info "FirewallD防火墙规则已添加"
    else
        log_warn "未检测到防火墙管理工具"
    fi
}

# 测试服务
test_service() {
    log_info "测试服务..."
    
    # 启动服务
    python model_server.py &
    server_pid=$!
    
    # 等待服务启动
    sleep 10
    
    # 测试健康检查接口
    if curl -s http://localhost:8000/health > /dev/null; then
        log_info "服务启动成功！"
        echo "健康检查: http://localhost:8000/health"
        echo "API文档: http://localhost:8000/docs"
    else
        log_error "服务启动失败"
    fi
    
    # 停止测试服务
    kill $server_pid 2>/dev/null || true
}

# 主函数
main() {
    echo "开始部署LLM大模型服务..."
    
    check_system_requirements
    setup_virtual_environment
    install_dependencies
    create_directories
    download_model
    create_systemd_service
    configure_firewall
    test_service
    
    echo "=========================================="
    log_info "部署完成！"
    echo "=========================================="
    echo ""
    echo "使用方法："
    echo "1. 启动服务: sudo systemctl start llm-server"
    echo "2. 停止服务: sudo systemctl stop llm-server"
    echo "3. 查看状态: sudo systemctl status llm-server"
    echo "4. 查看日志: sudo journalctl -u llm-server -f"
    echo "5. 测试API: curl http://localhost:8000/health"
    echo ""
    echo "API文档: http://your-server-ip:8000/docs"
    echo "健康检查: http://your-server-ip:8000/health"
    echo ""
    echo "注意事项："
    echo "- 首次启动可能需要较长时间加载模型"
    echo "- 建议在screen或tmux中运行以保持服务稳定"
    echo "- 定期检查日志和系统资源使用情况"
}

# 运行主函数
main "$@"
