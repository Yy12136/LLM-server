#!/bin/bash

# 远程服务器LLM大模型部署脚本
# 适用于Ubuntu/CentOS系统

set -e  # 遇到错误立即退出

echo "=========================================="
echo "远程服务器LLM大模型部署脚本"
echo "=========================================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# 检查是否为root用户
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log_error "请不要使用root用户运行此脚本"
        log_info "请使用普通用户运行，脚本会在需要时请求sudo权限"
        exit 1
    fi
}

# 检测操作系统
detect_os() {
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        OS=$NAME
        VER=$VERSION_ID
    else
        log_error "无法检测操作系统"
        exit 1
    fi
    
    log_info "检测到操作系统: $OS $VER"
}

# 安装系统依赖
install_system_deps() {
    log_step "安装系统依赖..."
    
    if [[ "$OS" == *"Ubuntu"* ]] || [[ "$OS" == *"Debian"* ]]; then
        sudo apt update
        sudo apt install -y python3 python3-pip python3-venv git curl wget htop nvidia-utils-535
    elif [[ "$OS" == *"CentOS"* ]] || [[ "$OS" == *"Red Hat"* ]]; then
        sudo yum update -y
        sudo yum install -y python3 python3-pip git curl wget htop
    else
        log_warn "未识别的操作系统，请手动安装Python3和相关依赖"
    fi
    
    log_info "系统依赖安装完成"
}

# 检查硬件要求
check_hardware() {
    log_step "检查硬件要求..."
    
    # 检查GPU
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "未检测到NVIDIA GPU或驱动未安装"
        log_info "请确保已安装NVIDIA驱动: https://developer.nvidia.com/cuda-downloads"
        exit 1
    fi
    
    # 检查GPU内存
    gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    if [[ $gpu_memory -lt 20000 ]]; then
        log_warn "GPU内存可能不足，建议20GB+，当前: ${gpu_memory}MB"
        log_info "建议使用较小的模型，如Qwen2.5-7B或Qwen2.5-14B"
    else
        log_info "GPU内存检查通过: ${gpu_memory}MB"
    fi
    
    # 检查系统内存
    total_memory=$(free -g | awk '/^Mem:/{print $2}')
    if [[ $total_memory -lt 16 ]]; then
        log_warn "系统内存可能不足，建议16GB+，当前: ${total_memory}GB"
    else
        log_info "系统内存检查通过: ${total_memory}GB"
    fi
    
    # 检查磁盘空间
    available_space=$(df -BG / | awk 'NR==2{print $4}' | sed 's/G//')
    if [[ $available_space -lt 100 ]]; then
        log_warn "磁盘空间可能不足，建议100GB+，当前可用: ${available_space}GB"
    else
        log_info "磁盘空间检查通过: ${available_space}GB"
    fi
}

# 设置Python环境
setup_python_env() {
    log_step "设置Python环境..."
    
    # 创建虚拟环境
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        log_info "虚拟环境创建完成"
    else
        log_info "虚拟环境已存在"
    fi
    
    # 激活虚拟环境
    source venv/bin/activate
    log_info "虚拟环境已激活"
    
    # 升级pip
    pip install --upgrade pip
    log_info "pip已升级"
}

# 安装Python依赖
install_python_deps() {
    log_step "安装Python依赖..."
    
    # 安装PyTorch (CUDA版本)
    log_info "安装PyTorch (CUDA版本)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    # 安装其他依赖
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
        log_warn "requirements.txt文件不存在，安装基础依赖..."
        pip install transformers accelerate bitsandbytes fastapi uvicorn huggingface_hub
    fi
    
    log_info "Python依赖安装完成"
}

# 创建目录结构
create_directories() {
    log_step "创建目录结构..."
    
    # 创建模型目录
    sudo mkdir -p /data/models
    sudo chown -R $USER:$USER /data/models
    
    # 创建日志目录
    sudo mkdir -p /var/log
    sudo chown -R $USER:$USER /var/log
    
    # 创建本地日志目录
    mkdir -p logs
    
    log_info "目录结构创建完成"
}

# HuggingFace认证
setup_huggingface_auth() {
    log_step "设置HuggingFace认证..."
    
    echo "请选择HuggingFace认证方式:"
    echo "1. 使用huggingface-cli登录"
    echo "2. 设置环境变量"
    echo "3. 跳过（稍后手动设置）"
    
    read -p "请选择 (1-3): " auth_choice
    
    case $auth_choice in
        1)
            log_info "请按照提示输入你的HuggingFace Token:"
            huggingface-cli login
            ;;
        2)
            read -p "请输入你的HuggingFace Token: " hf_token
            echo "export HUGGINGFACE_HUB_TOKEN=\"$hf_token\"" >> ~/.bashrc
            export HUGGINGFACE_HUB_TOKEN="$hf_token"
            log_info "Token已设置到环境变量"
            ;;
        3)
            log_warn "跳过HuggingFace认证设置"
            log_info "请稍后手动运行: huggingface-cli login"
            ;;
        *)
            log_warn "无效选择，跳过认证设置"
            ;;
    esac
}

# 下载模型
download_model() {
    log_step "准备模型下载..."
    
    echo "请选择要下载的模型:"
    echo "1. Qwen2.5-7B-Instruct (约14GB，适合资源受限环境)"
    echo "2. Qwen2.5-14B-Instruct (约28GB，平衡选择)"
    echo "3. Qwen2.5-32B-Instruct (约64GB，推荐A800)"
    echo "4. 跳过下载"
    
    read -p "请选择 (1-4): " model_choice
    
    case $model_choice in
        1)
            model_name="qwen2.5-7b"
            ;;
        2)
            model_name="qwen2.5-14b"
            ;;
        3)
            model_name="qwen2.5-32b"
            ;;
        4)
            log_info "跳过模型下载"
            return
            ;;
        *)
            log_error "无效选择"
            return
            ;;
    esac
    
    log_info "开始下载模型: $model_name"
    
    if [ -f "model_download.py" ]; then
        python model_download.py download --model $model_name
    else
        log_warn "model_download.py不存在，使用huggingface-cli下载"
        model_id="Qwen/${model_name^}-Instruct"
        huggingface-cli download $model_id --local-dir /data/models/${model_name^}-Instruct
    fi
    
    log_info "模型下载完成"
}

# 配置防火墙
configure_firewall() {
    log_step "配置防火墙..."
    
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

# 创建systemd服务
create_systemd_service() {
    log_step "创建systemd服务..."
    
    current_dir=$(pwd)
    current_user=$(whoami)
    
    cat > llm-server.service << EOF
[Unit]
Description=LLM Model Server
After=network.target

[Service]
Type=simple
User=$current_user
WorkingDirectory=$current_dir
Environment=PATH=$current_dir/venv/bin
Environment=PYTHONPATH=$current_dir
ExecStart=$current_dir/venv/bin/python vllm_server.py
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

# 测试服务
test_service() {
    log_step "测试服务..."
    
    # 检查服务文件是否存在
    if [ ! -f "vllm_server.py" ]; then
        log_warn "vllm_server.py不存在，跳过服务测试"
        return
    fi
    
    # 启动服务进行测试
    log_info "启动服务进行测试..."
    python vllm_server.py &
    server_pid=$!
    
    # 等待服务启动
    sleep 15
    
    # 测试健康检查接口
    if curl -s http://localhost:8000/health > /dev/null; then
        log_info "✅ 服务启动成功！"
        echo "健康检查: http://localhost:8000/health"
        echo "API文档: http://localhost:8000/docs"
    else
        log_warn "⚠️ 服务可能启动失败，请检查日志"
    fi
    
    # 停止测试服务
    kill $server_pid 2>/dev/null || true
    sleep 2
}

# 显示使用说明
show_usage() {
    echo "=========================================="
    log_info "部署完成！"
    echo "=========================================="
    echo ""
    echo "🚀 服务管理命令:"
    echo "  启动服务: sudo systemctl start llm-server"
    echo "  停止服务: sudo systemctl stop llm-server"
    echo "  重启服务: sudo systemctl restart llm-server"
    echo "  查看状态: sudo systemctl status llm-server"
    echo "  查看日志: sudo journalctl -u llm-server -f"
    echo "  开机自启: sudo systemctl enable llm-server"
    echo ""
    echo "🌐 API访问地址:"
    echo "  健康检查: http://$(hostname -I | awk '{print $1}'):8000/health"
    echo "  API文档: http://$(hostname -I | awk '{print $1}'):8000/docs"
    echo "  聊天接口: http://$(hostname -I | awk '{print $1}'):8000/chat"
    echo ""
    echo "📝 测试命令:"
    echo "  curl http://$(hostname -I | awk '{print $1}'):8000/health"
    echo "  python test_client.py"
    echo ""
    echo "⚠️  注意事项:"
    echo "  - 首次启动可能需要5-10分钟加载模型"
    echo "  - 建议在screen或tmux中运行以保持服务稳定"
    echo "  - 定期检查日志和系统资源使用情况"
    echo "  - 如需修改配置，请编辑config.py后重启服务"
    echo ""
}

# 主函数
main() {
    log_info "开始远程服务器LLM大模型部署..."
    
    check_root
    detect_os
    check_hardware
    install_system_deps
    setup_python_env
    install_python_deps
    create_directories
    setup_huggingface_auth
    download_model
    configure_firewall
    create_systemd_service
    test_service
    show_usage
}

# 运行主函数
main "$@"
