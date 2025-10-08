#!/usr/bin/env python3
"""
LLM服务器启动脚本
提供交互式界面选择启动方式
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_dependencies():
    """检查依赖是否安装"""
    try:
        import torch
        import transformers
        print("✅ PyTorch和Transformers已安装")
        return True
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        return False

def check_vllm():
    """检查VLLM是否安装"""
    try:
        import vllm
        print("✅ VLLM已安装")
        return True
    except ImportError:
        print("⚠️ VLLM未安装，将使用Transformers引擎")
        return False

def check_model():
    """检查模型是否下载"""
    model_dir = Path("/data/models")
    if not model_dir.exists():
        print("❌ 模型目录不存在，请先下载模型")
        return False
    
    # 检查是否有Qwen模型
    qwen_models = list(model_dir.glob("*Qwen*"))
    if not qwen_models:
        print("❌ 未找到Qwen模型，请先下载模型")
        print("运行: python model_download.py download --model qwen2.5-32b")
        return False
    
    print(f"✅ 找到模型: {[m.name for m in qwen_models]}")
    return True

def start_transformers_server():
    """启动Transformers服务器"""
    print("🚀 启动Transformers服务器...")
    try:
        subprocess.run([sys.executable, "model_server.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")

def start_vllm_server():
    """启动VLLM服务器"""
    print("🚀 启动VLLM服务器...")
    try:
        subprocess.run([sys.executable, "vllm_server.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")

def interactive_start():
    """交互式启动"""
    print("=" * 50)
    print("🤖 LLM大模型服务器启动器")
    print("=" * 50)
    
    # 检查环境
    if not check_dependencies():
        print("请先安装依赖: pip install -r requirements.txt")
        return
    
    if not check_model():
        print("请先下载模型")
        return
    
    vllm_available = check_vllm()
    
    print("\n请选择启动方式:")
    print("1. VLLM引擎 (高性能，推荐)")
    print("2. Transformers引擎 (标准)")
    print("3. 退出")
    
    if not vllm_available:
        print("\n⚠️ VLLM未安装，只能选择Transformers引擎")
    
    while True:
        try:
            choice = input("\n请输入选择 (1-3): ").strip()
            
            if choice == "1":
                if vllm_available:
                    start_vllm_server()
                    break
                else:
                    print("❌ VLLM未安装，请选择其他选项")
            elif choice == "2":
                start_transformers_server()
                break
            elif choice == "3":
                print("👋 再见!")
                break
            else:
                print("❌ 无效选择，请输入1-3")
        except KeyboardInterrupt:
            print("\n👋 再见!")
            break

def main():
    parser = argparse.ArgumentParser(description="LLM服务器启动器")
    parser.add_argument("--engine", choices=["transformers", "vllm"], 
                       help="指定启动引擎")
    parser.add_argument("--check", action="store_true", 
                       help="只检查环境，不启动服务器")
    
    args = parser.parse_args()
    
    if args.check:
        print("🔍 检查环境...")
        check_dependencies()
        check_vllm()
        check_model()
        return
    
    if args.engine:
        # 直接启动指定引擎
        if not check_dependencies() or not check_model():
            return
        
        if args.engine == "vllm":
            if check_vllm():
                start_vllm_server()
            else:
                print("❌ VLLM未安装")
        elif args.engine == "transformers":
            start_transformers_server()
    else:
        # 交互式启动
        interactive_start()

if __name__ == "__main__":
    main()
