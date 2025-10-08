"""
模型下载和管理脚本
支持从Hugging Face下载Qwen系列模型
"""

import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download
import torch

class ModelDownloader:
    """模型下载器"""
    
    # 支持的模型列表
    SUPPORTED_MODELS = {
        "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
        "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct", 
        "qwen2.5-32b": "Qwen/Qwen2.5-32B-Instruct",
        "qwen2.5-72b": "Qwen/Qwen2.5-72B-Instruct",
        "qwen2-7b": "Qwen/Qwen2-7B-Instruct",
        "qwen2-14b": "Qwen/Qwen2-14B-Instruct",
        "qwen2-32b": "Qwen/Qwen2-32B-Instruct",
        "qwen2-72b": "Qwen/Qwen2-72B-Instruct",
    }
    
    def __init__(self, base_dir: str = "/data/models"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def get_model_info(self, model_key: str) -> dict:
        """获取模型信息"""
        if model_key not in self.SUPPORTED_MODELS:
            raise ValueError(f"不支持的模型: {model_key}")
        
        model_name = self.SUPPORTED_MODELS[model_key]
        local_path = self.base_dir / model_name.split('/')[-1]
        
        # 估算模型大小
        model_sizes = {
            "7b": "~14GB",
            "14b": "~28GB", 
            "32b": "~64GB",
            "72b": "~144GB"
        }
        
        size_key = model_key.split('-')[-1]
        estimated_size = model_sizes.get(size_key, "未知")
        
        return {
            "model_name": model_name,
            "local_path": str(local_path),
            "estimated_size": estimated_size,
            "model_key": model_key
        }
    
    def check_disk_space(self, required_gb: int) -> bool:
        """检查磁盘空间"""
        try:
            statvfs = os.statvfs(self.base_dir)
            free_space_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
            return free_space_gb > required_gb
        except:
            return True  # 如果无法检查，假设空间足够
    
    def download_model(self, model_key: str, use_auth_token: str = None) -> str:
        """下载模型"""
        model_info = self.get_model_info(model_key)
        model_name = model_info["model_name"]
        local_path = model_info["local_path"]
        
        print(f"开始下载模型: {model_name}")
        print(f"本地路径: {local_path}")
        print(f"预估大小: {model_info['estimated_size']}")
        print("-" * 50)
        
        # 检查是否已存在
        if os.path.exists(local_path) and os.listdir(local_path):
            print(f"模型已存在于: {local_path}")
            choice = input("是否重新下载？(y/N): ").lower()
            if choice != 'y':
                return local_path
        
        try:
            # 下载模型
            downloaded_path = snapshot_download(
                repo_id=model_name,
                local_dir=local_path,
                token=use_auth_token,
                resume_download=True,  # 支持断点续传
                local_files_only=False
            )
            
            print(f"✅ 模型下载完成: {downloaded_path}")
            return downloaded_path
            
        except Exception as e:
            print(f"❌ 模型下载失败: {e}")
            raise e
    
    def verify_model(self, model_key: str) -> bool:
        """验证模型完整性"""
        model_info = self.get_model_info(model_key)
        local_path = model_info["local_path"]
        
        print(f"验证模型: {local_path}")
        
        if not os.path.exists(local_path):
            print("❌ 模型目录不存在")
            return False
        
        # 检查必要文件
        required_files = [
            "config.json",
            "tokenizer.json", 
            "tokenizer_config.json"
        ]
        
        for file in required_files:
            file_path = os.path.join(local_path, file)
            if not os.path.exists(file_path):
                print(f"❌ 缺少必要文件: {file}")
                return False
        
        # 检查模型文件
        model_files = [f for f in os.listdir(local_path) if f.endswith(('.bin', '.safetensors'))]
        if not model_files:
            print("❌ 未找到模型权重文件")
            return False
        
        print(f"✅ 模型验证通过，包含 {len(model_files)} 个模型文件")
        return True
    
    def list_models(self):
        """列出所有支持的模型"""
        print("支持的模型列表:")
        print("-" * 80)
        print(f"{'模型名称':<20} {'HuggingFace ID':<30} {'预估大小':<15}")
        print("-" * 80)
        
        for key, model_id in self.SUPPORTED_MODELS.items():
            size_key = key.split('-')[-1]
            sizes = {
                "7b": "~14GB",
                "14b": "~28GB",
                "32b": "~64GB", 
                "72b": "~144GB"
            }
            size = sizes.get(size_key, "未知")
            print(f"{key:<20} {model_id:<30} {size:<15}")
    
    def list_downloaded_models(self):
        """列出已下载的模型"""
        print("已下载的模型:")
        print("-" * 50)
        
        if not os.path.exists(self.base_dir):
            print("模型目录不存在")
            return
        
        for item in os.listdir(self.base_dir):
            item_path = os.path.join(self.base_dir, item)
            if os.path.isdir(item_path):
                # 检查是否包含模型文件
                has_model = any(f.endswith(('.bin', '.safetensors')) for f in os.listdir(item_path))
                status = "✅ 完整" if has_model else "⚠️ 不完整"
                print(f"{item:<30} {status}")

def main():
    parser = argparse.ArgumentParser(description="Qwen模型下载和管理工具")
    parser.add_argument("command", choices=["download", "verify", "list", "list-downloaded"], 
                       help="操作命令")
    parser.add_argument("--model", "-m", help="模型名称 (如: qwen2.5-32b)")
    parser.add_argument("--base-dir", default="/data/models", help="模型存储目录")
    parser.add_argument("--token", help="HuggingFace访问令牌 (可选)")
    
    args = parser.parse_args()
    
    downloader = ModelDownloader(args.base_dir)
    
    if args.command == "list":
        downloader.list_models()
        
    elif args.command == "list-downloaded":
        downloader.list_downloaded_models()
        
    elif args.command == "download":
        if not args.model:
            print("请指定要下载的模型名称")
            print("使用 'python model_download.py list' 查看支持的模型")
            return
        
        try:
            model_info = downloader.get_model_info(args.model)
            print(f"准备下载: {model_info['model_name']}")
            print(f"存储路径: {model_info['local_path']}")
            print(f"预估大小: {model_info['estimated_size']}")
            
            confirm = input("确认下载？(y/N): ").lower()
            if confirm == 'y':
                downloader.download_model(args.model, args.token)
            else:
                print("下载已取消")
                
        except ValueError as e:
            print(f"错误: {e}")
            
    elif args.command == "verify":
        if not args.model:
            print("请指定要验证的模型名称")
            return
            
        try:
            if downloader.verify_model(args.model):
                print("✅ 模型验证通过")
            else:
                print("❌ 模型验证失败")
        except Exception as e:
            print(f"验证过程中发生错误: {e}")

if __name__ == "__main__":
    main()
