"""
大模型部署配置文件
"""
import os
from typing import Dict, Any

class ModelConfig:
    """模型配置类"""
    
    # 模型基本信息 - 使用最新的Qwen2.5系列
    MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"  # 目前最新的32B模型
    MODEL_PATH = "/data/models/Qwen2.5-32B-Instruct"  # 本地模型路径
    
    # 备选模型选项 (可根据需要切换)
    # MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"  # 14B版本，显存需求更低
    # MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"  # 72B版本，性能更强但需要更多资源
    TRUST_REMOTE_CODE = True
    
    # 推理配置
    MAX_TOKENS = 2048
    TEMPERATURE = 0.7
    TOP_P = 0.9
    REPETITION_PENALTY = 1.1
    
    # 量化配置
    USE_QUANTIZATION = True
    QUANTIZATION_BITS = 4  # 4-bit量化
    
    # 内存和GPU配置
    MAX_MEMORY_USAGE = 0.9  # 最大内存使用率
    GPU_MEMORY_FRACTION = 0.95  # GPU内存使用比例
    
    # 批处理配置
    BATCH_SIZE = 1
    MAX_BATCH_SIZE = 4
    
    # 服务器配置
    HOST = "0.0.0.0"
    PORT = 8000
    WORKERS = 1
    
    # 日志配置
    LOG_LEVEL = "INFO"
    LOG_FILE = "/var/log/llm_deployment.log"
    
    @classmethod
    def get_model_kwargs(cls) -> Dict[str, Any]:
        """获取模型加载参数"""
        kwargs = {
            "torch_dtype": "auto",
            "device_map": "auto",
            "trust_remote_code": cls.TRUST_REMOTE_CODE,
        }
        
        if cls.USE_QUANTIZATION:
            kwargs.update({
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "float16",
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4"
            })
        
        return kwargs
    
    @classmethod
    def get_generation_kwargs(cls) -> Dict[str, Any]:
        """获取文本生成参数"""
        return {
            "max_new_tokens": cls.MAX_TOKENS,
            "temperature": cls.TEMPERATURE,
            "top_p": cls.TOP_P,
            "repetition_penalty": cls.REPETITION_PENALTY,
            "do_sample": True,
            "pad_token_id": 151643,  # Qwen模型的pad_token_id
            "eos_token_id": 151643,  # Qwen模型的eos_token_id
        }

# 环境变量配置
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第一块GPU
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 避免tokenizer并行警告
