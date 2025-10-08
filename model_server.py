"""
大模型API服务器
使用FastAPI构建高性能的LLM推理服务
"""

import asyncio
import logging
import time
import traceback
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import psutil
import GPUtil

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from config import ModelConfig

# 配置日志
logging.basicConfig(
    level=getattr(logging, ModelConfig.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(ModelConfig.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 全局变量
model = None
tokenizer = None

class ChatRequest(BaseModel):
    """聊天请求模型"""
    messages: List[Dict[str, str]] = Field(..., description="对话历史")
    max_tokens: Optional[int] = Field(None, description="最大生成token数")
    temperature: Optional[float] = Field(None, description="温度参数")
    top_p: Optional[float] = Field(None, description="top_p参数")
    stream: bool = Field(False, description="是否流式输出")

class ChatResponse(BaseModel):
    """聊天响应模型"""
    response: str = Field(..., description="模型回复")
    usage: Dict[str, int] = Field(..., description="token使用统计")
    model: str = Field(..., description="模型名称")
    processing_time: float = Field(..., description="处理时间(秒)")

class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="服务状态")
    model_loaded: bool = Field(..., description="模型是否加载")
    gpu_memory: Dict[str, float] = Field(..., description="GPU内存使用情况")
    system_memory: float = Field(..., description="系统内存使用率")

def load_model():
    """加载模型和tokenizer"""
    global model, tokenizer
    
    try:
        logger.info(f"开始加载模型: {ModelConfig.MODEL_NAME}")
        start_time = time.time()
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            ModelConfig.MODEL_NAME,
            trust_remote_code=ModelConfig.TRUST_REMOTE_CODE
        )
        
        # 配置量化
        if ModelConfig.USE_QUANTIZATION:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        else:
            bnb_config = None
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            ModelConfig.MODEL_NAME,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=ModelConfig.TRUST_REMOTE_CODE,
            low_cpu_mem_usage=True
        )
        
        # 设置生成参数
        model.generation_config.pad_token_id = tokenizer.eos_token_id
        
        load_time = time.time() - start_time
        logger.info(f"模型加载完成，耗时: {load_time:.2f}秒")
        
        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"模型参数总数: {total_params:,}")
        logger.info(f"可训练参数: {trainable_params:,}")
        
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise e

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时加载模型
    logger.info("正在启动LLM服务器...")
    load_model()
    logger.info("LLM服务器启动完成")
    
    yield
    
    # 关闭时清理资源
    logger.info("正在关闭LLM服务器...")
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    torch.cuda.empty_cache()
    logger.info("LLM服务器已关闭")

# 创建FastAPI应用
app = FastAPI(
    title="LLM大模型API服务",
    description="基于Qwen2.5-32B-Instruct的大模型推理服务",
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def format_chat_messages(messages: List[Dict[str, str]]) -> str:
    """格式化聊天消息为模型输入"""
    formatted_text = ""
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        if role == "user":
            formatted_text += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            formatted_text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    
    formatted_text += "<|im_start|>assistant\n"
    return formatted_text

def get_system_info():
    """获取系统信息"""
    # GPU信息
    gpu_memory = {}
    try:
        gpus = GPUtil.getGPUs()
        for i, gpu in enumerate(gpus):
            gpu_memory[f"gpu_{i}"] = {
                "used": gpu.memoryUsed,
                "total": gpu.memoryTotal,
                "utilization": gpu.memoryUtil * 100
            }
    except:
        gpu_memory = {"error": "无法获取GPU信息"}
    
    # 系统内存
    memory = psutil.virtual_memory()
    system_memory = memory.percent
    
    return gpu_memory, system_memory

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口"""
    gpu_memory, system_memory = get_system_info()
    
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        gpu_memory=gpu_memory,
        system_memory=system_memory
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """聊天接口"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        start_time = time.time()
        
        # 格式化输入
        input_text = format_chat_messages(request.messages)
        
        # 编码输入
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        # 获取生成参数
        generation_kwargs = ModelConfig.get_generation_kwargs()
        if request.max_tokens:
            generation_kwargs["max_new_tokens"] = request.max_tokens
        if request.temperature:
            generation_kwargs["temperature"] = request.temperature
        if request.top_p:
            generation_kwargs["top_p"] = request.top_p
        
        # 生成回复
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                **generation_kwargs
            )
        
        # 解码输出
        response_text = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # 计算token使用情况
        input_tokens = inputs.input_ids.shape[1]
        output_tokens = outputs.shape[1] - input_tokens
        total_tokens = input_tokens + output_tokens
        
        processing_time = time.time() - start_time
        
        logger.info(f"请求处理完成，耗时: {processing_time:.2f}秒，输入token: {input_tokens}，输出token: {output_tokens}")
        
        return ChatResponse(
            response=response_text,
            usage={
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": total_tokens
            },
            model=ModelConfig.MODEL_NAME,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"聊天请求处理失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"处理请求时发生错误: {str(e)}")

@app.get("/models")
async def list_models():
    """列出可用模型"""
    return {
        "models": [
            {
                "id": ModelConfig.MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local"
            }
        ]
    }

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "LLM大模型API服务",
        "version": "1.0.0",
        "model": ModelConfig.MODEL_NAME,
        "status": "running"
    }

if __name__ == "__main__":
    uvicorn.run(
        "model_server:app",
        host=ModelConfig.HOST,
        port=ModelConfig.PORT,
        workers=ModelConfig.WORKERS,
        log_level=ModelConfig.LOG_LEVEL.lower()
    )
