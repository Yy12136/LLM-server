"""
基于VLLM的高性能LLM API服务器
使用VLLM引擎提供更快的推理速度和更高的并发性能
"""

import asyncio
import logging
import time
import traceback
from typing import List, Dict, Any, Optional, AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import psutil
import GPUtil

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

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
engine = None

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

class StreamResponse(BaseModel):
    """流式响应模型"""
    id: str = Field(..., description="请求ID")
    object: str = Field("chat.completion.chunk", description="对象类型")
    created: int = Field(..., description="创建时间戳")
    model: str = Field(..., description="模型名称")
    choices: List[Dict[str, Any]] = Field(..., description="选择列表")

class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="服务状态")
    model_loaded: bool = Field(..., description="模型是否加载")
    gpu_memory: Dict[str, float] = Field(..., description="GPU内存使用情况")
    system_memory: float = Field(..., description="系统内存使用率")
    engine_info: Dict[str, Any] = Field(..., description="VLLM引擎信息")

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

def create_engine():
    """创建VLLM引擎"""
    global engine
    
    try:
        logger.info(f"正在初始化VLLM引擎，模型: {ModelConfig.MODEL_NAME}")
        start_time = time.time()
        
        # VLLM引擎参数
        engine_args = AsyncEngineArgs(
            model=ModelConfig.MODEL_NAME,
            trust_remote_code=ModelConfig.TRUST_REMOTE_CODE,
            tensor_parallel_size=1,  # 单GPU
            gpu_memory_utilization=ModelConfig.GPU_MEMORY_FRACTION,
            max_model_len=ModelConfig.MAX_TOKENS * 2,  # 设置最大序列长度
            quantization="fp16",  # 使用FP16量化
            enforce_eager=True,  # 启用eager模式提高稳定性
            disable_log_stats=True,  # 禁用日志统计
            max_num_seqs=ModelConfig.MAX_BATCH_SIZE,  # 最大批处理大小
        )
        
        # 创建异步引擎
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        load_time = time.time() - start_time
        logger.info(f"VLLM引擎初始化完成，耗时: {load_time:.2f}秒")
        
    except Exception as e:
        logger.error(f"VLLM引擎初始化失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise e

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化引擎
    logger.info("正在启动VLLM服务器...")
    create_engine()
    logger.info("VLLM服务器启动完成")
    
    yield
    
    # 关闭时清理资源
    logger.info("正在关闭VLLM服务器...")
    if engine is not None:
        await engine.shutdown()
    logger.info("VLLM服务器已关闭")

# 创建FastAPI应用
app = FastAPI(
    title="VLLM大模型API服务",
    description="基于VLLM引擎的高性能LLM推理服务",
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

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口"""
    gpu_memory, system_memory = get_system_info()
    
    # 获取引擎信息
    engine_info = {}
    if engine is not None:
        engine_info = {
            "model_name": ModelConfig.MODEL_NAME,
            "max_model_len": engine.engine.model_config.max_model_len,
            "max_num_seqs": engine.engine.scheduler_config.max_num_seqs,
        }
    
    return HealthResponse(
        status="healthy" if engine is not None else "unhealthy",
        model_loaded=engine is not None,
        gpu_memory=gpu_memory,
        system_memory=system_memory,
        engine_info=engine_info
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """聊天接口"""
    if engine is None:
        raise HTTPException(status_code=503, detail="VLLM引擎未初始化")
    
    try:
        start_time = time.time()
        
        # 格式化输入
        input_text = format_chat_messages(request.messages)
        
        # 创建采样参数
        sampling_params = SamplingParams(
            max_tokens=request.max_tokens or ModelConfig.MAX_TOKENS,
            temperature=request.temperature or ModelConfig.TEMPERATURE,
            top_p=request.top_p or ModelConfig.TOP_P,
            repetition_penalty=ModelConfig.REPETITION_PENALTY,
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        
        # 生成请求ID
        request_id = random_uuid()
        
        # 提交生成请求
        results_generator = engine.generate(
            input_text,
            sampling_params,
            request_id=request_id
        )
        
        # 等待结果
        final_output = None
        async for request_output in results_generator:
            if request_output.finished:
                final_output = request_output
                break
        
        if final_output is None:
            raise HTTPException(status_code=500, detail="生成失败")
        
        # 提取输出文本
        response_text = final_output.outputs[0].text
        
        # 计算token使用情况
        input_tokens = len(final_output.prompt_token_ids)
        output_tokens = len(final_output.outputs[0].token_ids)
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

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式聊天接口"""
    if engine is None:
        raise HTTPException(status_code=503, detail="VLLM引擎未初始化")
    
    try:
        # 格式化输入
        input_text = format_chat_messages(request.messages)
        
        # 创建采样参数
        sampling_params = SamplingParams(
            max_tokens=request.max_tokens or ModelConfig.MAX_TOKENS,
            temperature=request.temperature or ModelConfig.TEMPERATURE,
            top_p=request.top_p or ModelConfig.TOP_P,
            repetition_penalty=ModelConfig.REPETITION_PENALTY,
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        
        # 生成请求ID
        request_id = random_uuid()
        
        # 提交生成请求
        results_generator = engine.generate(
            input_text,
            sampling_params,
            request_id=request_id
        )
        
        async def generate_stream():
            """生成流式响应"""
            async for request_output in results_generator:
                if not request_output.finished:
                    # 构建流式响应
                    chunk = StreamResponse(
                        id=request_id,
                        created=int(time.time()),
                        model=ModelConfig.MODEL_NAME,
                        choices=[{
                            "index": 0,
                            "delta": {
                                "content": request_output.outputs[0].text
                            },
                            "finish_reason": None
                        }]
                    )
                    yield f"data: {chunk.json()}\n\n"
                else:
                    # 发送结束标记
                    final_chunk = StreamResponse(
                        id=request_id,
                        created=int(time.time()),
                        model=ModelConfig.MODEL_NAME,
                        choices=[{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }]
                    )
                    yield f"data: {final_chunk.json()}\n\n"
                    yield "data: [DONE]\n\n"
                    break
        
        from fastapi.responses import StreamingResponse
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
        
    except Exception as e:
        logger.error(f"流式聊天请求处理失败: {str(e)}")
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
                "owned_by": "local",
                "engine": "vllm"
            }
        ]
    }

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "VLLM大模型API服务",
        "version": "1.0.0",
        "model": ModelConfig.MODEL_NAME,
        "engine": "vllm",
        "status": "running"
    }

if __name__ == "__main__":
    uvicorn.run(
        "vllm_server:app",
        host=ModelConfig.HOST,
        port=ModelConfig.PORT,
        workers=1,  # VLLM不支持多worker
        log_level=ModelConfig.LOG_LEVEL.lower()
    )
