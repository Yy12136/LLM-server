"""
LLM API客户端示例
演示如何调用远程部署的大模型API
"""

import requests
import json
import time
from typing import List, Dict, Optional, Generator
import asyncio
import aiohttp

class LLMAPIClient:
    """LLM API客户端类"""
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 300):
        """
        初始化客户端
        
        Args:
            base_url: API服务地址
            timeout: 请求超时时间（秒）
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        
        # 设置请求头
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'LLM-API-Client/1.0'
        })
    
    def health_check(self) -> Dict:
        """健康检查"""
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e), "status": "unhealthy"}
    
    def list_models(self) -> Dict:
        """列出可用模型"""
        try:
            response = self.session.get(
                f"{self.base_url}/models",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def chat(self, 
             messages: List[Dict[str, str]], 
             max_tokens: Optional[int] = None,
             temperature: Optional[float] = None,
             top_p: Optional[float] = None,
             stream: bool = False) -> Dict:
        """
        发送聊天请求
        
        Args:
            messages: 对话历史列表
            max_tokens: 最大生成token数
            temperature: 温度参数
            top_p: top_p参数
            stream: 是否流式输出
            
        Returns:
            响应结果
        """
        data = {
            "messages": messages,
            "stream": stream
        }
        
        # 添加可选参数
        if max_tokens is not None:
            data["max_tokens"] = max_tokens
        if temperature is not None:
            data["temperature"] = temperature
        if top_p is not None:
            data["top_p"] = top_p
        
        try:
            if stream:
                return self._stream_chat(data)
            else:
                response = self.session.post(
                    f"{self.base_url}/chat",
                    json=data,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def _stream_chat(self, data: Dict) -> Generator[str, None, None]:
        """流式聊天"""
        try:
            response = self.session.post(
                f"{self.base_url}/chat/stream",
                json=data,
                stream=True,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data_str = line[6:]  # 移除 'data: ' 前缀
                        if data_str != '[DONE]':
                            try:
                                chunk = json.loads(data_str)
                                if 'choices' in chunk and chunk['choices']:
                                    content = chunk['choices'][0]['delta'].get('content', '')
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                pass
        except Exception as e:
            yield f"Error: {str(e)}"

class AsyncLLMAPIClient:
    """异步LLM API客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 300):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
    
    async def health_check(self) -> Dict:
        """异步健康检查"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/health", timeout=10) as response:
                    return await response.json()
        except Exception as e:
            return {"error": str(e), "status": "unhealthy"}
    
    async def chat(self, 
                   messages: List[Dict[str, str]], 
                   max_tokens: Optional[int] = None,
                   temperature: Optional[float] = None,
                   top_p: Optional[float] = None) -> Dict:
        """异步聊天请求"""
        data = {
            "messages": messages
        }
        
        if max_tokens is not None:
            data["max_tokens"] = max_tokens
        if temperature is not None:
            data["temperature"] = temperature
        if top_p is not None:
            data["top_p"] = top_p
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat",
                    json=data,
                    timeout=self.timeout
                ) as response:
                    return await response.json()
        except Exception as e:
            return {"error": str(e)}

def demo_basic_usage():
    """基础使用示例"""
    print("=" * 60)
    print("LLM API客户端基础使用示例")
    print("=" * 60)
    
    # 初始化客户端
    client = LLMAPIClient("http://your-server-ip:8000")
    
    # 1. 健康检查
    print("\n1. 健康检查...")
    health = client.health_check()
    if "error" in health:
        print(f"❌ 健康检查失败: {health['error']}")
        return
    else:
        print(f"✅ 服务状态: {health['status']}")
        print(f"✅ 模型加载: {health['model_loaded']}")
        print(f"✅ GPU内存: {health['gpu_memory']}")
    
    # 2. 列出模型
    print("\n2. 列出可用模型...")
    models = client.list_models()
    if "error" in models:
        print(f"❌ 获取模型列表失败: {models['error']}")
    else:
        for model in models['models']:
            print(f"✅ 可用模型: {model['id']}")
    
    # 3. 简单对话
    print("\n3. 简单对话测试...")
    messages = [
        {"role": "user", "content": "你好，请简单介绍一下你自己。"}
    ]
    
    start_time = time.time()
    response = client.chat(messages, max_tokens=200, temperature=0.7)
    end_time = time.time()
    
    if "error" in response:
        print(f"❌ 对话失败: {response['error']}")
    else:
        print(f"✅ 模型回复: {response['response']}")
        print(f"✅ 处理时间: {response['processing_time']:.2f}秒")
        print(f"✅ Token使用: {response['usage']}")
        print(f"✅ 总耗时: {end_time - start_time:.2f}秒")

def demo_stream_chat():
    """流式聊天示例"""
    print("\n" + "=" * 60)
    print("流式聊天示例")
    print("=" * 60)
    
    client = LLMAPIClient("http://your-server-ip:8000")
    
    messages = [
        {"role": "user", "content": "请写一首关于春天的诗，要求押韵。"}
    ]
    
    print("AI回复: ", end="", flush=True)
    
    for chunk in client.chat(messages, max_tokens=300, stream=True):
        print(chunk, end="", flush=True)
    
    print("\n")

def demo_multi_turn_chat():
    """多轮对话示例"""
    print("\n" + "=" * 60)
    print("多轮对话示例")
    print("=" * 60)
    
    client = LLMAPIClient("http://your-server-ip:8000")
    
    # 模拟多轮对话
    conversation = [
        {"role": "user", "content": "什么是人工智能？"},
        {"role": "assistant", "content": "人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。"},
        {"role": "user", "content": "请详细解释一下机器学习和深度学习的区别。"}
    ]
    
    response = client.chat(conversation, max_tokens=400, temperature=0.7)
    
    if "error" in response:
        print(f"❌ 多轮对话失败: {response['error']}")
    else:
        print(f"✅ 模型回复: {response['response']}")
        print(f"✅ 处理时间: {response['processing_time']:.2f}秒")

def demo_async_usage():
    """异步使用示例"""
    print("\n" + "=" * 60)
    print("异步API调用示例")
    print("=" * 60)
    
    async def async_demo():
        client = AsyncLLMAPIClient("http://your-server-ip:8000")
        
        # 并发发送多个请求
        tasks = []
        for i in range(3):
            messages = [{"role": "user", "content": f"请用一句话描述第{i+1}个问题的重要性。"}]
            task = client.chat(messages, max_tokens=100)
            tasks.append(task)
        
        # 等待所有请求完成
        responses = await asyncio.gather(*tasks)
        
        for i, response in enumerate(responses):
            if "error" in response:
                print(f"❌ 请求{i+1}失败: {response['error']}")
            else:
                print(f"✅ 请求{i+1}回复: {response['response']}")
    
    # 运行异步示例
    asyncio.run(async_demo())

def interactive_chat():
    """交互式聊天"""
    print("\n" + "=" * 60)
    print("交互式聊天 (输入 'quit' 退出)")
    print("=" * 60)
    
    client = LLMAPIClient("http://your-server-ip:8000")
    
    # 检查服务状态
    health = client.health_check()
    if "error" in health:
        print(f"❌ 服务不可用: {health['error']}")
        return
    
    messages = []
    
    while True:
        user_input = input("\n用户: ").strip()
        
        if user_input.lower() in ['quit', 'exit', '退出', 'q']:
            print("再见！")
            break
        
        if not user_input:
            continue
        
        messages.append({"role": "user", "content": user_input})
        
        print("AI: ", end="", flush=True)
        start_time = time.time()
        
        response = client.chat(messages, max_tokens=500, temperature=0.7)
        
        if "error" in response:
            print(f"❌ 错误: {response['error']}")
        else:
            print(response['response'])
            messages.append({"role": "assistant", "content": response['response']})
            
            # 显示统计信息
            end_time = time.time()
            print(f"\n[处理时间: {response['processing_time']:.2f}秒, "
                  f"总耗时: {end_time - start_time:.2f}秒, "
                  f"Token使用: {response['usage']['total_tokens']}]")
            
            # 保持对话历史在合理范围内
            if len(messages) > 10:
                messages = messages[-10:]

def performance_test():
    """性能测试"""
    print("\n" + "=" * 60)
    print("性能测试")
    print("=" * 60)
    
    client = LLMAPIClient("http://your-server-ip:8000")
    
    # 测试不同长度的输入
    test_cases = [
        ("短输入", "你好"),
        ("中等输入", "请详细解释一下什么是深度学习，包括它的基本原理、主要应用领域和发展历程。"),
        ("长输入", "请详细解释一下什么是深度学习，包括它的基本原理、主要应用领域和发展历程。同时，请对比分析深度学习和传统机器学习方法的优缺点，并举例说明深度学习在各个行业中的实际应用案例。最后，请预测一下深度学习技术未来的发展趋势。")
    ]
    
    for test_name, content in test_cases:
        print(f"\n{test_name}测试...")
        messages = [{"role": "user", "content": content}]
        
        start_time = time.time()
        response = client.chat(messages, max_tokens=200, temperature=0.7)
        end_time = time.time()
        
        if "error" not in response:
            print(f"✅ 输入长度: {len(content)} 字符")
            print(f"✅ 处理时间: {response['processing_time']:.2f}秒")
            print(f"✅ Token使用: {response['usage']}")
            print(f"✅ 回复长度: {len(response['response'])} 字符")
            print(f"✅ 总耗时: {end_time - start_time:.2f}秒")
        else:
            print(f"❌ 测试失败: {response['error']}")

def main():
    """主函数"""
    print("LLM API客户端示例程序")
    print("请确保已正确配置服务器地址")
    
    # 检查服务器地址配置
    server_url = input("请输入服务器地址 (默认: http://localhost:8000): ").strip()
    if not server_url:
        server_url = "http://localhost:8000"
    
    # 更新全局客户端配置
    global LLMAPIClient
    original_init = LLMAPIClient.__init__
    
    def new_init(self, base_url=server_url, timeout=300):
        original_init(self, base_url, timeout)
    
    LLMAPIClient.__init__ = new_init
    
    print(f"\n使用服务器地址: {server_url}")
    
    try:
        # 运行各种示例
        demo_basic_usage()
        demo_stream_chat()
        demo_multi_turn_chat()
        demo_async_usage()
        performance_test()
        
        # 交互式聊天
        print("\n是否开始交互式聊天？(y/N): ", end="")
        if input().lower().startswith('y'):
            interactive_chat()
        
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n程序运行出错: {e}")

if __name__ == "__main__":
    main()
