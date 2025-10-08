"""
LLM模型API测试客户端
用于测试部署的大模型服务
"""

import requests
import json
import time
from typing import List, Dict

class LLMClient:
    """LLM API客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> Dict:
        """健康检查"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict:
        """发送聊天请求"""
        data = {
            "messages": messages,
            **kwargs
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/chat",
                json=data,
                timeout=300  # 5分钟超时
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def list_models(self) -> Dict:
        """列出可用模型"""
        try:
            response = self.session.get(f"{self.base_url}/models")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

def test_basic_functionality():
    """测试基本功能"""
    print("=" * 50)
    print("LLM模型API测试")
    print("=" * 50)
    
    client = LLMClient()
    
    # 1. 健康检查
    print("\n1. 健康检查...")
    health = client.health_check()
    if "error" in health:
        print(f"❌ 健康检查失败: {health['error']}")
        return
    else:
        print(f"✅ 服务状态: {health['status']}")
        print(f"✅ 模型加载: {health['model_loaded']}")
        print(f"✅ GPU内存使用: {health['gpu_memory']}")
        print(f"✅ 系统内存使用: {health['system_memory']:.1f}%")
    
    # 2. 列出模型
    print("\n2. 列出可用模型...")
    models = client.list_models()
    if "error" in models:
        print(f"❌ 获取模型列表失败: {models['error']}")
    else:
        print(f"✅ 可用模型: {models['models'][0]['id']}")
    
    # 3. 简单对话测试
    print("\n3. 简单对话测试...")
    messages = [
        {"role": "user", "content": "你好，请简单介绍一下你自己。"}
    ]
    
    start_time = time.time()
    response = client.chat(messages, max_tokens=200, temperature=0.7)
    end_time = time.time()
    
    if "error" in response:
        print(f"❌ 对话测试失败: {response['error']}")
    else:
        print(f"✅ 模型回复: {response['response']}")
        print(f"✅ 处理时间: {response['processing_time']:.2f}秒")
        print(f"✅ Token使用: {response['usage']}")
        print(f"✅ 客户端总耗时: {end_time - start_time:.2f}秒")
    
    # 4. 多轮对话测试
    print("\n4. 多轮对话测试...")
    messages = [
        {"role": "user", "content": "什么是人工智能？"},
        {"role": "assistant", "content": "人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。"},
        {"role": "user", "content": "请详细解释一下机器学习和深度学习的区别。"}
    ]
    
    response = client.chat(messages, max_tokens=300)
    if "error" in response:
        print(f"❌ 多轮对话测试失败: {response['error']}")
    else:
        print(f"✅ 模型回复: {response['response']}")
        print(f"✅ 处理时间: {response['processing_time']:.2f}秒")

def test_performance():
    """性能测试"""
    print("\n" + "=" * 50)
    print("性能测试")
    print("=" * 50)
    
    client = LLMClient()
    
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
        response = client.chat(messages, max_tokens=200)
        end_time = time.time()
        
        if "error" not in response:
            print(f"✅ 输入长度: {len(content)} 字符")
            print(f"✅ 处理时间: {response['processing_time']:.2f}秒")
            print(f"✅ Token使用: {response['usage']}")
            print(f"✅ 回复长度: {len(response['response'])} 字符")
        else:
            print(f"❌ 测试失败: {response['error']}")

def interactive_chat():
    """交互式聊天"""
    print("\n" + "=" * 50)
    print("交互式聊天 (输入 'quit' 退出)")
    print("=" * 50)
    
    client = LLMClient()
    
    messages = []
    
    while True:
        user_input = input("\n用户: ").strip()
        
        if user_input.lower() in ['quit', 'exit', '退出']:
            print("再见！")
            break
        
        if not user_input:
            continue
        
        messages.append({"role": "user", "content": user_input})
        
        print("AI: ", end="", flush=True)
        response = client.chat(messages, max_tokens=500, temperature=0.7)
        
        if "error" in response:
            print(f"❌ 错误: {response['error']}")
        else:
            print(response['response'])
            messages.append({"role": "assistant", "content": response['response']})
            
            # 保持对话历史在合理范围内
            if len(messages) > 10:
                messages = messages[-10:]

if __name__ == "__main__":
    try:
        # 基本功能测试
        test_basic_functionality()
        
        # 性能测试
        test_performance()
        
        # 交互式聊天
        print("\n是否开始交互式聊天？(y/N): ", end="")
        if input().lower().startswith('y'):
            interactive_chat()
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
