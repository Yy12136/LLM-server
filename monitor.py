"""
LLM服务监控脚本
实时监控模型服务的运行状态、性能指标和资源使用情况
"""

import time
import json
import requests
import psutil
import GPUtil
from datetime import datetime
from typing import Dict, List
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LLMMonitor:
    """LLM服务监控类"""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.session = requests.Session()
        self.session.timeout = 10
    
    def get_system_metrics(self) -> Dict:
        """获取系统指标"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 内存使用情况
            memory = psutil.virtual_memory()
            
            # 磁盘使用情况
            disk = psutil.disk_usage('/')
            
            # GPU信息
            gpu_info = {}
            try:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    gpu_info[f"gpu_{i}"] = {
                        "name": gpu.name,
                        "memory_used": gpu.memoryUsed,
                        "memory_total": gpu.memoryTotal,
                        "memory_percent": gpu.memoryUtil * 100,
                        "temperature": gpu.temperature,
                        "utilization": gpu.load * 100
                    }
            except Exception as e:
                gpu_info = {"error": str(e)}
            
            return {
                "timestamp": datetime.now().isoformat(),
                "cpu_percent": cpu_percent,
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                    "used": memory.used
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": (disk.used / disk.total) * 100
                },
                "gpu": gpu_info
            }
        except Exception as e:
            logger.error(f"获取系统指标失败: {e}")
            return {"error": str(e)}
    
    def get_api_health(self) -> Dict:
        """获取API健康状态"""
        try:
            response = self.session.get(f"{self.api_url}/health")
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    def test_api_performance(self) -> Dict:
        """测试API性能"""
        try:
            test_message = {
                "messages": [{"role": "user", "content": "你好"}],
                "max_tokens": 50
            }
            
            start_time = time.time()
            response = self.session.post(
                f"{self.api_url}/chat",
                json=test_message,
                timeout=60
            )
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "status": "success",
                    "response_time": end_time - start_time,
                    "processing_time": result.get("processing_time", 0),
                    "tokens_generated": result.get("usage", {}).get("completion_tokens", 0)
                }
            else:
                return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    def generate_report(self) -> Dict:
        """生成监控报告"""
        system_metrics = self.get_system_metrics()
        api_health = self.get_api_health()
        performance = self.test_api_performance()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system": system_metrics,
            "api_health": api_health,
            "performance": performance
        }
    
    def check_alerts(self, metrics: Dict) -> List[str]:
        """检查告警条件"""
        alerts = []
        
        # 系统资源告警
        if "system" in metrics:
            system = metrics["system"]
            
            # CPU使用率告警
            if system.get("cpu_percent", 0) > 90:
                alerts.append(f"CPU使用率过高: {system['cpu_percent']:.1f}%")
            
            # 内存使用率告警
            if system.get("memory", {}).get("percent", 0) > 90:
                alerts.append(f"内存使用率过高: {system['memory']['percent']:.1f}%")
            
            # 磁盘使用率告警
            if system.get("disk", {}).get("percent", 0) > 90:
                alerts.append(f"磁盘使用率过高: {system['disk']['percent']:.1f}%")
            
            # GPU内存告警
            gpu = system.get("gpu", {})
            for gpu_id, gpu_info in gpu.items():
                if isinstance(gpu_info, dict) and "memory_percent" in gpu_info:
                    if gpu_info["memory_percent"] > 95:
                        alerts.append(f"GPU {gpu_id} 内存使用率过高: {gpu_info['memory_percent']:.1f}%")
        
        # API健康告警
        if "api_health" in metrics:
            api = metrics["api_health"]
            if api.get("status") != "healthy":
                alerts.append(f"API服务状态异常: {api.get('status', 'unknown')}")
        
        # 性能告警
        if "performance" in metrics:
            perf = metrics["performance"]
            if perf.get("response_time", 0) > 30:
                alerts.append(f"API响应时间过长: {perf['response_time']:.2f}秒")
        
        return alerts
    
    def log_metrics(self, metrics: Dict):
        """记录指标到日志"""
        timestamp = metrics.get("timestamp", "")
        
        # 系统指标
        system = metrics.get("system", {})
        cpu = system.get("cpu_percent", 0)
        memory = system.get("memory", {}).get("percent", 0)
        
        # GPU指标
        gpu_info = system.get("gpu", {})
        gpu_memory = 0
        if isinstance(gpu_info, dict):
            for gpu_id, gpu_data in gpu_info.items():
                if isinstance(gpu_data, dict) and "memory_percent" in gpu_data:
                    gpu_memory = max(gpu_memory, gpu_data["memory_percent"])
        
        # API状态
        api_health = metrics.get("api_health", {})
        api_status = api_health.get("status", "unknown")
        
        # 性能指标
        performance = metrics.get("performance", {})
        response_time = performance.get("response_time", 0)
        
        logger.info(
            f"监控指标 - "
            f"CPU: {cpu:.1f}%, "
            f"内存: {memory:.1f}%, "
            f"GPU内存: {gpu_memory:.1f}%, "
            f"API状态: {api_status}, "
            f"响应时间: {response_time:.2f}s"
        )
    
    def save_metrics_to_file(self, metrics: Dict, filename: str = "metrics.json"):
        """保存指标到文件"""
        try:
            # 读取现有数据
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                data = {"records": []}
            
            # 添加新记录
            data["records"].append(metrics)
            
            # 只保留最近1000条记录
            if len(data["records"]) > 1000:
                data["records"] = data["records"][-1000:]
            
            # 保存数据
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"保存指标到文件失败: {e}")

def run_monitoring_loop(interval: int = 60):
    """运行监控循环"""
    monitor = LLMMonitor()
    
    logger.info("开始监控LLM服务...")
    
    try:
        while True:
            # 生成监控报告
            metrics = monitor.generate_report()
            
            # 记录指标
            monitor.log_metrics(metrics)
            
            # 保存到文件
            monitor.save_metrics_to_file(metrics)
            
            # 检查告警
            alerts = monitor.check_alerts(metrics)
            if alerts:
                logger.warning(f"检测到告警: {'; '.join(alerts)}")
            
            # 等待下次监控
            time.sleep(interval)
            
    except KeyboardInterrupt:
        logger.info("监控被用户中断")
    except Exception as e:
        logger.error(f"监控过程中发生错误: {e}")

def generate_summary_report(filename: str = "metrics.json"):
    """生成汇总报告"""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        records = data.get("records", [])
        if not records:
            print("没有监控数据")
            return
        
        # 统计信息
        total_records = len(records)
        cpu_values = [r.get("system", {}).get("cpu_percent", 0) for r in records]
        memory_values = [r.get("system", {}).get("memory", {}).get("percent", 0) for r in records]
        response_times = [r.get("performance", {}).get("response_time", 0) for r in records]
        
        # 过滤有效值
        cpu_values = [v for v in cpu_values if v > 0]
        memory_values = [v for v in memory_values if v > 0]
        response_times = [v for v in response_times if v > 0]
        
        print("=" * 50)
        print("LLM服务监控汇总报告")
        print("=" * 50)
        print(f"监控记录数: {total_records}")
        print(f"时间范围: {records[0].get('timestamp', 'unknown')} 到 {records[-1].get('timestamp', 'unknown')}")
        
        if cpu_values:
            print(f"\nCPU使用率:")
            print(f"  平均: {sum(cpu_values)/len(cpu_values):.1f}%")
            print(f"  最高: {max(cpu_values):.1f}%")
            print(f"  最低: {min(cpu_values):.1f}%")
        
        if memory_values:
            print(f"\n内存使用率:")
            print(f"  平均: {sum(memory_values)/len(memory_values):.1f}%")
            print(f"  最高: {max(memory_values):.1f}%")
            print(f"  最低: {min(memory_values):.1f}%")
        
        if response_times:
            print(f"\nAPI响应时间:")
            print(f"  平均: {sum(response_times)/len(response_times):.2f}秒")
            print(f"  最高: {max(response_times):.2f}秒")
            print(f"  最低: {min(response_times):.2f}秒")
        
        # API状态统计
        api_statuses = [r.get("api_health", {}).get("status", "unknown") for r in records]
        status_counts = {}
        for status in api_statuses:
            status_counts[status] = status_counts.get(status, 0) + 1
        
        print(f"\nAPI状态统计:")
        for status, count in status_counts.items():
            print(f"  {status}: {count} 次 ({count/total_records*100:.1f}%)")
        
    except Exception as e:
        print(f"生成汇总报告失败: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM服务监控工具")
    parser.add_argument("--interval", type=int, default=60, help="监控间隔(秒)")
    parser.add_argument("--report", action="store_true", help="生成汇总报告")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API服务地址")
    
    args = parser.parse_args()
    
    if args.report:
        generate_summary_report()
    else:
        monitor = LLMMonitor(args.api_url)
        run_monitoring_loop(args.interval)
