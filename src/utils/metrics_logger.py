# src/utils/metrics_logger.py
import json
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

class MetricsLogger:
    """Логирование метрик для AI агента"""
    
    def __init__(self, log_dir: str = "./data/metrics"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Метрики для разных компонентов
        self.prompt_metrics: List[Dict] = []
        self.rag_metrics: List[Dict] = []
        self.agent_metrics: List[Dict] = []
        self.system_metrics: List[Dict] = []
        
        # Агрегированные метрики
        self.aggregated_metrics = {
            "prompts": {
                "total_requests": 0,
                "total_refusals": 0,
                "total_response_length": 0,
                "format_compliance_count": 0,
                "quality_scores": []
            },
            "rag": {
                "total_retrievals": 0,
                "total_chunks_retrieved": 0,
                "total_retrieval_latency": 0,
                "confidence_scores": [],
                "source_diversity": []
            },
            "agents": {
                "total_tasks": 0,
                "completed_tasks": 0,
                "total_steps": 0,
                "tool_calls": {},
                "tool_successes": {},
                "errors": {},
                "total_cost": 0
            },
            "system": {
                "total_requests": 0,
                "successful_requests": 0,
                "total_latency": 0,
                "total_cost": 0,
                "errors": 0,
                "start_time": datetime.now().isoformat(),
                "uptime_seconds": 0
            }
        }
    
    # ========== PROMPT METRICS ==========
    
    def log_prompt_metrics(
        self,
        response_quality_score: float,
        format_compliant: bool,
        refused: bool,
        response_length: int,
        metadata: Optional[Dict] = None
    ):
        """Логировать метрики промпта"""
        timestamp = datetime.now().isoformat()
        metric = {
            "timestamp": timestamp,
            "response_quality_score": response_quality_score,
            "format_compliant": format_compliant,
            "refused": refused,
            "response_length": response_length,
            "metadata": metadata or {}
        }
        
        self.prompt_metrics.append(metric)
        
        # Обновляем агрегированные метрики
        self.aggregated_metrics["prompts"]["total_requests"] += 1
        if refused:
            self.aggregated_metrics["prompts"]["total_refusals"] += 1
        if format_compliant:
            self.aggregated_metrics["prompts"]["format_compliance_count"] += 1
        self.aggregated_metrics["prompts"]["total_response_length"] += response_length
        self.aggregated_metrics["prompts"]["quality_scores"].append(response_quality_score)
        
        self._save_metric("prompts", metric)
    
    # ========== RAG METRICS ==========
    
    def log_rag_metrics(
        self,
        retrieval_confidence_scores: List[float],
        num_chunks_retrieved: int,
        source_diversity: int,
        retrieval_latency: float,
        metadata: Optional[Dict] = None
    ):
        """Логировать метрики RAG"""
        timestamp = datetime.now().isoformat()
        metric = {
            "timestamp": timestamp,
            "retrieval_confidence_scores": retrieval_confidence_scores,
            "num_chunks_retrieved": num_chunks_retrieved,
            "source_diversity": source_diversity,
            "retrieval_latency": retrieval_latency,
            "metadata": metadata or {}
        }
        
        self.rag_metrics.append(metric)
        
        # Обновляем агрегированные метрики
        self.aggregated_metrics["rag"]["total_retrievals"] += 1
        self.aggregated_metrics["rag"]["total_chunks_retrieved"] += num_chunks_retrieved
        self.aggregated_metrics["rag"]["total_retrieval_latency"] += retrieval_latency
        self.aggregated_metrics["rag"]["confidence_scores"].extend(retrieval_confidence_scores)
        self.aggregated_metrics["rag"]["source_diversity"].append(source_diversity)
        
        self._save_metric("rag", metric)
    
    # ========== AGENT METRICS ==========
    
    def log_agent_metrics(
        self,
        task_completed: bool,
        steps_to_completion: int,
        tool_calls: Dict[str, int],
        tool_successes: Dict[str, int],
        error_type: Optional[str] = None,
        cost_per_task: float = 0.0,
        metadata: Optional[Dict] = None
    ):
        """Логировать метрики агента"""
        timestamp = datetime.now().isoformat()
        metric = {
            "timestamp": timestamp,
            "task_completed": task_completed,
            "steps_to_completion": steps_to_completion,
            "tool_calls": tool_calls,
            "tool_successes": tool_successes,
            "error_type": error_type,
            "cost_per_task": cost_per_task,
            "metadata": metadata or {}
        }
        
        self.agent_metrics.append(metric)
        
        # Обновляем агрегированные метрики
        self.aggregated_metrics["agents"]["total_tasks"] += 1
        if task_completed:
            self.aggregated_metrics["agents"]["completed_tasks"] += 1
        self.aggregated_metrics["agents"]["total_steps"] += steps_to_completion
        self.aggregated_metrics["agents"]["total_cost"] += cost_per_task
        
        # Обновляем статистику по инструментам
        for tool_name, count in tool_calls.items():
            if tool_name not in self.aggregated_metrics["agents"]["tool_calls"]:
                self.aggregated_metrics["agents"]["tool_calls"][tool_name] = 0
            self.aggregated_metrics["agents"]["tool_calls"][tool_name] += count
        
        for tool_name, count in tool_successes.items():
            if tool_name not in self.aggregated_metrics["agents"]["tool_successes"]:
                self.aggregated_metrics["agents"]["tool_successes"][tool_name] = 0
            self.aggregated_metrics["agents"]["tool_successes"][tool_name] += count
        
        # Обновляем ошибки
        if error_type:
            if error_type not in self.aggregated_metrics["agents"]["errors"]:
                self.aggregated_metrics["agents"]["errors"][error_type] = 0
            self.aggregated_metrics["agents"]["errors"][error_type] += 1
        
        self._save_metric("agents", metric)
    
    # ========== SYSTEM METRICS ==========
    
    def log_system_metrics(
        self,
        task_success: bool,
        user_satisfaction: Optional[float] = None,
        latency: float = 0.0,
        cost_per_request: float = 0.0,
        error: bool = False,
        error_message: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """Логировать системные метрики"""
        timestamp = datetime.now().isoformat()
        metric = {
            "timestamp": timestamp,
            "task_success": task_success,
            "user_satisfaction": user_satisfaction,
            "latency": latency,
            "cost_per_request": cost_per_request,
            "error": error,
            "error_message": error_message,
            "metadata": metadata or {}
        }
        
        self.system_metrics.append(metric)
        
        # Обновляем агрегированные метрики
        self.aggregated_metrics["system"]["total_requests"] += 1
        if task_success:
            self.aggregated_metrics["system"]["successful_requests"] += 1
        if error:
            self.aggregated_metrics["system"]["errors"] += 1
        self.aggregated_metrics["system"]["total_latency"] += latency
        self.aggregated_metrics["system"]["total_cost"] += cost_per_request
        
        # Обновляем uptime
        start_time = datetime.fromisoformat(self.aggregated_metrics["system"]["start_time"])
        uptime = (datetime.now() - start_time).total_seconds()
        self.aggregated_metrics["system"]["uptime_seconds"] = uptime
        
        self._save_metric("system", metric)
    
    # ========== UTILITY METHODS ==========
    
    def _save_metric(self, category: str, metric: Dict):
        """Сохранить метрику в файл"""
        log_file = self.log_dir / f"{category}_metrics.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(metric, ensure_ascii=False) + "\n")
    
    def get_aggregated_metrics(self) -> Dict:
        """Получить агрегированные метрики"""
        metrics = self.aggregated_metrics.copy()
        
        # Вычисляем средние значения
        if metrics["prompts"]["total_requests"] > 0:
            metrics["prompts"]["average_response_length"] = (
                metrics["prompts"]["total_response_length"] / 
                metrics["prompts"]["total_requests"]
            )
            metrics["prompts"]["refusal_rate"] = (
                metrics["prompts"]["total_refusals"] / 
                metrics["prompts"]["total_requests"]
            )
            metrics["prompts"]["format_compliance_rate"] = (
                metrics["prompts"]["format_compliance_count"] / 
                metrics["prompts"]["total_requests"]
            )
            if metrics["prompts"]["quality_scores"]:
                metrics["prompts"]["average_quality_score"] = sum(
                    metrics["prompts"]["quality_scores"]
                ) / len(metrics["prompts"]["quality_scores"])
        
        if metrics["rag"]["total_retrievals"] > 0:
            metrics["rag"]["average_chunks_retrieved"] = (
                metrics["rag"]["total_chunks_retrieved"] / 
                metrics["rag"]["total_retrievals"]
            )
            metrics["rag"]["average_retrieval_latency"] = (
                metrics["rag"]["total_retrieval_latency"] / 
                metrics["rag"]["total_retrievals"]
            )
            if metrics["rag"]["confidence_scores"]:
                metrics["rag"]["average_confidence_score"] = sum(
                    metrics["rag"]["confidence_scores"]
                ) / len(metrics["rag"]["confidence_scores"])
            if metrics["rag"]["source_diversity"]:
                metrics["rag"]["average_source_diversity"] = sum(
                    metrics["rag"]["source_diversity"]
                ) / len(metrics["rag"]["source_diversity"])
        
        if metrics["agents"]["total_tasks"] > 0:
            metrics["agents"]["task_completion_rate"] = (
                metrics["agents"]["completed_tasks"] / 
                metrics["agents"]["total_tasks"]
            )
            metrics["agents"]["average_steps_to_completion"] = (
                metrics["agents"]["total_steps"] / 
                metrics["agents"]["total_tasks"]
            )
            metrics["agents"]["average_cost_per_task"] = (
                metrics["agents"]["total_cost"] / 
                metrics["agents"]["total_tasks"]
            )
            # Вычисляем success rate для каждого инструмента
            metrics["agents"]["tool_success_rates"] = {}
            for tool_name in metrics["agents"]["tool_calls"]:
                calls = metrics["agents"]["tool_calls"][tool_name]
                successes = metrics["agents"]["tool_successes"].get(tool_name, 0)
                metrics["agents"]["tool_success_rates"][tool_name] = (
                    successes / calls if calls > 0 else 0
                )
        
        if metrics["system"]["total_requests"] > 0:
            metrics["system"]["task_success_rate"] = (
                metrics["system"]["successful_requests"] / 
                metrics["system"]["total_requests"]
            )
            metrics["system"]["average_latency"] = (
                metrics["system"]["total_latency"] / 
                metrics["system"]["total_requests"]
            )
            metrics["system"]["average_cost_per_request"] = (
                metrics["system"]["total_cost"] / 
                metrics["system"]["total_requests"]
            )
            metrics["system"]["error_rate"] = (
                metrics["system"]["errors"] / 
                metrics["system"]["total_requests"]
            )
        
        return metrics
    
    def save_aggregated_metrics(self):
        """Сохранить агрегированные метрики в файл"""
        metrics = self.get_aggregated_metrics()
        metrics_file = self.log_dir / "aggregated_metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    def print_metrics_summary(self):
        """Вывести краткую сводку метрик"""
        metrics = self.get_aggregated_metrics()
        
        print("\n" + "="*60)
        print("📊 СВОДКА МЕТРИК")
        print("="*60)
        
        # PROMPTS
        print("\n📝 PROMPTS:")
        if metrics["prompts"]["total_requests"] > 0:
            print(f"  • Всего запросов: {metrics['prompts']['total_requests']}")
            print(f"  • Средний quality score: {metrics['prompts'].get('average_quality_score', 0):.2f}")
            print(f"  • Format compliance rate: {metrics['prompts'].get('format_compliance_rate', 0):.2%}")
            print(f"  • Refusal rate: {metrics['prompts'].get('refusal_rate', 0):.2%}")
            print(f"  • Средняя длина ответа: {metrics['prompts'].get('average_response_length', 0):.0f} символов")
        
        # RAG
        print("\n🔍 RAG:")
        if metrics["rag"]["total_retrievals"] > 0:
            print(f"  • Всего поисков: {metrics['rag']['total_retrievals']}")
            print(f"  • Средний confidence score: {metrics['rag'].get('average_confidence_score', 0):.2f}")
            print(f"  • Среднее количество chunks: {metrics['rag'].get('average_chunks_retrieved', 0):.1f}")
            print(f"  • Средняя source diversity: {metrics['rag'].get('average_source_diversity', 0):.1f}")
            print(f"  • Средняя latency: {metrics['rag'].get('average_retrieval_latency', 0):.3f}с")
        
        # AGENTS
        print("\n🤖 AGENTS:")
        if metrics["agents"]["total_tasks"] > 0:
            print(f"  • Всего задач: {metrics['agents']['total_tasks']}")
            print(f"  • Task completion rate: {metrics['agents'].get('task_completion_rate', 0):.2%}")
            print(f"  • Среднее количество шагов: {metrics['agents'].get('average_steps_to_completion', 0):.1f}")
            print(f"  • Средняя стоимость задачи: ${metrics['agents'].get('average_cost_per_task', 0):.4f}")
            if metrics["agents"]["tool_success_rates"]:
                print("  • Tool success rates:")
                for tool, rate in metrics["agents"]["tool_success_rates"].items():
                    print(f"    - {tool}: {rate:.2%}")
            if metrics["agents"]["errors"]:
                print("  • Ошибки:")
                for error_type, count in metrics["agents"]["errors"].items():
                    print(f"    - {error_type}: {count}")
        
        # SYSTEM
        print("\n⚙️ SYSTEM:")
        if metrics["system"]["total_requests"] > 0:
            print(f"  • Всего запросов: {metrics['system']['total_requests']}")
            print(f"  • Task success rate: {metrics['system'].get('task_success_rate', 0):.2%}")
            print(f"  • Средняя latency: {metrics['system'].get('average_latency', 0):.3f}с")
            print(f"  • Средняя стоимость запроса: ${metrics['system'].get('average_cost_per_request', 0):.4f}")
            print(f"  • Error rate: {metrics['system'].get('error_rate', 0):.2%}")
            uptime_hours = metrics["system"]["uptime_seconds"] / 3600
            print(f"  • Uptime: {uptime_hours:.2f} часов")
        
        print("="*60 + "\n")