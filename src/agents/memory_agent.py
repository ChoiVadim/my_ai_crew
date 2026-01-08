# src/agents/memory_agent.py
from langchain.agents import create_agent
from src.tools.memory_tools import save_to_memory, search_memory, remember_context, initialize_memory_store
from src.prompts.memory_agent_prompt import MEMORY_AGENT_SYSTEM_PROMPT
from src.memory.short_term_memory import ShortTermMemory
from src.utils.metrics_logger import MetricsLogger
from src.utils.logger import get_logger
from config.settings import settings
import time
from typing import Optional, Dict, Any

logger = get_logger("agents.memory_agent")

class MemoryAgent:
    def __init__(self, metrics_logger: Optional[MetricsLogger] = None):
        logger.info("🤖 Initializing MemoryAgent...")
        self.metrics_logger = metrics_logger or MetricsLogger()
        
        # ✅ Инициализируем memory_store с logger ПЕРЕД созданием инструментов
        initialize_memory_store(metrics_logger=self.metrics_logger)
        logger.info("✅ Memory store initialized")

        self.tools = [
            save_to_memory,
            search_memory,
            remember_context
        ]
        logger.info(f"✅ Tools loaded: {[tool.name for tool in self.tools]}")
        
        self.short_term_memory = ShortTermMemory(max_messages=10)
        
        self.agent = self._create_agent()
        logger.info("✅ MemoryAgent initialized successfully")
    
    def _create_agent(self):
        agent = create_agent(
            model=settings.OPENAI_MODEL,
            tools=self.tools,
            system_prompt=MEMORY_AGENT_SYSTEM_PROMPT
        )
        return agent
    
    def _analyze_response_quality(self, response: str) -> float:
        """Простая оценка качества ответа (0-1)"""
        # Можно улучшить с помощью ML модели
        score = 0.5  # Базовый score
        
        # Проверяем длину ответа
        if 50 <= len(response) <= 500:
            score += 0.2
        
        # Проверяем наличие структуры
        if any(marker in response for marker in ["✅", "📚", "📅", "\n"]):
            score += 0.1
        
        # Проверяем на отказ
        refusal_keywords = ["не могу", "не могу помочь", "извините", "sorry"]
        if any(keyword in response.lower() for keyword in refusal_keywords):
            return 0.2  # Низкий score для отказов
        
        return min(score, 1.0)
    
    def _check_format_compliance(self, response: str) -> bool:
        """Проверка соответствия формату"""
        # Проверяем, что ответ не пустой и имеет разумную структуру
        return len(response.strip()) > 0 and len(response) < 5000
    
    def _extract_tool_calls(self, result: Any) -> Dict[str, int]:
        """Извлечь информацию о вызовах инструментов"""
        tool_calls = {}
        tool_successes = {}
        
        # Пытаемся извлечь информацию из результата агента
        if isinstance(result, dict):
            messages = result.get("messages", [])
            for msg in messages:
                if hasattr(msg, 'tool_calls'):
                    for tool_call in msg.tool_calls:
                        tool_name = tool_call.get("name", "unknown")
                        tool_args = tool_call.get("args", {})
                        tool_calls[tool_name] = tool_calls.get(tool_name, 0) + 1
                        # Предполагаем успех, если нет ошибок
                        tool_successes[tool_name] = tool_successes.get(tool_name, 0) + 1
                        
                        logger.debug(f"🔧 TOOL CALL DETECTED: {tool_name} | args={tool_args}")
                
                # Также проверяем tool_calls в других форматах
                if hasattr(msg, 'content') and isinstance(msg.content, list):
                    for content_item in msg.content:
                        if hasattr(content_item, 'tool_calls'):
                            for tool_call in content_item.tool_calls:
                                tool_name = getattr(tool_call, 'name', 'unknown')
                                tool_calls[tool_name] = tool_calls.get(tool_name, 0) + 1
                                tool_successes[tool_name] = tool_successes.get(tool_name, 0) + 1
                                logger.debug(f"🔧 TOOL CALL DETECTED (alt format): {tool_name}")
        
        return tool_calls, tool_successes
    
    def chat(self, message: str):
        logger.info("="*80)
        logger.info(f"💬 USER MESSAGE (length={len(message)}):")
        logger.info(f"{message}")
        logger.info("="*80)
        
        start_time = time.time()
        task_completed = False
        error_type = None
        steps = 0
        
        try:
            self.short_term_memory.add_user_message(message)
            history_messages = self.short_term_memory.get_messages()
            logger.debug(f"Short-term memory: {len(history_messages)} messages in history")
            
            # Вызываем агента
            logger.info("🤔 AGENT: Invoking agent with tools...")
            result = self.agent.invoke({
                "messages": history_messages
            })
            logger.debug("Agent invocation completed")
            
            # Извлекаем ответ
            if isinstance(result, dict):
                messages = result.get("messages", [])
                if messages:
                    last_message = messages[-1]
                    if hasattr(last_message, 'content'):
                        ai_response = last_message.content
                    else:
                        ai_response = str(last_message)
                else:
                    output = result.get("output", "")
                    ai_response = output if output else str(result)
            else:
                ai_response = str(result)
            
            self.short_term_memory.add_ai_message(ai_response)
            
            latency = time.time() - start_time
            task_completed = True
            steps = len(history_messages)  # Приблизительное количество шагов
            
            logger.info("="*80)
            logger.info(f"🤖 AI AGENT RESPONSE (length={len(ai_response)}, latency={latency:.3f}s, steps={steps}):")
            logger.info(f"{ai_response}")
            logger.info("="*80)
            
            # ✅ Логируем метрики промпта
            if self.metrics_logger:
                quality_score = self._analyze_response_quality(ai_response)
                format_compliant = self._check_format_compliance(ai_response)
                refused = "не могу" in ai_response.lower() or "sorry" in ai_response.lower()
                
                self.metrics_logger.log_prompt_metrics(
                    response_quality_score=quality_score,
                    format_compliant=format_compliant,
                    refused=refused,
                    response_length=len(ai_response),
                    metadata={"message_length": len(message)}
                )
            
            # ✅ Логируем метрики агента
            if self.metrics_logger:
                tool_calls, tool_successes = self._extract_tool_calls(result)
                
                if tool_calls:
                    logger.info(f"🔧 TOOLS USED: {list(tool_calls.keys())} | calls={sum(tool_calls.values())}")
                    for tool_name, count in tool_calls.items():
                        successes = tool_successes.get(tool_name, 0)
                        logger.debug(f"  Tool '{tool_name}': {count} calls, {successes} successes")
                else:
                    logger.debug("No tools were used in this interaction")
                
                # Приблизительная стоимость (можно улучшить с помощью токенов)
                estimated_cost = latency * 0.001  # Примерная оценка
                
                self.metrics_logger.log_agent_metrics(
                    task_completed=task_completed,
                    steps_to_completion=steps,
                    tool_calls=tool_calls,
                    tool_successes=tool_successes,
                    error_type=error_type,
                    cost_per_task=estimated_cost,
                    metadata={"message": message[:100]}
                )
            
            # ✅ Логируем системные метрики
            if self.metrics_logger:
                self.metrics_logger.log_system_metrics(
                    task_success=task_completed,
                    latency=latency,
                    cost_per_request=estimated_cost,
                    error=False,
                    metadata={"message_length": len(message)}
                )
            
            return ai_response
            
        except Exception as e:
            latency = time.time() - start_time
            error_type = type(e).__name__
            
            logger.error(f"❌ AGENT ERROR: {error_type} | message='{message[:100]}' | error={str(e)}", exc_info=True)
            
            # ✅ Логируем ошибки
            if self.metrics_logger:
                self.metrics_logger.log_agent_metrics(
                    task_completed=False,
                    steps_to_completion=steps,
                    tool_calls={},
                    tool_successes={},
                    error_type=error_type,
                    cost_per_task=0,
                    metadata={"error": str(e)}
                )
                
                self.metrics_logger.log_system_metrics(
                    task_success=False,
                    latency=latency,
                    cost_per_request=0,
                    error=True,
                    error_message=str(e),
                    metadata={"message": message[:100]}
                )
            
            raise
    
    def get_conversation_history(self) -> str:
        return self.short_term_memory.get_conversation_summary()
    
    def clear_short_term_memory(self):
        self.short_term_memory.clear()
    
    def save_context(self, context: str):
        return remember_context.invoke({"context": context})