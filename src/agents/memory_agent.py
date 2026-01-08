# src/agents/memory_agent.py
from langchain.agents import create_agent
from src.tools.memory_tools import save_to_memory, search_memory, remember_context
from src.prompts.memory_agent_prompt import MEMORY_AGENT_SYSTEM_PROMPT
from src.memory.short_term_memory import ShortTermMemory
from config.settings import settings

class MemoryAgent:
    def __init__(self):
        self.tools = [
            save_to_memory,
            search_memory,
            remember_context
        ]
        
        self.short_term_memory = ShortTermMemory(max_messages=10)
        
        self.agent = self._create_agent()
    
    def _create_agent(self):
        agent = create_agent(
            model=settings.OPENAI_MODEL,
            tools=self.tools,
            system_prompt=MEMORY_AGENT_SYSTEM_PROMPT
        )
        
        return agent
    
    def chat(self, message: str):
        self.short_term_memory.add_user_message(message)
        history_messages = self.short_term_memory.get_messages()
        
        # ✅ Вызываем агента с историей
        result = self.agent.invoke({
            "messages": history_messages
        })
        
        # ✅ Извлекаем ответ агента
        if isinstance(result, dict):
            # Если результат - словарь, ищем последнее сообщение AI
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
        
        # ✅ Добавляем ответ агента в краткосрочную память
        self.short_term_memory.add_ai_message(ai_response)
        
        return ai_response
    
    def get_conversation_history(self) -> str:
        """Получить историю разговора"""
        return self.short_term_memory.get_conversation_summary()
    
    def clear_short_term_memory(self):
        """Очистить краткосрочную память"""
        self.short_term_memory.clear()
    
    def save_context(self, context: str):
        """Сохранить контекст текущей сессии"""
        return remember_context.invoke({"context": context})