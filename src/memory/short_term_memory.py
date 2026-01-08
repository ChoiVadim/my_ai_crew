# src/memory/short_term_memory.py
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from typing import List, Optional
from datetime import datetime

class ShortTermMemory:
    """Краткосрочная память для хранения истории текущей сессии"""
    
    def __init__(self, max_messages: int = 10):
        """
        Инициализация краткосрочной памяти
        
        Args:
            max_messages: Максимальное количество сообщений для хранения
        """
        self.max_messages = max_messages
        self.messages: List[BaseMessage] = []
        self.session_start = datetime.now()
    
    def add_user_message(self, content: str):
        """Добавить сообщение пользователя"""
        message = HumanMessage(content=content)
        self.messages.append(message)
        self._trim_messages()
    
    def add_ai_message(self, content: str):
        """Добавить сообщение агента"""
        message = AIMessage(content=content)
        self.messages.append(message)
        self._trim_messages()
    
    def get_messages(self) -> List[BaseMessage]:
        """Получить все сообщения"""
        return self.messages.copy()
    
    def get_recent_messages(self, n: int = 5) -> List[BaseMessage]:
        """Получить последние N сообщений"""
        return self.messages[-n:] if len(self.messages) > n else self.messages
    
    def clear(self):
        """Очистить память"""
        self.messages = []
        self.session_start = datetime.now()
    
    def _trim_messages(self):
        """Обрезать сообщения до максимального количества"""
        if len(self.messages) > self.max_messages:
            # Оставляем последние max_messages сообщений
            self.messages = self.messages[-self.max_messages:]
    
    def get_conversation_summary(self) -> str:
        """Получить краткое резюме разговора"""
        if not self.messages:
            return "Нет истории разговора."
        
        summary = f"История разговора ({len(self.messages)} сообщений):\n"
        for i, msg in enumerate(self.messages[-5:], 1):  # Последние 5 сообщений
            role = "Пользователь" if isinstance(msg, HumanMessage) else "Агент"
            content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            summary += f"{i}. {role}: {content}\n"
        
        return summary
    
    def __len__(self):
        return len(self.messages)