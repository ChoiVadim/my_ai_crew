# src/tools/memory_tools.py
from langchain.tools import tool
from src.memory.memory_store import MemoryStore

# Глобальный экземпляр хранилища памяти
memory_store = MemoryStore()

@tool
def save_to_memory(content: str, category: str = "general") -> str:
    """
    Сохранить важную информацию в долгосрочную память агента.
    
    Args:
        content: Информация для сохранения
        category: Категория информации (work, personal, project, etc.)
    
    Use this when:
    - Пользователь просит запомнить что-то важное
    - Нужно сохранить контекст разговора
    - Получена важная информация, которая может понадобиться позже
    """
    metadata = {"category": category}
    result = memory_store.save_memory(content, metadata)
    return result  # Уже содержит эмодзи и сообщение

@tool
def search_memory(query: str, limit: int = 5) -> str:
    """
    Найти информацию из сохраненной памяти по запросу.
    
    Args:
        query: Поисковый запрос
        limit: Количество результатов (по умолчанию 5)
    
    Use this when:
    - Пользователь спрашивает о чем-то, что могло быть сохранено ранее
    - Нужно вспомнить контекст предыдущих разговоров
    - Требуется найти связанную информацию
    """
    # ✅ Используем метод с scores для лучшей информации
    memories = memory_store.retrieve_memories_with_scores(query, k=limit)
    
    if not memories:
        return "❌ В памяти не найдено релевантной информации."
    
    result = "📚 Найденная информация из памяти:\n\n"
    for i, mem in enumerate(memories, 1):
        # Для similarity_search relevance_score будет 1.0
        # Для similarity_search_with_score будет реальный score
        score_text = ""
        if mem['relevance_score'] != 1.0:
            score_text = f" [Релевантность: {1-mem['relevance_score']:.2%}]"
        
        result += f"{i}.{score_text}\n"
        result += f"   {mem['content']}\n"
        if mem['metadata'].get('timestamp'):
            result += f"   📅 Сохранено: {mem['metadata']['timestamp']}\n"
        result += "\n"
    
    return result

@tool
def remember_context(context: str) -> str:
    """
    Сохранить контекст текущего разговора или рабочей сессии.
    Полезно для сохранения состояния работы над задачей.
    
    Args:
        context: Описание текущего контекста или состояния
    """
    metadata = {"category": "context", "type": "session_context"}
    result = memory_store.save_memory(context, metadata)
    return result