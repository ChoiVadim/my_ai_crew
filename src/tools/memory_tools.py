# src/tools/memory_tools.py
from langchain.tools import tool
from src.memory.memory_store import MemoryStore
from src.utils.logger import get_logger
from typing import Optional

logger = get_logger("tools.memory_tools")

# ✅ Глобальный экземпляр хранилища памяти
memory_store: Optional[MemoryStore] = None

def initialize_memory_store(metrics_logger=None):
    """Инициализировать memory_store с metrics_logger"""
    global memory_store
    if memory_store is None:
        # Создаем новый экземпляр с logger
        memory_store = MemoryStore(metrics_logger=metrics_logger)
    else:
        # Обновляем существующий экземпляр с logger
        memory_store.set_metrics_logger(metrics_logger)
    return memory_store

# Инициализируем по умолчанию (будет обновлен с logger позже)
if memory_store is None:
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
    global memory_store
    if memory_store is None:
        memory_store = MemoryStore()
    
    logger.info("="*80)
    logger.info(f"🔧 TOOL CALL: save_to_memory | category={category} | content_length={len(content)}")
    logger.info(f"📝 CONTENT TO SAVE:")
    logger.info(f"{content}")
    logger.info("="*80)
    
    metadata = {"category": category}
    result = memory_store.save_memory(content, metadata)
    
    logger.info(f"✅ TOOL RESULT: save_to_memory | result={result}")
    return result

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
    global memory_store
    if memory_store is None:
        memory_store = MemoryStore()
    
    logger.info("="*80)
    logger.info(f"🔍 TOOL CALL: search_memory | query='{query}' | limit={limit}")
    logger.info("="*80)
    
    # ✅ Используем метод с scores для лучшей информации
    memories = memory_store.retrieve_memories_with_scores(query, k=limit)
    
    if not memories:
        logger.info(f"❌ TOOL RESULT: search_memory | found=0 results")
        return "❌ В памяти не найдено релевантной информации."
    
    logger.info("="*80)
    logger.info(f"✅ TOOL RESULT: search_memory | found={len(memories)} results")
    logger.info("="*80)
    
    for i, mem in enumerate(memories, 1):
        logger.info(f"\n📄 RETRIEVED CHUNK {i}:")
        logger.info(f"  Score: {mem.get('relevance_score', 'N/A'):.4f}")
        logger.info(f"  Category: {mem.get('metadata', {}).get('category', 'N/A')}")
        logger.info(f"  Timestamp: {mem.get('metadata', {}).get('timestamp', 'N/A')}")
        logger.info(f"  Content:")
        logger.info(f"  {mem.get('content', '')}")
        logger.info("-"*80)
    
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
    global memory_store
    if memory_store is None:
        memory_store = MemoryStore()
    
    logger.info("="*80)
    logger.info(f"🔧 TOOL CALL: remember_context | context_length={len(context)}")
    logger.info(f"📝 CONTEXT TO SAVE:")
    logger.info(f"{context}")
    logger.info("="*80)
    
    metadata = {"category": "context", "type": "session_context"}
    result = memory_store.save_memory(context, metadata)
    
    logger.info(f"✅ TOOL RESULT: remember_context | result={result}")
    return result