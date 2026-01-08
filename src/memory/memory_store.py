# src/memory/memory_store.py
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config.settings import settings
import os
from datetime import datetime
import uuid
from src.utils.metrics_logger import MetricsLogger
from src.utils.logger import get_logger
import time

logger = get_logger("memory.store")

class MemoryStore:
    """Управление долгосрочной памятью агента"""
    
    def __init__(self, persist_directory=None, metrics_logger=None):
        self.metrics_logger = metrics_logger
        # Используем настройки из конфига
        persist_directory = persist_directory or settings.MEMORY_DIR
        os.makedirs(persist_directory, exist_ok=True)
        
        # ✅ Инициализируем embeddings согласно документации
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",  # Можно использовать более дешевую модель
            openai_api_key=settings.OPENAI_API_KEY
        )
        self.persist_directory = persist_directory
        
        # ✅ Инициализируем Chroma согласно документации LangChain
        # Согласно docs: Chroma(collection_name, embedding_function, persist_directory)
        try:
            # Пытаемся загрузить существующую коллекцию
            self.vectorstore = Chroma(
                collection_name="memory_store",
                embedding_function=self.embeddings,
                persist_directory=persist_directory
            )
            # Проверяем, что коллекция существует и не пустая
            # Если коллекция пустая, это нормально - просто продолжаем
        except Exception:
            # Если коллекция не существует, создаем новую
            # Chroma автоматически создаст коллекцию при первом add_documents
            self.vectorstore = Chroma(
                collection_name="memory_store",
                embedding_function=self.embeddings,
                persist_directory=persist_directory
            )
        
        # Инициализируем text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.MEMORY_CHUNK_SIZE,
            chunk_overlap=settings.MEMORY_CHUNK_OVERLAP
        )
    def set_metrics_logger(self, metrics_logger: MetricsLogger):
        """Установить metrics_logger после создания экземпляра"""
        self.metrics_logger = metrics_logger

    def retrieve_memories_with_scores(self, query: str, k: int = 5):
        """Найти релевантную информацию из памяти со scores"""
        logger.info(f"🔍 RAG SEARCH: Starting retrieval | query='{query[:100]}' | k={k}")
        start_time = time.time()
        
        try:
            if hasattr(self.vectorstore, 'similarity_search_with_score'):
                logger.debug("Using similarity_search_with_score method")
                results = self.vectorstore.similarity_search_with_score(
                    query=query,
                    k=k
                )
                
                memories = []
                confidence_scores = []
                sources = set()
                
                for doc, score in results:
                    memories.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "relevance_score": float(score)
                    })
                    # Конвертируем distance в confidence score
                    # Chroma использует косинусное расстояние (0-2), где 0 = идеальное совпадение
                    # Конвертируем в confidence: чем меньше distance, тем выше confidence
                    distance = float(score)
                    # Нормализуем: distance 0 -> confidence 1.0, distance 2 -> confidence 0.0
                    confidence = max(0.0, min(1.0, 1.0 - (distance / 2.0)))
                    confidence_scores.append(confidence)
                    if doc.metadata.get("category"):
                        sources.add(doc.metadata["category"])
                
                retrieval_latency = time.time() - start_time
                
                logger.info(f"✅ RAG SEARCH: Found {len(memories)} results | latency={retrieval_latency:.3f}s | "
                          f"avg_confidence={sum(confidence_scores)/len(confidence_scores) if confidence_scores else 0:.3f} | "
                          f"source_diversity={len(sources)}")
                
                # Логируем каждый извлеченный чанк полностью
                for i, (mem, conf) in enumerate(zip(memories, confidence_scores), 1):
                    logger.info(f"\n📄 RETRIEVED CHUNK {i} FROM RAG:")
                    logger.info(f"  Confidence Score: {conf:.4f}")
                    logger.info(f"  Distance Score: {mem['relevance_score']:.4f}")
                    logger.info(f"  Category: {mem['metadata'].get('category', 'N/A')}")
                    logger.info(f"  Timestamp: {mem['metadata'].get('timestamp', 'N/A')}")
                    logger.info(f"  Content length: {len(mem['content'])}")
                    logger.info(f"  Full Content:")
                    logger.info(f"  {mem['content']}")
                    logger.info("-"*80)
                
                # ✅ ВАЖНО: Логируем RAG метрики ВСЕГДА, даже если результатов нет
                if self.metrics_logger:
                    # Убеждаемся, что есть хотя бы один confidence score для логирования
                    if not confidence_scores:
                        confidence_scores = [0.0]  # Минимум один score для пустого результата
                    
                    self.metrics_logger.log_rag_metrics(
                        retrieval_confidence_scores=confidence_scores,
                        num_chunks_retrieved=len(memories),
                        source_diversity=len(sources),
                        retrieval_latency=retrieval_latency,
                        metadata={"query": query[:100], "k": k, "found_results": len(memories) > 0, "method": "similarity_search_with_score"}
                    )
                else:
                    logger.warning("metrics_logger is None - RAG metrics will not be logged!")
                
                return memories
            else:
                # ✅ Если нет similarity_search_with_score, используем обычный поиск, но тоже логируем
                logger.debug("Using similarity_search fallback method")
                results = self.vectorstore.similarity_search(
                    query=query,
                    k=k
                )
                
                memories = []
                sources = set()
                
                for doc in results:
                    memories.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "relevance_score": 1.0  # Нет score, используем дефолтное значение
                    })
                    if doc.metadata.get("category"):
                        sources.add(doc.metadata["category"])
                
                retrieval_latency = time.time() - start_time
                
                logger.info(f"✅ RAG SEARCH: Found {len(memories)} results (fallback method) | latency={retrieval_latency:.3f}s | "
                          f"source_diversity={len(sources)}")
                
                # Логируем каждый извлеченный чанк полностью
                for i, mem in enumerate(memories, 1):
                    logger.info(f"\n📄 RETRIEVED CHUNK {i} FROM RAG (fallback):")
                    logger.info(f"  Category: {mem['metadata'].get('category', 'N/A')}")
                    logger.info(f"  Timestamp: {mem['metadata'].get('timestamp', 'N/A')}")
                    logger.info(f"  Content length: {len(mem['content'])}")
                    logger.info(f"  Full Content:")
                    logger.info(f"  {mem['content']}")
                    logger.info("-"*80)
                
                # Логируем даже для обычного поиска
                if self.metrics_logger:
                    # Используем дефолтный confidence score 0.5 для результатов без score
                    confidence_scores = [0.5] * len(memories) if memories else [0.0]
                    
                    self.metrics_logger.log_rag_metrics(
                        retrieval_confidence_scores=confidence_scores,
                        num_chunks_retrieved=len(memories),
                        source_diversity=len(sources),
                        retrieval_latency=retrieval_latency,
                        metadata={"query": query[:100], "k": k, "method": "similarity_search", "found_results": len(memories) > 0}
                    )
                else:
                    logger.warning("metrics_logger is None - RAG metrics will not be logged!")
                
                return memories
                
        except Exception as e:
            retrieval_latency = time.time() - start_time
            if self.metrics_logger:
                self.metrics_logger.log_rag_metrics(
                    retrieval_confidence_scores=[],
                    num_chunks_retrieved=0,
                    source_diversity=0,
                    retrieval_latency=retrieval_latency,
                    metadata={"error": str(e), "error_type": type(e).__name__, "query": query[:100], "k": k}
                )
            else:
                print("⚠️ WARNING: metrics_logger is None when trying to log error metrics!")
            print(f"Ошибка при поиске в памяти: {e}")
            return []
            
    def save_memory(self, content: str, metadata: dict = None):
        """
        Сохранить информацию в память
        
        Args:
            content: Текст для сохранения
            metadata: Дополнительные метаданные
        
        Returns:
            str: Сообщение о результате сохранения
        """
        category = metadata.get("category", "unknown") if metadata else "unknown"
        logger.info(f"💾 SAVING MEMORY: category={category} | content_length={len(content)}")
        
        timestamp = datetime.now().isoformat()
        
        default_metadata = {
            "timestamp": timestamp,
            "type": "memory"
        }
        if metadata:
            default_metadata.update(metadata)
        
        # ✅ Создаем Document согласно документации
        doc = Document(page_content=content, metadata=default_metadata)
        
        # ✅ Разбиваем на чанки
        documents = self.text_splitter.split_documents([doc])
        logger.info(f"📦 Split into {len(documents)} chunks")
        
        # Логируем каждый чанк, который будет сохранен
        for i, chunk_doc in enumerate(documents, 1):
            logger.info(f"\n💾 CHUNK {i} TO SAVE:")
            logger.info(f"  Category: {chunk_doc.metadata.get('category', 'N/A')}")
            logger.info(f"  Timestamp: {chunk_doc.metadata.get('timestamp', 'N/A')}")
            logger.info(f"  Content length: {len(chunk_doc.page_content)}")
            logger.info(f"  Content:")
            logger.info(f"  {chunk_doc.page_content}")
            logger.info("-"*80)
        
        # ✅ Генерируем уникальные ID для каждого документа
        # Согласно документации: add_documents(documents, ids)
        document_ids = [str(uuid.uuid4()) for _ in documents]
        
        # ✅ Добавляем документы согласно документации LangChain
        # docs: vector_store.add_documents(documents=[doc1, doc2], ids=["id1", "id2"])
        try:
            self.vectorstore.add_documents(
                documents=documents,
                ids=document_ids
            )
            # Chroma автоматически сохраняет при использовании persist_directory
            result = f"✅ Сохранено {len(documents)} фрагментов информации в память"
            logger.info(f"✅ MEMORY SAVED: {len(documents)} chunks successfully stored | category={category}")
            return result
        except Exception as e:
            logger.error(f"❌ MEMORY SAVE ERROR: {type(e).__name__} | error={str(e)}", exc_info=True)
            return f"❌ Ошибка при сохранении: {str(e)}"
    
    def retrieve_memories(self, query: str, k: int = 5):
        """
        Найти релевантную информацию из памяти
        
        Args:
            query: Поисковый запрос
            k: Количество результатов для возврата
        
        Returns:
            list: Список словарей с информацией о найденных воспоминаниях
        """
        try:
            # ✅ Используем similarity_search согласно документации
            # docs: similar_docs = vector_store.similarity_search("your query here", k=3)
            results = self.vectorstore.similarity_search(
                query=query,
                k=k
            )
            
            # Если нужны также scores, можно использовать similarity_search_with_score
            # Но для простоты используем базовый метод
            
            memories = []
            for doc in results:
                memories.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": 1.0  # similarity_search не возвращает score
                })
            
            return memories
            
        except Exception as e:
            # Если БД пустая или произошла ошибка, возвращаем пустой список
            print(f"Ошибка при поиске в памяти: {e}")
            return []
    
    def delete_memory(self, ids: list):
        """
        Удалить воспоминания по ID
        
        Args:
            ids: Список ID для удаления
        
        Returns:
            bool: True если успешно удалено
        """
        try:
            # ✅ Согласно документации: vector_store.delete(ids=["id1"])
            self.vectorstore.delete(ids=ids)
            return True
        except Exception as e:
            print(f"Ошибка при удалении: {e}")
            return False
    
    def get_all_memories(self, limit: int = 100):
        """
        Получить все сохраненные воспоминания
        
        Args:
            limit: Максимальное количество результатов
        
        Returns:
            dict: Словарь с ids, documents, metadatas
        """
        try:
            # Получаем все документы из коллекции
            all_docs = self.vectorstore.get(limit=limit)
            return all_docs
        except Exception as e:
            print(f"Ошибка при получении всех воспоминаний: {e}")
            return {"ids": [], "documents": [], "metadatas": []}
    
    def clear_all_memories(self):
        """
        Очистить всю память
        
        Returns:
            bool: True если успешно очищено
        """
        try:
            all_ids = self.get_all_memories()["ids"]
            if all_ids:
                self.delete_memory(all_ids)
            return True
        except Exception as e:
            print(f"Ошибка при очистке памяти: {e}")
            return False