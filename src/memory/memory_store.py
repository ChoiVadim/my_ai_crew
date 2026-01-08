# src/memory/memory_store.py
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config.settings import settings
import os
from datetime import datetime
import uuid

class MemoryStore:
    """Управление долгосрочной памятью агента"""
    
    def __init__(self, persist_directory=None):
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
    
    def save_memory(self, content: str, metadata: dict = None):
        """
        Сохранить информацию в память
        
        Args:
            content: Текст для сохранения
            metadata: Дополнительные метаданные
        
        Returns:
            str: Сообщение о результате сохранения
        """
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
            return f"✅ Сохранено {len(documents)} фрагментов информации в память"
        except Exception as e:
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
    
    def retrieve_memories_with_scores(self, query: str, k: int = 5):
        """
        Найти релевантную информацию из памяти со scores
        
        Args:
            query: Поисковый запрос
            k: Количество результатов для возврата
        
        Returns:
            list: Список словарей с информацией о найденных воспоминаниях и scores
        """
        try:
            # Используем метод с scores, если доступен
            if hasattr(self.vectorstore, 'similarity_search_with_score'):
                results = self.vectorstore.similarity_search_with_score(
                    query=query,
                    k=k
                )
                
                memories = []
                for doc, score in results:
                    memories.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "relevance_score": float(score)
                    })
                
                return memories
            else:
                # Fallback на обычный поиск
                return self.retrieve_memories(query, k)
                
        except Exception as e:
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