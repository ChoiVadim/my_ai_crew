from dotenv import load_dotenv
import os
from pathlib import Path

# Определяем путь к .env файлу (в корне проекта)
BASE_DIR = Path(__file__).parent.parent
ENV_FILE = BASE_DIR / ".env"

# Загружаем переменные окружения
load_dotenv(ENV_FILE)

class Settings:
    """Настройки приложения"""
    
    # OpenAI настройки
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    
    # Anthropic настройки (если используете)
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    
    # Настройки памяти
    MEMORY_DIR: str = os.getenv("MEMORY_DIR", "./data/memory")
    MEMORY_CHUNK_SIZE: int = int(os.getenv("MEMORY_CHUNK_SIZE", "1000"))
    MEMORY_CHUNK_OVERLAP: int = int(os.getenv("MEMORY_CHUNK_OVERLAP", "200"))
    
    # Настройки агента
    AGENT_VERBOSE: bool = os.getenv("AGENT_VERBOSE", "true").lower() == "true"
    
    @classmethod
    def validate(cls):
        """Проверка обязательных настроек"""
        if not cls.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY не установлен! "
                "Убедитесь, что файл .env существует и содержит OPENAI_API_KEY"
            )
        return True
    
    @classmethod
    def get_openai_config(cls):
        """Получить конфигурацию для OpenAI"""
        return {
            "api_key": cls.OPENAI_API_KEY,
            "model": cls.OPENAI_MODEL,
            "temperature": cls.OPENAI_TEMPERATURE,
        }

# Создаем глобальный экземпляр настроек
settings = Settings()

# Валидируем при импорте
settings.validate()