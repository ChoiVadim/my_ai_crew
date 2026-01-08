# src/utils/logger.py
"""
Настройка системы логирования для AI агента.
Логирует действия агента, использование инструментов, RAG поиски и т.д.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

def setup_logger(
    name: str = "ai_agent",
    log_dir: str = "./data/logs",
    level: int = logging.INFO,
    log_to_file: bool = True,
    log_to_console: bool = True
) -> logging.Logger:
    """
    Настроить и вернуть logger для приложения.
    
    Args:
        name: Имя логгера
        log_dir: Директория для логов
        level: Уровень логирования
        log_to_file: Логировать в файл
        log_to_console: Логировать в консоль
    
    Returns:
        logging.Logger: Настроенный logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Избегаем дублирования handlers
    if logger.handlers:
        return logger
    
    # Формат логов
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler для консоли
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Handler для файла
    if log_to_file:
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        
        # Файл для всех логов
        log_file = log_dir_path / f"agent_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Глобальный logger для приложения
_app_logger: Optional[logging.Logger] = None

def get_logger(name: str = "ai_agent") -> logging.Logger:
    """
    Получить глобальный logger приложения.
    
    Args:
        name: Имя логгера
    
    Returns:
        logging.Logger: Logger
    """
    global _app_logger
    if _app_logger is None:
        _app_logger = setup_logger(name)
    return _app_logger
