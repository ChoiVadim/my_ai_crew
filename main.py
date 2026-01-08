# main.py
from config.settings import settings
from src.agents.memory_agent import MemoryAgent
from src.utils.metrics_logger import MetricsLogger
from src.utils.logger import setup_logger

def main():
    # ✅ Инициализируем logger приложения
    logger = setup_logger("ai_agent")
    logger.info("="*60)
    logger.info("🚀 Starting AI Agent Application")
    logger.info("="*60)
    
    # ✅ Инициализируем logger метрик
    metrics_logger = MetricsLogger()
    agent = MemoryAgent(metrics_logger=metrics_logger)
    
    print("🤖 Агент с краткосрочной и долгосрочной памятью готов!")
    print("Команды: 'history' - показать историю, 'clear' - очистить память")
    print("         'metrics' - показать метрики, 'exit' - выход\n")
    logger.info("✅ Agent ready for user interaction")

    try:
        while True:
            user_input = input("Вы: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'выход']:
                logger.info("👋 User requested exit")
                # ✅ Сохраняем метрики перед выходом
                metrics_logger.save_aggregated_metrics()
                metrics_logger.print_metrics_summary()
                logger.info("✅ Metrics saved, shutting down")
                break
            
            if not user_input:
                continue
            
            # ✅ Специальные команды
            if user_input.lower() == 'history':
                logger.info("📜 User requested conversation history")
                print("\n📜 История разговора:")
                print(agent.get_conversation_history())
                print()
                continue
            
            if user_input.lower() == 'clear':
                logger.info("🧹 User requested memory clear")
                agent.clear_short_term_memory()
                print("✅ Краткосрочная память очищена\n")
                continue
            
            if user_input.lower() == 'metrics':
                logger.info("📊 User requested metrics summary")
                metrics_logger.print_metrics_summary()
                continue
            
            logger.info(f"💬 USER INPUT: '{user_input[:100]}'")
            result = agent.chat(user_input)
            print(f"Агент: {result}\n")
            
    except KeyboardInterrupt:
        logger.info("⚠️ Keyboard interrupt received")
        metrics_logger.save_aggregated_metrics()
        metrics_logger.print_metrics_summary()
        logger.info("✅ Metrics saved, shutting down")

if __name__ == "__main__":
    main()