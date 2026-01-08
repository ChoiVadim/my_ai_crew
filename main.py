# main.py
from config.settings import settings
from src.agents.memory_agent import MemoryAgent

def main():
    agent = MemoryAgent()
    
    print("🤖 Агент с краткосрочной и долгосрочной памятью готов!")
    print("Команды: 'history' - показать историю, 'clear' - очистить краткосрочную память\n")

    while True:
        user_input = input("Вы: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'выход']:
            break
        
        if not user_input:
            continue
        
        # ✅ Специальные команды
        if user_input.lower() == 'history':
            print("\n📜 История разговора:")
            print(agent.get_conversation_history())
            print()
            continue
        
        if user_input.lower() == 'clear':
            agent.clear_short_term_memory()
            print("✅ Краткосрочная память очищена\n")
            continue
        
        print("\n🤔 Агент думает...\n")
        result = agent.chat(user_input)
        
        # Обработка результата
        if isinstance(result, dict):
            messages = result.get("messages", [])
            if messages:
                last_message = messages[-1]
                if hasattr(last_message, 'content'):
                    print(f"Агент: {last_message.content}")
                else:
                    print(f"Агент: {str(last_message)}")
            else:
                output = result.get("output", "")
                print(f"Агент: {output if output else result}")
        else:
            print(f"Агент: {result}")
        
        print()

if __name__ == "__main__":
    main()