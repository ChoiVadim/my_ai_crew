# AI Agent Study - Personal AI Agent with Memory

A project for learning LangChain and creating an AI agent that automates memory and work processes.

## 🎯 Project Goal

Create an intelligent AI agent that:
- **Automates memory**: Saves important information and retrieves it when needed
- **Manages work processes**: Helps organize information and tasks
- **Has long-term memory**: Uses vector database (Chroma) for information storage
- **Has short-term memory**: Remembers the context of the current conversation
- **Tracks metrics**: Logs all actions for analysis and improvement

## 📁 Project Structure

```
AIAgentStudy/
├── config/                    # Configuration
│   └── settings.py            # Application settings (API keys, parameters)
├── src/
│   ├── agents/                # Agents
│   │   └── memory_agent.py   # Main agent with memory
│   ├── memory/                # Memory system
│   │   ├── memory_store.py   # Long-term memory (Chroma DB)
│   │   └── short_term_memory.py  # Short-term memory (session)
│   ├── tools/                 # Agent tools
│   │   └── memory_tools.py   # Tools for memory operations
│   ├── prompts/               # Prompts
│   │   └── memory_agent_prompt.py  # System prompt for agent
│   └── utils/                 # Utilities
│       ├── logger.py          # Logging system
│       └── metrics_logger.py   # Metrics logging
├── data/
│   ├── logs/                  # Application logs
│   ├── memory/                # Chroma database
│   └── metrics/                # Metrics (JSON/JSONL)
├── notebooks/                  # Jupyter notebooks
│   └── model_testing.ipynb   # Model evaluation notebook
├── tests/                      # Tests
├── main.py                     # Entry point
└── pyproject.toml             # Dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### 2. Environment Setup

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o
OPENAI_TEMPERATURE=0.7

# Optional
MEMORY_DIR=./data/memory
MEMORY_CHUNK_SIZE=1000
MEMORY_CHUNK_OVERLAP=200
```

### 3. Run

```bash
python main.py
```

## 🧠 Architecture

### System Components

#### 1. **MemoryAgent** (`src/agents/memory_agent.py`)
Main agent that:
- Processes user messages
- Uses tools for memory operations
- Manages short-term memory (conversation context)
- Logs all actions

#### 2. **MemoryStore** (`src/memory/memory_store.py`)
Long-term memory based on Chroma:
- Saves information to vector database
- Performs semantic search
- Splits text into chunks for efficient storage
- Logs all RAG operations

#### 3. **ShortTermMemory** (`src/memory/short_term_memory.py`)
Short-term memory for current session:
- Stores last N messages
- Maintains conversation context
- Automatically trims old messages

#### 4. **Memory Tools** (`src/tools/memory_tools.py`)
Tools for the agent:
- `save_to_memory`: Save information
- `search_memory`: Search in long-term memory
- `remember_context`: Save session context

#### 5. **MetricsLogger** (`src/utils/metrics_logger.py`)
Metrics tracking system:
- **PROMPTS**: Quality scores, format compliance, refusal rates, response length
- **RAG**: Confidence scores, chunks retrieved, source diversity, latency
- **AGENTS**: Task completion, steps, tool success rates, errors, cost
- **SYSTEM**: End-to-end success, latency, cost, error rate, uptime

#### 6. **Logger** (`src/utils/logger.py`)
Logging system:
- Logs to file (`data/logs/agent_YYYYMMDD.log`)
- Logs to console
- Detailed logging of all operations

## 📊 Metrics

The system tracks metrics across 4 categories:

### PROMPTS
- Response quality scores
- Format compliance rate
- Refusal rates
- Average response length

### RAG
- Retrieval confidence scores
- Number of chunks retrieved
- Source diversity
- Retrieval latency

### AGENTS
- Task completion rate
- Average steps to completion
- Tool success rates
- Error types
- Cost per task

### SYSTEM
- End-to-end task success
- User satisfaction (optional)
- Latency
- Cost per request
- Error rate
- Uptime

View metrics: type `metrics` command in console.

## 🔧 Usage

### Main Commands

- **Regular chat**: Just type messages to the agent
- **`history`**: Show current conversation history
- **`clear`**: Clear short-term memory
- **`metrics`**: Show metrics summary
- **`exit`**: Exit application

### Usage Examples

```
You: My name is Vadim, I'm working on AI Agent Study project
Agent: [Will save information to memory]

You: What do you know about me?
Agent: [Will find information in memory and respond]

You: Remember what I told you about my project?
Agent: [Will use short-term memory for context]
```

## 📝 Logging

All actions are logged to:
- **File**: `data/logs/agent_YYYYMMDD.log`
- **Console**: Real-time output

Logs include:
- Full user messages
- Full AI responses
- What exactly is saved to memory (all chunks)
- What exactly is retrieved from RAG (all chunks)
- Tool usage
- Tool results
- Errors with full tracebacks

## 🛠 Technologies

- **LangChain**: Framework for working with LLM
- **Chroma**: Vector database for long-term memory
- **OpenAI**: Language model (GPT-4o)
- **Python 3.12+**: Programming language

## 📈 Metrics and Monitoring

All metrics are saved to:
- `data/metrics/aggregated_metrics.json` - aggregated metrics
- `data/metrics/*_metrics.jsonl` - detailed metrics by category

Metrics are updated in real-time and saved on exit.

## 🔐 Security

- API keys stored in `.env` (not committed to git)
- Logs don't contain sensitive data
- `.env` added to `.gitignore`

## 📚 Documentation

- [Model Selection and Evaluation](docs/model_selection_and_evaluation.md) - Guide for model selection and evaluation
- [Model Testing Notebook](notebooks/model_testing.ipynb) - Jupyter notebook for testing different models

## 📚 Learned Concepts

During development, learned:
- ✅ AI agent project structure
- ✅ LangChain Agents and Tools
- ✅ Vector Stores (Chroma)
- ✅ Embeddings and RAG
- ✅ Memory Management (short-term and long-term)
- ✅ Metrics Tracking
- ✅ Logging Best Practices
- ✅ Configuration Management
- ✅ Model Selection and Evaluation

## 🚧 Development Roadmap

- [ ] Add more tools (calendar, tasks, files)
- [ ] Improve memory categorization
- [ ] Add timestamps and priorities
- [ ] Implement deletion of outdated information
- [ ] Add web interface
- [ ] Integration with external APIs

## 📄 License

Project created for learning and studying AI agents.

## 👤 Author

Vadim - Learning AI agent development with LangChain
