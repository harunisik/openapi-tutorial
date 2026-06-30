# AI Tutorial: LangChain & LangGraph Mastery

A comprehensive learning and experimentation repository exploring modern AI frameworks, with hands-on implementations of LangChain, LangGraph, and core Python engineering patterns. This project demonstrates proficiency in AI/LLM development, software architecture, and practical AI system design.

## 🎯 Project Overview

This repository documents my journey mastering contemporary AI frameworks and best practices. It contains:

- **Progressive Lessons**: 12+ structured LangChain lessons (prompts → agents → RAG)
- **Graph-Based AI**: Advanced LangGraph implementations with multi-agent coordination
- **RAG Systems**: End-to-end Retrieval-Augmented Generation with embeddings
- **Task Management**: Clean, production-grade Python module demonstrating software design principles
- **Real-world Integrations**: OpenAI API integration, embeddings, chunking strategies

## 🚀 Key Features

### LangChain Curriculum
- **Foundations** (L1-L4): ChatOpenAI, Runnables, Messages, Streaming
- **Intermediate** (L5-L9): Prompt engineering, Output parsers, Tools, Memory management
- **Advanced** (L10-L12): Agent construction, Semantic chunking, RAG pipelines

### LangGraph Implementations
- **Graph-Based Workflows**: Node composition with research, reasoning, and validation stages
- **Multi-Agent Systems**: Loop-aware supervisors for complex task delegation
- **Tool Integration**: Agentic systems with dynamic tool selection and error handling

### Production-Grade Modules
- **Task Management System**: Full-featured task service with storage layer, custom exceptions, and state validation
- **Document Processing**: RAG pipeline with configurable chunking and embeddings
- **Semantic Search**: Similarity-based retrieval using embedding models

## 📂 Project Structure

```
├── src/
│   ├── langchain/           # 12+ lessons (foundations to RAG)
│   │   ├── lesson1_langchain.py        # Prompt chains, basic invoke
│   │   ├── lesson2_runnables.py        # Runnable composition
│   │   ├── lesson3_batch.py            # Batch processing
│   │   ├── lesson4_streaming.py        # Streaming outputs
│   │   ├── lesson5_prompting.py        # Advanced prompts
│   │   ├── lesson6_runnable_map.py     # Parallel processing
│   │   ├── lesson7_*.py                # Output parsers (Pydantic, structured)
│   │   ├── lesson8_memory.py           # Conversation memory
│   │   ├── lesson9_tools.py            # Tool integration
│   │   ├── lesson10_agents.py          # Agent frameworks
│   │   ├── lesson11_*.py               # Embeddings, chunking
│   │   ├── lesson12_rag.py             # Full RAG pipeline
│   │   └── rag_demo/                   # RAG implementation (ingest, retrieve, query)
│   │
│   ├── langgraph/           # Advanced multi-agent systems
│   │   ├── lesson5_graph_node.py              # Research/reasoning/validation nodes
│   │   ├── lesson7_loop_aware_supervisor.py  # Multi-agent coordination
│   │   ├── lesson9_tool_usage.py             # Dynamic tool selection
│   │   └── retriever.py                      # Semantic search
│   │
│   ├── python/              # Core Python & software design
│   │   ├── taskmanager/     # Production task management system
│   │   │   ├── models.py    # Task dataclass
│   │   │   ├── service.py   # Business logic, exception handling
│   │   │   ├── storage.py   # Data persistence
│   │   │   └── __init__.py
│   │   └── main.py
│   │
│   └── config.py            # Configuration management
```

## 🛠️ Tech Stack

- **Language**: Python 3.9+
- **AI Frameworks**: LangChain, LangGraph
- **LLM**: OpenAI (GPT-4, GPT-4 mini, text-embedding-3-small)
- **Libraries**: Pydantic, dotenv
- **Concepts**: Prompt engineering, RAG, multi-agent systems, embeddings

## 💡 Key Learnings & Implementation Highlights

### 1. **Prompt Engineering Mastery**
- Structured prompt templates with role-based messaging
- Dynamic variable injection for flexible prompting
- Output parsing with Pydantic for type-safe responses

### 2. **Agentic AI Design**
- Research → Reasoning → Validation pipeline architecture
- Loop-aware supervisors for agent coordination
- Error recovery and composite reasoning patterns

### 3. **RAG (Retrieval-Augmented Generation)**
- Document chunking with configurable overlap
- Semantic similarity search using embeddings
- Multi-step ingestion and retrieval pipelines

### 4. **Software Engineering**
- Clean architecture with separation of concerns
- Custom exception hierarchy for domain-specific errors
- Decorator patterns for cross-cutting concerns
- Dataclass-based models for type safety

## 🔧 Getting Started

### Prerequisites
- Python 3.9 or higher
- OpenAI API key

### Installation

```bash
# Clone the repository
git clone https://github.com/harunisik/openapi-tutorial.git
cd ai-tutorial

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Running Examples

```bash
# Run LangChain lesson (e.g., lesson 1)
python -m src.langchain.lesson1_langchain

# Run LangGraph example
python -m src.langgraph.lesson9_tool_usage

# Run task manager demo
python -m src.python.main

# Run RAG pipeline
python -m src.langchain.rag_demo.main
```

## 📊 Configuration

Edit `.env` to customize:

```env
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-4.1-mini           # Model selection
EMBEDDING_MODEL_NAME=text-embedding-3-small
CHUNK_SIZE=500                       # Document chunk size
OVERLAP_SIZE=50                      # Chunk overlap
MEMORY_ENABLED=true                  # Memory for agents
SHOPAGENT_DEBUG=1                    # Debug mode
```

## 🎓 Learning Path

1. **Start**: Lesson 1-4 (foundations)
2. **Build**: Lesson 5-9 (intermediate)
3. **Master**: Lesson 10-12 (advanced RAG)
4. **Explore**: LangGraph examples (multi-agent systems)
5. **Polish**: Task management system (software design)

## 💼 Portfolio Value

This project demonstrates:

- ✅ **Deep LLM/AI Knowledge**: Full LangChain curriculum with progressive complexity
- ✅ **System Design**: Multi-agent architectures, graph-based workflows, RAG pipelines
- ✅ **Python Proficiency**: Clean code, design patterns, type safety, error handling
- ✅ **Production Readiness**: Configuration management, custom exceptions, testable components
- ✅ **Problem-Solving**: Real-world AI challenges (chunking, embeddings, memory, tool calling)

## 📝 Recent Commits

- Implement task management system with models, service, and storage components
- Lesson 9: Tool usage with composite reasoning and validation nodes
- Lesson 7: Loop-aware supervisor with multi-agent coordination
- Lesson 5: Graph node structure for research, reasoning, and validation

## 🤝 Contributing

This is a personal learning project, but insights and feedback are always welcome!

## 📄 License

MIT License - feel free to use this as a reference for your own learning journey.

---

**Built with** ❤️ while mastering modern AI frameworks. Feel free to explore, learn, and adapt!
