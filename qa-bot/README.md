# Document Question Answering Bot - Demonstration

## 🚨 IMPORTANT: DEMONSTRATION TOOL ONLY

This is a **proof-of-concept demonstration** of how a "Talk to Your Documents" tool can be built using RAG (Retrieval-Augmented Generation) architecture. This tool demonstrates the technical implementation of document processing, vector embeddings, semantic search, and retrieval pipelines.

## ⚠️ Critical Limitations

This tool is **NOT designed for production use**. Key limitations include:

- **Minimal AI Model**: Uses DistilGPT-2 (82M parameters) - extremely small compared to modern LLMs like GPT-4 or Claude (70B+ parameters)
- **Document Retrieval Only**: Shows relevant document sections based on your query, but does NOT generate coherent answers
- **Demonstration Purpose**: This showcases the RAG retrieval pipeline, not advanced AI capabilities
- **Resource Constraints**: Designed to run on minimal hardware without GPU requirements for educational purposes
- **No Answer Generation**: The AI model is too small to follow instructions, so it only demonstrates semantic search and document retrieval

## ✅ What This Tool Demonstrates

### Technical Architecture
- **Document Processing**: Loading and parsing PDF/text files
- **Text Chunking**: Splitting documents into manageable pieces
- **Vector Storage**: Creating embeddings and storing in Chroma vector database
- **Semantic Search**: Finding relevant document chunks for queries
- **RAG Pipeline**: Combining retrieval with generation using LangChain
- **Web Interface**: Gradio-based user interface

### RAG Implementation
- Document → Chunks → Embeddings → Vector Store
- Query → Embedding → Similarity Search → Context Retrieval
- Context + Query → LLM → Response Generation

## 🚀 How to Run

```bash
# Navigate to the qa-bot directory
cd qa-bot

# Activate virtual environment
source venv/bin/activate

# Run the application
python qabot.py
```

The application will launch at `http://127.0.0.1:7860`

## 📁 File Structure

```
qa-bot/
├── qabot.py              # Main demonstration tool
├── README.md             # This file (you are here)
└── venv/                 # Virtual environment
```

## 🔧 Technical Details

### Models Used
- **LLM**: DistilGPT-2 (82M parameters) - lightweight, stable
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (22MB)
- **Vector Store**: Chroma (in-memory)
- **Framework**: LangChain 0.3+ with LCEL chains

### Dependencies
- `langchain` - RAG framework
- `transformers` - HuggingFace models
- `chromadb` - Vector database
- `gradio` - Web interface
- `torch` - PyTorch backend

## 🎯 For Production Use

To build a production-quality document QA system, you would need:

### Better Models
- **LLM**: GPT-4, Claude, or Llama 2/3 (70B+ parameters)
- **Embeddings**: text-embedding-ada-002 or similar high-quality models
- **Infrastructure**: GPU servers, API access, or cloud services

### Enhanced Features
- Better document parsing (handling tables, images, etc.)
- Improved chunking strategies
- Advanced retrieval methods (hybrid search, reranking)
- Response quality filtering and validation
- User feedback and model fine-tuning

## 📚 Educational Value

This tool demonstrates technical skills in:
- **RAG Architecture**: Understanding retrieval-augmented generation systems
- **Document Processing**: Working with PDF/text parsing and chunking
- **Vector Databases**: Implementing semantic search with embeddings
- **LangChain Framework**: Building production-ready RAG pipelines
- **AI Engineering**: Balancing model capabilities with resource constraints
- **Web Interface Development**: Creating user-friendly interfaces with Gradio

## 🎓 Course Context

This project demonstrates skills learned in the [IBM AI Engineering Professional Certificate](https://www.coursera.org/professional-certificates/ai-engineer) on Coursera. The RAG architecture and document processing techniques are based on course materials, with significant modifications and improvements made to create a working demonstration tool.

## 🙏 Acknowledgements

- This project was developed as part of the [IBM AI Engineering Professional Certificate](https://www.coursera.org/professional-certificates/ai-engineer) on Coursera
- RAG architecture concepts and document processing techniques are based on course materials
- Significant modifications and improvements were made to create a working demonstration tool
- Special thanks to IBM instructors and the Coursera team for providing excellent course materials

## 📄 License & Disclaimer

This project is provided for **educational purposes only**. The code is shared to demonstrate AI engineering skills in RAG implementation, document processing, and vector database usage.

### ⚠️ Important Notice

- **Course materials** are provided by Coursera/IBM
- **This implementation** includes significant modifications beyond the original course materials
- **Not intended for production use** - Educational demonstration only
- **DistilGPT-2 model** is used for demonstration purposes due to resource constraints

---

**Remember**: This tool demonstrates *how* to build a document retrieval system using RAG architecture, not a production-quality document QA system. The intentional limitations showcase realistic constraints when building AI systems with limited resources.
