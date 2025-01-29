# DeepSeek R1 RAG with Groq API

This repository demonstrates a Retrieval-Augmented Generation (RAG) implementation using DeepSeek R1 Distill LLaMA 70B model through Groq's high-performance inference API. The project includes both a command-line interface and a Streamlit web application for easy interaction.

## Features

- RAG implementation using DeepSeek R1 Distill LLaMA 70B
- High-performance inference using Groq API
- Hybrid retrieval system combining:
  - FAISS vector store for semantic search
  - BM25 for keyword-based retrieval
  - LLM-based reranking
- PDF document processing and chunking
- Interactive CLI and Streamlit web interface
- Conversation memory for context-aware responses
- Streaming responses for better user experience

## Architecture

- **Embedding Model**: `sentence-transformers/all-mpnet-base-v2` for document embedding
- **Vector Store**: FAISS for efficient similarity search
- **Text Processing**: RecursiveCharacterTextSplitter for optimal document chunking
- **Retrieval**: Ensemble retriever combining vector and keyword search
- **Interface**: Both CLI and Streamlit web application

## Prerequisites

- Python 3.8+
- Groq API Key (get free credits at [Groq Cloud](https://groq.com/))
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/lesteroliver911/deepseek-r1-groq-rag.git
cd deepseek-r1-groq-rag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Groq API key:
```bash
export GROQ_API_KEY='your-api-key-here'
```

## Usage

### Command Line Interface

Run the CLI version:
```bash
python main.py
```

The CLI will prompt you to:
1. Enter the path to your PDF file
2. Ask questions about the document's content

### Streamlit Web Interface

Run the Streamlit app:
```bash
streamlit run main_streamlit.py
```

The web interface provides:
- PDF upload functionality
- Interactive question-answering
- Real-time streaming responses
- Thought process visualization
- Clean, modern UI

## Configuration

The RAG system can be configured using the `RAGConfig` class:

- `chunk_size`: Size of text chunks (default: 2000)
- `chunk_overlap`: Overlap between chunks (default: 1000)
- `embedding_model`: Model for text embeddings
- `llm_model`: LLM model for completions
- `temperature`: Response creativity (0.0-1.0)
- `top_k_results`: Number of context chunks to retrieve
- `use_reranking`: Enable/disable LLM-based reranking
- `memory_window`: Number of conversation turns to remember

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)

## Acknowledgments

- Groq Cloud for the DeepSeek R1 Distill LLaMA 70B model
- LangChain for the RAG framework
- Streamlit for the web interface
