# Document Q&A System with RAG and Gemini AI

A web-based question-answering system that uses Retrieval-Augmented Generation (RAG) with Google's Gemini AI to provide accurate answers based on document content.

## Features

- **RAG Implementation**: Automatic document segmentation into 18 parts
- **Smart Retrieval**: Relevant section identification for each query
- **Gemini AI Integration**: Powered by Google's latest AI model
- **Web Interface**: Clean Streamlit-based UI
- **Context Highlighting**: Emphasizes relevant document sections
- **Automatic Updates**: Dynamic document processing when content changes

## Project Structure

```
├── app.py                 # Main Streamlit application
├── document_processor.py  # Document segmentation and processing
├── rag_engine.py         # RAG implementation with vector search
├── gemini_client.py      # Gemini AI integration
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── data/                 # Document storage
└── README.md            # Project documentation
```

## Setup Instructions

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables:
   ```bash
   cp .env.example .env
   # Add your Gemini API key to .env
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## How It Works

1. **Document Processing**: Documents are automatically segmented into 18 parts using semantic analysis
2. **Query Processing**: User questions are analyzed to identify relevant document sections
3. **Retrieval**: Only 1-3 most relevant sections are retrieved for each query
4. **Generation**: Gemini AI generates answers based on retrieved context
5. **Response**: Answers are displayed with highlighted relevant sections

## Technology Stack

- **Python 3.8+**
- **Streamlit** - Web interface
- **Google Gemini AI** - Language model
- **ChromaDB** - Vector database
- **Sentence Transformers** - Text embeddings
- **LangChain** - RAG framework
