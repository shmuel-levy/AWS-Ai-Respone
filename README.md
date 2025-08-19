# 📚 Document Q&A System

A sophisticated **Retrieval-Augmented Generation (RAG)** system that enables intelligent document question-answering using Google's Gemini AI. This system automatically segments documents into semantically meaningful chunks and provides accurate, context-aware answers based solely on document content.

## 🚀 Features

### Core Functionality
- **Intelligent Document Processing**: Automatic segmentation of documents into 18 semantically meaningful chunks
- **Semantic Search**: Advanced vector-based search using ChromaDB and sentence transformers
- **AI-Powered Answers**: Integration with Google's Gemini AI for natural language responses
- **Multi-Format Support**: Handles PDF and text documents seamlessly
- **Real-time Processing**: Automatic document change detection and reprocessing

### Technical Features
- **RTL Language Support**: Full support for Hebrew and other right-to-left languages
- **Vector Database**: ChromaDB for efficient semantic search and retrieval
- **Embedding Models**: Sentence transformers for semantic understanding
- **Web Interface**: Modern Streamlit-based user interface
- **Configuration Management**: Environment-based configuration system

### Security & Performance
- **API Key Management**: Secure environment variable handling
- **Error Handling**: Comprehensive error handling and user feedback
- **Performance Optimization**: Efficient chunking and retrieval algorithms
- **Memory Management**: Optimized for large document processing

## 🏗️ Architecture

The system follows a modular architecture with clear separation of concerns:

```
Document Q&A System
├── 📄 Document Processor
│   ├── Content extraction (PDF/TXT)
│   ├── Semantic chunking (18 parts)
│   └── Keyword extraction
├── 🔍 RAG Engine
│   ├── Vector database (ChromaDB)
│   ├── Semantic search
│   └── Context building
├── 🤖 AI Client
│   ├── Gemini AI integration
│   └── Answer generation
└── 🌐 Web Interface
    ├── Streamlit UI
    ├── File upload
    └── Query interface
```

## 📋 Requirements

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: 1GB free space for dependencies and data

### Python Dependencies
```
google-generativeai>=0.3.2
streamlit>=1.28.1
python-dotenv>=1.0.0
chromadb>=0.4.18
sentence-transformers>=2.2.2
PyPDF2>=3.0.1
pdfplumber>=0.10.3
scikit-learn>=1.3.0
numpy>=1.24.3
pandas>=2.0.3
```

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd document-qa-system
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_CHUNKS=18
RETRIEVAL_TOP_K=3
CHROMA_PERSIST_DIRECTORY=./chroma_db
```

### 5. Get Gemini API Key
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add it to your `.env` file

## 🚀 Usage

### Running the Application

1. **Start the Web Interface**:
   ```bash
   streamlit run app.py
   ```

2. **Access the Application**:
   - Open your browser
   - Navigate to `http://localhost:8501`

3. **Upload a Document**:
   - Use the sidebar to upload a PDF or text file
   - Wait for processing to complete

4. **Ask Questions**:
   - Type your question in the query box
   - Get AI-generated answers based on document content

### Running Tests

```bash
python test_system.py
```

This will run comprehensive tests on all system components and generate a test report.

## 📖 Example Questions

Once you upload a document, you can ask questions like:

### General Questions
- "What is the main topic of this document?"
- "Can you summarize the key points?"
- "What are the important dates mentioned?"

### Technical Questions
- "What are the practical examples and implementation details?"
- "How is the architecture designed?"
- "What security measures are mentioned?"

### Specific Questions
- "What are the AWS services used in this project?"
- "How is the deployment process structured?"
- "What are the cost optimization strategies?"

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini AI API key | Required |
| `CHUNK_SIZE` | Size of document chunks | 1000 |
| `CHUNK_OVERLAP` | Overlap between chunks | 200 |
| `MAX_CHUNKS` | Maximum number of chunks | 18 |
| `RETRIEVAL_TOP_K` | Number of relevant chunks to retrieve | 3 |

### Advanced Configuration

You can modify the configuration in `config.py` for advanced settings:

```python
# Document processing
CHUNK_SIZE = 1500  # Larger chunks
MAX_CHUNKS = 24    # More chunks

# Retrieval settings
RETRIEVAL_TOP_K = 5  # More context
```

## 🧪 Testing

The system includes comprehensive testing:

```bash
# Run all tests
python test_system.py

# Test specific components
python -c "from document_processor import DocumentProcessor; print('Document processor works')"
python -c "from rag_engine import RAGEngine; print('RAG engine works')"
python -c "from gemini_client import GeminiClient; print('AI client works')"
```

## 📁 Project Structure

```
document-qa-system/
├── 📄 app.py                 # Main Streamlit application
├── 📄 config.py              # Configuration management
├── 📄 document_processor.py  # Document processing and chunking
├── 📄 rag_engine.py          # RAG system and vector search
├── 📄 gemini_client.py       # Gemini AI integration
├── 📄 test_system.py         # Comprehensive testing
├── 📄 requirements.txt       # Python dependencies
├── 📄 README.md              # This file
├── 📄 .env                   # Environment variables (create this)
├── 📄 .gitignore             # Git ignore rules
├── 📁 documents/             # Uploaded documents
├── 📁 data/                  # Processed chunks
└── 📁 chroma_db/             # Vector database
```

## 🔍 How It Works

### 1. Document Processing
- **Content Extraction**: Extracts text from PDF or text files
- **Semantic Chunking**: Splits document into 18 semantically meaningful chunks
- **Keyword Extraction**: Identifies important terms in each chunk

### 2. Vector Indexing
- **Embedding Generation**: Creates vector representations of chunks
- **Database Storage**: Stores vectors in ChromaDB for fast retrieval
- **Metadata Management**: Maintains chunk metadata and relationships

### 3. Query Processing
- **Semantic Search**: Finds most relevant chunks for user query
- **Context Building**: Combines relevant chunks into context
- **Answer Generation**: Uses Gemini AI to generate accurate answers

### 4. Response Delivery
- **Answer Formatting**: Presents answers in clear, structured format
- **Source Context**: Shows relevant document sections
- **RTL Support**: Properly handles Hebrew and other RTL languages

## 🐛 Troubleshooting

### Common Issues

1. **API Key Error**:
   - Ensure `GEMINI_API_KEY` is set in `.env` file
   - Verify the API key is valid and has sufficient quota

2. **Document Processing Fails**:
   - Check file format (PDF or TXT only)
   - Ensure file is not corrupted
   - Verify sufficient disk space

3. **Memory Issues**:
   - Reduce `CHUNK_SIZE` in configuration
   - Process smaller documents
   - Increase system RAM

4. **Performance Issues**:
   - Clear ChromaDB cache: `rm -rf chroma_db/`
   - Restart the application
   - Check system resources

### Getting Help

1. **Check Logs**: Look for error messages in the terminal
2. **Run Tests**: Execute `python test_system.py` to identify issues
3. **Verify Configuration**: Ensure all environment variables are set correctly

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Make your changes**
4. **Run tests**: `python test_system.py`
5. **Commit changes**: `git commit -m 'Add feature'`
6. **Push to branch**: `git push origin feature-name`
7. **Submit a pull request**

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Google Gemini AI** for providing the language model
- **ChromaDB** for vector database functionality
- **Streamlit** for the web interface framework
- **Sentence Transformers** for semantic embeddings

## 📞 Support

For support and questions:
- **Issues**: Create an issue in the repository
- **Documentation**: Check this README and inline code comments
- **Testing**: Run the test suite to verify system functionality

---

**Built with ❤️ for intelligent document processing and AI-powered question answering.**
