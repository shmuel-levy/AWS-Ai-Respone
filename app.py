"""
Streamlit Web Application for Document Q&A System
Main interface for users to ask questions about documents using AI-powered search
"""
import streamlit as st
import os
from pathlib import Path

# Try to import components with fallback handling
try:
    from document_processor import DocumentProcessor
    from rag_engine import RAGEngine
    from gemini_client import GeminiClient
    from config import Config
    CHROMA_AVAILABLE = True
except ImportError as e:
    CHROMA_AVAILABLE = False
except Exception as e:
    CHROMA_AVAILABLE = False

def initialize_application_state():
    """Initialize Streamlit session state variables for the application."""
    if 'document_loaded' not in st.session_state:
        st.session_state.document_loaded = False
    if 'document_processor' not in st.session_state:
        st.session_state.document_processor = None
    if 'rag_engine' not in st.session_state:
        st.session_state.rag_engine = None
    if 'gemini_client' not in st.session_state:
        st.session_state.gemini_client = None

def process_uploaded_document(file_path: str):
    """
    Process uploaded document and create semantic chunks.

    Args:
        file_path: Path to the uploaded document

    Returns:
        List of document chunks with metadata, or None if processing fails
    """
    if not CHROMA_AVAILABLE:
        st.error("❌ Document processing is not available. Please check your installation.")
        return None
        
    try:
        processor = DocumentProcessor()
        chunks = processor.process_document(file_path)

        st.session_state.document_processor = processor
        st.session_state.document_loaded = True

        return chunks
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return None

def setup_page_configuration():
    """Configure Streamlit page settings and load external CSS files."""
    st.set_page_config(
        page_title="Document Q&A System",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Load external CSS files
    load_css_files()

def load_css_files():
    """Load external CSS files for styling."""
    try:
        # Load main CSS
        with open('styles/main.css', 'r', encoding='utf-8') as f:
            main_css = f.read()
        
        # Apply CSS
        st.markdown(f"""
            <style>
            {main_css}
            </style>
        """, unsafe_allow_html=True)
        
    except FileNotFoundError as e:
        st.error(f"CSS file not found: {e}")
    except Exception as e:
        st.error(f"Error loading CSS files: {e}")

def render_header():
    """Render the main header section."""
    st.markdown('<h1 class="main-header">📚 Document Q&A System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions about your documents using AI-powered search</p>', unsafe_allow_html=True)

def render_upload_section():
    """Render the document upload section."""
    st.markdown("### 📄 Upload Document")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF or TXT file",
        type=['pdf', 'txt'],
        help="Upload a document to start asking questions"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_file_path = f"temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process document
        with st.spinner("Processing document..."):
            chunks = process_uploaded_document(temp_file_path)
        
        if chunks:
            st.success(f"✅ Document processed successfully!")
            st.info(f"Created {len(chunks)} semantic chunks")
            
            # Clean up temp file
            try:
                os.remove(temp_file_path)
            except:
                pass
        else:
            st.error("❌ Error processing document")
            
            # Clean up temp file
            try:
                os.remove(temp_file_path)
            except:
                pass

def render_question_section():
    """Render the question input section."""
    st.markdown("### ❓ Ask a Question")
    
    if not st.session_state.document_loaded:
        st.info("📝 Please upload a document first")
        return
    
    question = st.text_input(
        "Type your question:",
        placeholder="Example: What is the main topic of this document?",
        help="Ask any question about the uploaded document"
    )
    
    if st.button("🔍 Search Answer", type="primary"):
        if question.strip():
            process_question(question)
        else:
            st.warning("⚠️ Please type a question")

def process_question(question: str):
    """Process user question and generate answer."""
    if not st.session_state.document_loaded:
        st.error("❌ No document available")
        return
    
    try:
        # Initialize components if needed
        if not st.session_state.rag_engine:
            st.session_state.rag_engine = RAGEngine()
            st.session_state.rag_engine.initialize_database()
            
            # Add chunks to database
            chunks = st.session_state.document_processor.document_chunks
            st.session_state.rag_engine.index_document_chunks(chunks)
        
        if not st.session_state.gemini_client:
            st.session_state.gemini_client = GeminiClient()
        
        # Find relevant chunks
        relevant_chunks = st.session_state.rag_engine.find_relevant_chunks(question)
        
        if not relevant_chunks:
            st.warning("⚠️ No relevant chunks found")
            return
        
        # Build context
        context = "\n\n".join([chunk['content'] for chunk in relevant_chunks])
        
        # Generate answer
        with st.spinner("Searching for answer..."):
            answer = st.session_state.gemini_client.generate_answer_from_context(question, context)
        
        # Display results
        display_answer(question, answer, relevant_chunks)
        
    except Exception as e:
        st.error(f"❌ Error processing question: {str(e)}")

def display_answer(question: str, answer: str, relevant_chunks: list):
    """Display the generated answer and relevant context."""
    st.markdown("### 💡 Answer")
    st.markdown(answer)
    
    # Display relevant context
    st.markdown("### 📖 Relevant Context")
    
    for i, chunk in enumerate(relevant_chunks):
        with st.expander(f"Chunk {i+1} ({chunk.get('word_count', 'N/A')} words)"):
            chunk_content = chunk.get('content', '')
            st.markdown(chunk_content)

def render_sidebar():
    """Render the sidebar with additional information."""
    with st.sidebar:
        st.markdown("### ℹ️ Information")
        
        if CHROMA_AVAILABLE:
            st.success("🚀 Full Mode Active")
            st.info("All features available")
        else:
            st.warning("⚠️ Limited Mode")
            st.info("Some features may be limited")
        
        st.markdown("---")
        st.markdown("### 🔧 Features")
        st.markdown("- 📄 PDF and TXT support")
        st.markdown("- 🔍 Semantic search")
        st.markdown("- 🤖 AI-powered answers")
        st.markdown("- 🌐 Multi-language support")
        
        st.markdown("---")
        st.markdown("### 📚 How it works")
        st.markdown("1. Upload document")
        st.markdown("2. Ask questions")
        st.markdown("3. Get AI-generated answers")
        
        if st.button("🔄 Reset Session"):
            st.session_state.clear()
            st.rerun()

def main():
    """Main application function."""
    setup_page_configuration()
    initialize_application_state()
    
    render_header()
    
    # Create responsive columns for mobile
    col1, col2 = st.columns([3, 1])
    
    with col1:
        render_upload_section()
        render_question_section()
    
    with col2:
        # Mobile-friendly sidebar content
        if st.button("ℹ️ Info", help="Click for information"):
            st.sidebar.empty()
            render_sidebar()
    
    # Always show sidebar on desktop, conditional on mobile
    if st.session_state.get('show_sidebar', True):
        render_sidebar()

if __name__ == "__main__":
    main()
