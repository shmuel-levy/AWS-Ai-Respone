"""
Streamlit Web Application for Document Q&A System
Main interface for users to ask questions about documents using AI-powered search
"""
import streamlit as st
import os
from pathlib import Path
from document_processor import DocumentProcessor
from rag_engine import RAGEngine
from gemini_client import GeminiClient
from config import Config


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
    try:
        processor = DocumentProcessor()
        chunks = processor.process_document(file_path)
        
        st.session_state.document_processor = processor
        st.session_state.document_loaded = True
        
        return chunks
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return None


def setup_retrieval_system(chunks):
    """
    Setup RAG engine with processed document chunks.
    
    Args:
        chunks: List of document chunks to index
        
    Returns:
        True if setup successful, False otherwise
    """
    try:
        rag_engine = RAGEngine()
        rag_engine.index_document_chunks(chunks)
        st.session_state.rag_engine = rag_engine
        return True
    except Exception as e:
        st.error(f"Error setting up retrieval system: {str(e)}")
        return False


def setup_ai_client():
    """
    Setup Gemini AI client for answer generation.
    
    Returns:
        True if setup successful, False otherwise
    """
    try:
        client = GeminiClient()
        st.session_state.gemini_client = client
        return True
    except Exception as e:
        st.error(f"Error setting up AI client: {str(e)}")
        return False


def generate_answer_for_query(query: str):
    """
    Generate answer for user query using RAG system.
    
    Args:
        query: User's question
        
    Returns:
        Tuple of (answer, context) or error message
    """
    if not st.session_state.rag_engine or not st.session_state.gemini_client:
        return "System not properly initialized.", ""
    
    try:
        # Get relevant context from document
        context = st.session_state.rag_engine.build_query_context(query)
        
        if not context:
            return "I apologize, I don't know how to answer this question.", ""
        
        # Generate answer using AI
        answer = st.session_state.gemini_client.generate_answer_from_context(query, context)
        
        return answer, context
    except Exception as e:
        return f"Error processing query: {str(e)}", ""


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
        
        # Load RTL CSS
        with open('styles/rtl.css', 'r', encoding='utf-8') as f:
            rtl_css = f.read()
        
        # Apply CSS
        st.markdown(f"""
            <style>
            {main_css}
            {rtl_css}
            </style>
        """, unsafe_allow_html=True)
        
    except FileNotFoundError as e:
        st.error(f"CSS file not found: {e}")
    except Exception as e:
        st.error(f"Error loading CSS files: {e}")


def render_document_upload_section():
    """Render the document upload section in the sidebar."""
    with st.sidebar:
        st.header("📄 Document Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF or text file",
            type=['pdf', 'txt']
        )
        
        if uploaded_file is not None:
            # Save uploaded file
            documents_dir = Path("documents")
            documents_dir.mkdir(exist_ok=True)
            
            file_path = documents_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"File uploaded: {uploaded_file.name}")
            
            # Process document
            with st.spinner("Processing document..."):
                chunks = process_uploaded_document(str(file_path))
                
                if chunks:
                    if setup_retrieval_system(chunks):
                        st.success(f"Document processed into {len(chunks)} chunks")
                        
                        if setup_ai_client():
                            st.success("AI system ready!")
                        else:
                            st.warning("Gemini AI not available - check API key")
                    else:
                        st.error("Failed to setup retrieval system")


def render_query_interface():
    """Render the main query interface."""
    if st.session_state.document_loaded:
        st.header("❓ Ask Questions")
        
        # Query input
        query = st.text_input(
            "Enter your question:",
            placeholder="What would you like to know about the document?"
        )
        
        if st.button("Ask Question", type="primary"):
            if query:
                with st.spinner("Generating answer..."):
                    answer, context = generate_answer_for_query(query)
                
                # Display answer
                st.subheader("🤖 Answer")
                
                # Check if answer contains Hebrew characters and display appropriately
                if any('\u0590' <= char <= '\u05FF' for char in answer):
                    st.markdown(f'<div class="mixed-text">{answer}</div>', unsafe_allow_html=True)
                else:
                    st.write(answer)
                
                # Display context (expandable) with RTL support for Hebrew
                with st.expander("📖 View Source Context"):
                    # Check if context contains Hebrew characters
                    if any('\u0590' <= char <= '\u05FF' for char in context):
                        st.markdown(f'<div class="mixed-text">{context}</div>', unsafe_allow_html=True)
                    else:
                        st.text(context)
            else:
                st.warning("Please enter a question")
    else:
        render_welcome_screen()


def render_welcome_screen():
    """Render the welcome screen when no document is loaded."""
    st.info("👆 Please upload a document to get started")
    
    # Show sample questions
    st.subheader("💡 Example Questions")
    st.markdown("""
    Once you upload a document, you can ask questions like:
    - What is the main topic of this document?
    - Can you summarize the key points?
    - What are the important dates mentioned?
    - Who are the main people discussed?
    - What are the practical examples and implementation details?
    """)


def main():
    """Main application function."""
    # Setup page configuration
    setup_page_configuration()
    
    # Display main header
    st.markdown('<h1 class="main-header">📚 Document Q&A System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions about your documents using AI-powered search</p>', unsafe_allow_html=True)
    
    # Initialize application state
    initialize_application_state()
    
    # Render document upload section
    render_document_upload_section()
    
    # Render main query interface
    render_query_interface()


if __name__ == "__main__":
    main()
