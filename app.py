"""
Streamlit Web Application for Document Q&A System
Main interface for users to ask questions about documents
"""
import streamlit as st
import os
from pathlib import Path
from document_processor import DocumentProcessor
from rag_engine import RAGEngine
from gemini_client import GeminiClient
from config import Config


def initialize_session_state():
    """Initialize session state variables"""
    if 'document_loaded' not in st.session_state:
        st.session_state.document_loaded = False
    if 'processor' not in st.session_state:
        st.session_state.processor = None
    if 'rag_engine' not in st.session_state:
        st.session_state.rag_engine = None
    if 'gemini_client' not in st.session_state:
        st.session_state.gemini_client = None


def load_document(file_path: str):
    """Load and process document"""
    try:
        processor = DocumentProcessor()
        content = processor.load_document(file_path)
        chunks = processor.segment_document(content)
        
        st.session_state.processor = processor
        st.session_state.document_loaded = True
        
        return chunks
    except Exception as e:
        st.error(f"Error loading document: {str(e)}")
        return None


def setup_rag_engine(chunks):
    """Setup RAG engine with document chunks"""
    try:
        rag_engine = RAGEngine()
        rag_engine.add_chunks(chunks)
        st.session_state.rag_engine = rag_engine
        return True
    except Exception as e:
        st.error(f"Error setting up RAG engine: {str(e)}")
        return False


def setup_gemini_client():
    """Setup Gemini AI client"""
    try:
        client = GeminiClient()
        st.session_state.gemini_client = client
        return True
    except Exception as e:
        st.error(f"Error setting up Gemini client: {str(e)}")
        return False


def process_query(query: str):
    """Process user query and generate answer"""
    if not st.session_state.rag_engine or not st.session_state.gemini_client:
        return "System not properly initialized."
    
    try:
        # Get relevant context
        context = st.session_state.rag_engine.get_context_for_query(query)
        
        if not context:
            return "I apologize, I don't know how to answer this question."
        
        # Generate answer
        answer = st.session_state.gemini_client.generate_answer(query, context)
        
        return answer, context
    except Exception as e:
        return f"Error processing query: {str(e)}", ""


def main():
    """Main application function"""
    st.set_page_config(
        page_title="Document Q&A System",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Document Q&A System")
    st.markdown("Ask questions about your documents using AI-powered search")
    
    initialize_session_state()
    
    # Sidebar for document upload
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
                chunks = load_document(str(file_path))
                
                if chunks:
                    if setup_rag_engine(chunks):
                        st.success(f"Document processed into {len(chunks)} chunks")
                        
                        if setup_gemini_client():
                            st.success("AI system ready!")
                        else:
                            st.warning("Gemini AI not available - check API key")
                    else:
                        st.error("Failed to setup RAG engine")
    
    # Main content area
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
                    answer, context = process_query(query)
                
                # Display answer
                st.subheader("🤖 Answer")
                st.write(answer)
                
                # Display context (expandable)
                with st.expander("📖 View Source Context"):
                    st.text(context)
            else:
                st.warning("Please enter a question")
    else:
        st.info("👆 Please upload a document to get started")
        
        # Show sample questions
        st.subheader("💡 Example Questions")
        st.markdown("""
        Once you upload a document, you can ask questions like:
        - What is the main topic of this document?
        - Can you summarize the key points?
        - What are the important dates mentioned?
        - Who are the main people discussed?
        """)


if __name__ == "__main__":
    main()
