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
    st.error(f"⚠️ Import Error: {e}")
    st.info("Running in fallback mode - some features may be limited")
    CHROMA_AVAILABLE = False
except Exception as e:
    st.error(f"⚠️ Unexpected Error: {e}")
    st.info("Running in fallback mode - some features may be limited")
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
    if 'fallback_mode' not in st.session_state:
        st.session_state.fallback_mode = not CHROMA_AVAILABLE

def process_uploaded_document(file_path: str):
    """
    Process uploaded document and create semantic chunks.

    Args:
        file_path: Path to the uploaded document

    Returns:
        List of document chunks with metadata, or None if processing fails
    """
    if not CHROMA_AVAILABLE:
        st.error("❌ Document processing is not available in fallback mode")
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

def render_header():
    """Render the main header section."""
    st.markdown('<h1 class="main-header">📚 מערכת שאלות ותשובות על מסמכים</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Document Q&A System powered by Google Gemini AI</p>', unsafe_allow_html=True)

def render_upload_section():
    """Render the document upload section."""
    st.markdown("### 📄 העלאת מסמך / Upload Document")
    
    if st.session_state.fallback_mode:
        st.warning("⚠️ Running in fallback mode - document processing is limited")
        st.info("This is a demonstration mode. Full functionality requires local deployment.")
        return
    
    uploaded_file = st.file_uploader(
        "בחר קובץ PDF או TXT / Choose PDF or TXT file",
        type=['pdf', 'txt'],
        help="Upload a document to start asking questions"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_file_path = f"temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process document
        with st.spinner("מעבד מסמך... / Processing document..."):
            chunks = process_uploaded_document(temp_file_path)
        
        if chunks:
            st.success(f"✅ המסמך עובד בהצלחה! / Document processed successfully!")
            st.info(f"נוצרו {len(chunks)} חלקים / Created {len(chunks)} chunks")
            
            # Clean up temp file
            try:
                os.remove(temp_file_path)
            except:
                pass
        else:
            st.error("❌ שגיאה בעיבוד המסמך / Error processing document")
            
            # Clean up temp file
            try:
                os.remove(temp_file_path)
            except:
                pass

def render_question_section():
    """Render the question input section."""
    st.markdown("### ❓ שאל שאלה / Ask a Question")
    
    if not st.session_state.document_loaded and not st.session_state.fallback_mode:
        st.info("📝 אנא העלה מסמך תחילה / Please upload a document first")
        return
    
    if st.session_state.fallback_mode:
        st.info("🎭 Demo Mode: Try asking a question about AWS cloud architecture")
    
    question = st.text_input(
        "הקלד את השאלה שלך / Type your question:",
        placeholder="למשל: מה זה AWS Lambda? / Example: What is AWS Lambda?",
        help="Ask any question about the uploaded document"
    )
    
    if st.button("🔍 חפש תשובה / Search Answer", type="primary"):
        if question.strip():
            process_question(question)
        else:
            st.warning("⚠️ אנא הקלד שאלה / Please type a question")

def process_question(question: str):
    """Process user question and generate answer."""
    if st.session_state.fallback_mode:
        # Demo mode - provide sample answers
        provide_demo_answer(question)
        return
    
    if not st.session_state.document_loaded:
        st.error("❌ אין מסמך זמין / No document available")
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
            st.warning("⚠️ לא נמצאו חלקים רלוונטיים / No relevant chunks found")
            return
        
        # Build context
        context = "\n\n".join([chunk['content'] for chunk in relevant_chunks])
        
        # Generate answer
        with st.spinner("מחפש תשובה... / Searching for answer..."):
            answer = st.session_state.gemini_client.generate_answer_from_context(question, context)
        
        # Display results
        display_answer(question, answer, relevant_chunks)
        
    except Exception as e:
        st.error(f"❌ שגיאה בעיבוד השאלה / Error processing question: {str(e)}")

def provide_demo_answer(question: str):
    """Provide demo answers in fallback mode."""
    st.markdown("### 🎭 Demo Mode Answer")
    
    # Sample AWS-related answers
    demo_answers = {
        "aws lambda": "AWS Lambda is a serverless compute service that runs your code in response to events and automatically manages the underlying compute resources.",
        "ec2": "Amazon EC2 (Elastic Compute Cloud) provides scalable computing capacity in the AWS cloud, allowing you to launch virtual servers.",
        "s3": "Amazon S3 (Simple Storage Service) is an object storage service offering industry-leading scalability, data availability, security, and performance.",
        "cloud": "Cloud computing is the on-demand delivery of IT resources over the internet with pay-as-you-go pricing.",
        "serverless": "Serverless computing allows you to run code without provisioning or managing servers, paying only for the compute time you consume."
    }
    
    # Find relevant demo answer
    question_lower = question.lower()
    relevant_answer = None
    
    for key, answer in demo_answers.items():
        if key in question_lower:
            relevant_answer = answer
            break
    
    if relevant_answer:
        st.success("✅ Found relevant information!")
        st.markdown(f"**Answer:** {relevant_answer}")
    else:
        st.info("💡 This is a demo mode. Try asking about AWS services like Lambda, EC2, S3, or cloud computing.")
    
    st.markdown("---")
    st.markdown("**Note:** This is a demonstration. For full functionality, deploy locally with all dependencies.")

def display_answer(question: str, answer: str, relevant_chunks: list):
    """Display the generated answer and relevant context."""
    st.markdown("### 💡 תשובה / Answer")
    
    # Display answer with RTL support
    if any('\u0590' <= char <= '\u05FF' for char in answer):
        st.markdown(f'<div class="hebrew-answer">{answer}</div>', unsafe_allow_html=True)
    else:
        st.markdown(answer)
    
    # Display relevant context
    st.markdown("### 📖 הקשר רלוונטי / Relevant Context")
    
    for i, chunk in enumerate(relevant_chunks):
        with st.expander(f"חלק {i+1} / Chunk {i+1} (מילים {chunk.get('word_count', 'N/A')} / words {chunk.get('word_count', 'N/A')})"):
            chunk_content = chunk.get('content', '')
            if any('\u0590' <= char <= '\u05FF' for char in chunk_content):
                st.markdown(f'<div class="hebrew-text">{chunk_content}</div>', unsafe_allow_html=True)
            else:
                st.markdown(chunk_content)

def render_sidebar():
    """Render the sidebar with additional information."""
    with st.sidebar:
        st.markdown("### ℹ️ מידע / Information")
        
        if st.session_state.fallback_mode:
            st.warning("🎭 Demo Mode Active")
            st.info("Limited functionality for demonstration purposes")
        else:
            st.success("🚀 Full Mode Active")
            st.info("All features available")
        
        st.markdown("---")
        st.markdown("### 🔧 תכונות / Features")
        st.markdown("- 📄 תמיכה ב-PDF ו-TXT")
        st.markdown("- 🔍 חיפוש סמנטי")
        st.markdown("- 🤖 תשובות מבוססות AI")
        st.markdown("- 🌐 תמיכה בעברית ואנגלית")
        
        st.markdown("---")
        st.markdown("### 📚 איך זה עובד / How it works")
        st.markdown("1. העלה מסמך")
        st.markdown("2. שאל שאלה")
        st.markdown("3. קבל תשובה מבוססת על המסמך")
        
        if st.button("🔄 Reset Session"):
            st.session_state.clear()
            st.rerun()

def main():
    """Main application function."""
    setup_page_configuration()
    initialize_application_state()
    
    render_header()
    
    # Check deployment status
    if st.session_state.fallback_mode:
        st.warning("⚠️ **Deployment Notice:** Running in fallback mode due to dependency issues.")
        st.info("For full functionality, consider deploying locally or using a different hosting service.")
        st.markdown("---")
    
    render_upload_section()
    render_question_section()
    render_sidebar()

if __name__ == "__main__":
    main()
