"""
Test script to verify cloud compatibility without ChromaDB
"""
import os
import sys

def test_cloud_compatibility():
    """Test if the system works without ChromaDB (cloud environment)"""
    print("🧪 Testing Cloud Compatibility...")
    
    # Test 1: Import without ChromaDB
    try:
        # Temporarily remove chromadb from sys.modules if it exists
        if 'chromadb' in sys.modules:
            del sys.modules['chromadb']
        
        # Test imports
        from config import Config
        print("✅ Config imported successfully")
        
        from document_processor import DocumentProcessor
        print("✅ DocumentProcessor imported successfully")
        
        from rag_engine import RAGEngine
        print("✅ RAGEngine imported successfully")
        
        from gemini_client import GeminiClient
        print("✅ GeminiClient imported successfully")
        
        from app import initialize_application_state
        print("✅ App imported successfully")
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test 2: Initialize components
    try:
        # Test RAG Engine initialization
        engine = RAGEngine()
        info = engine.get_database_info()
        print(f"✅ RAG Engine initialized - Storage type: {info['storage_type']}")
        
        # Test Gemini Client initialization
        client = GeminiClient()
        print("✅ Gemini Client initialized")
        
        # Test Document Processor initialization
        processor = DocumentProcessor()
        print("✅ Document Processor initialized")
        
    except Exception as e:
        print(f"❌ Component initialization failed: {e}")
        return False
    
    # Test 3: Test basic functionality
    try:
        # Test document processing
        test_content = "This is a test document for cloud compatibility."
        
        # Create a test file
        with open("test_cloud.txt", "w", encoding="utf-8") as f:
            f.write(test_content)
        
        # Process the document
        chunks = processor.process_document("test_cloud.txt")
        print(f"✅ Document processing works - Created {len(chunks)} chunks")
        
        # Test RAG functionality
        engine.index_document_chunks(chunks)
        relevant_chunks = engine.find_relevant_chunks("test")
        print(f"✅ RAG search works - Found {len(relevant_chunks)} relevant chunks")
        
        # Clean up
        os.remove("test_cloud.txt")
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False
    
    print("🎉 All cloud compatibility tests passed!")
    return True

if __name__ == "__main__":
    success = test_cloud_compatibility()
    if success:
        print("\n✅ System is ready for cloud deployment!")
    else:
        print("\n❌ System needs fixes for cloud deployment!")
