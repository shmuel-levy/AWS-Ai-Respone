"""
Test script for Document Q&A System
Tests basic functionality of all components
"""
import os
from document_processor import DocumentProcessor
from rag_engine import RAGEngine
from gemini_client import GeminiClient
from config import Config


def test_document_processing():
    """Test document processing functionality"""
    print("Testing document processing...")
    
    processor = DocumentProcessor()
    
    # Create a sample document for testing
    sample_text = """
    This is a sample document for testing the RAG system.
    It contains information about artificial intelligence and machine learning.
    AI is a branch of computer science that focuses on creating intelligent machines.
    Machine learning is a subset of AI that enables computers to learn without being explicitly programmed.
    Deep learning is a type of machine learning that uses neural networks with multiple layers.
    Natural language processing is another important area of AI research.
    Computer vision helps machines understand and interpret visual information.
    Robotics combines AI with mechanical engineering to create autonomous systems.
    """
    
    # Process the document
    chunks = processor.segment_document(sample_text)
    print(f"Created {len(chunks)} chunks")
    
    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
        print(f"Chunk {i}: {chunk['content'][:100]}...")
    
    return chunks


def test_rag_engine(chunks):
    """Test RAG engine functionality"""
    print("\nTesting RAG engine...")
    
    rag_engine = RAGEngine()
    rag_engine.add_chunks(chunks)
    
    # Test search
    query = "What is artificial intelligence?"
    relevant_chunks = rag_engine.search_relevant_chunks(query)
    print(f"Found {len(relevant_chunks)} relevant chunks for query: '{query}'")
    
    context = rag_engine.get_context_for_query(query)
    print(f"Context length: {len(context)} characters")
    
    return context


def test_gemini_client(context):
    """Test Gemini client functionality"""
    print("\nTesting Gemini client...")
    
    try:
        client = GeminiClient()
        query = "What is artificial intelligence?"
        answer = client.generate_answer(query, context)
        print(f"Generated answer: {answer}")
        return True
    except Exception as e:
        print(f"Gemini client test failed: {e}")
        print("Make sure you have set the GEMINI_API_KEY in your .env file")
        return False


def main():
    """Run all tests"""
    print("Starting system tests...\n")
    
    # Test document processing
    chunks = test_document_processing()
    
    # Test RAG engine
    context = test_rag_engine(chunks)
    
    # Test Gemini client
    gemini_works = test_gemini_client(context)
    
    print("\n" + "="*50)
    print("Test Results:")
    print(f"Document Processing: ✓ ({len(chunks)} chunks created)")
    print(f"RAG Engine: ✓ (Context generated)")
    print(f"Gemini Client: {'✓' if gemini_works else '✗ (API key needed)'}")
    
    if gemini_works:
        print("\nAll components working! Ready to create web interface.")
    else:
        print("\nPlease set GEMINI_API_KEY in .env file to test full functionality.")


if __name__ == "__main__":
    main()
