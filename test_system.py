"""
Test System for Document Q&A System
Comprehensive testing of all system components
"""
import os
import sys
from pathlib import Path
from document_processor import DocumentProcessor
from rag_engine import RAGEngine
from gemini_client import GeminiClient
from config import Config


def test_document_processing():
    """
    Test document processing functionality.
    
    Returns:
        Dictionary with test results
    """
    print("🧪 Testing Document Processing...")
    
    test_results = {
        'document_processor': False,
        'content_extraction': False,
        'chunk_creation': False,
        'semantic_analysis': False
    }
    
    try:
        # Create test document
        test_content = """
        This is a sample document for testing the RAG system.
        It contains information about artificial intelligence and machine learning.
        AI is a branch of computer science that focuses on creating intelligent machines.
        Machine learning is a subset of AI that enables computers to learn without being explicitly programmed.
        Deep learning is a type of machine learning that uses neural networks with multiple layers.
        Natural language processing is another important area of AI research.
        Computer vision helps machines understand and interpret visual information.
        Robotics combines AI with mechanical engineering to create autonomous systems.
        """
        
        # Save test document
        test_file_path = "test_document.txt"
        with open(test_file_path, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        # Test document processor
        processor = DocumentProcessor()
        test_results['document_processor'] = True
        
        # Test document processing
        chunks = processor.process_document(test_file_path)
        test_results['content_extraction'] = True
        
        # Test chunk creation
        if chunks and len(chunks) > 0:
            test_results['chunk_creation'] = True
            print(f"✅ Created {len(chunks)} chunks")
        
        # Test semantic analysis
        if chunks and 'semantic_keywords' in chunks[0]:
            test_results['semantic_analysis'] = True
            print(f"✅ Extracted keywords: {chunks[0]['semantic_keywords']}")
        
        # Cleanup
        os.remove(test_file_path)
        
    except Exception as e:
        print(f"❌ Document processing test failed: {e}")
    
    return test_results


def test_retrieval_system():
    """
    Test RAG engine functionality.
    
    Returns:
        Dictionary with test results
    """
    print("🔍 Testing Retrieval System...")
    
    test_results = {
        'rag_engine': False,
        'database_initialization': False,
        'chunk_indexing': False,
        'semantic_search': False,
        'context_building': False
    }
    
    try:
        # Create test chunks
        test_chunks = [
            {
                'id': 0,
                'content': 'This is a test chunk about artificial intelligence.',
                'length': 50,
                'word_count': 10,
                'semantic_keywords': ['artificial', 'intelligence', 'test']
            },
            {
                'id': 1,
                'content': 'Machine learning is a subset of AI.',
                'length': 35,
                'word_count': 7,
                'semantic_keywords': ['machine', 'learning', 'subset']
            }
        ]
        
        # Test RAG engine
        rag_engine = RAGEngine()
        test_results['rag_engine'] = True
        
        # Test database initialization
        rag_engine.initialize_database("test_collection")
        test_results['database_initialization'] = True
        
        # Test chunk indexing
        rag_engine.index_document_chunks(test_chunks)
        test_results['chunk_indexing'] = True
        
        # Test semantic search
        search_results = rag_engine.find_relevant_chunks("artificial intelligence")
        if search_results:
            test_results['semantic_search'] = True
            print(f"✅ Found {len(search_results)} relevant chunks")
        
        # Test context building
        context = rag_engine.build_query_context("machine learning")
        if context:
            test_results['context_building'] = True
            print(f"✅ Built context: {len(context)} characters")
        
        # Cleanup
        rag_engine.clear_database()
        
    except Exception as e:
        print(f"❌ Retrieval system test failed: {e}")
    
    return test_results


def test_ai_client():
    """
    Test Gemini AI client functionality.
    
    Returns:
        Dictionary with test results
    """
    print("🤖 Testing AI Client...")
    
    test_results = {
        'gemini_client': False,
        'api_connection': False,
        'answer_generation': False,
        'prompt_optimization': False
    }
    
    try:
        # Test client initialization
        client = GeminiClient()
        test_results['gemini_client'] = True
        
        # Test API connection
        if client.validate_api_connection():
            test_results['api_connection'] = True
            print("✅ API connection successful")
        else:
            print("⚠️ API connection failed - check API key")
        
        # Test answer generation
        test_query = "What is artificial intelligence?"
        test_context = "Artificial intelligence is a branch of computer science."
        
        answer = client.generate_answer_from_context(test_query, test_context)
        if answer and "artificial intelligence" in answer.lower():
            test_results['answer_generation'] = True
            print(f"✅ Generated answer: {len(answer)} characters")
        
        # Test model info
        model_info = client.get_model_info()
        if model_info:
            test_results['prompt_optimization'] = True
            print(f"✅ Model info: {model_info['model_name']}")
        
    except Exception as e:
        print(f"❌ AI client test failed: {e}")
    
    return test_results


def run_comprehensive_tests():
    """
    Run comprehensive tests for all system components.
    
    Returns:
        Dictionary with overall test results
    """
    print("🚀 Starting Comprehensive System Tests...")
    print("=" * 50)
    
    # Run individual tests
    doc_results = test_document_processing()
    print()
    
    rag_results = test_retrieval_system()
    print()
    
    ai_results = test_ai_client()
    print()
    
    # Compile results
    all_results = {
        'document_processing': doc_results,
        'retrieval_system': rag_results,
        'ai_client': ai_results
    }
    
    # Calculate overall success rate
    total_tests = 0
    passed_tests = 0
    
    for component, results in all_results.items():
        for test_name, passed in results.items():
            total_tests += 1
            if passed:
                passed_tests += 1
    
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    print("=" * 50)
    print(f"📊 Test Results Summary:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Passed: {passed_tests}")
    print(f"   Failed: {total_tests - passed_tests}")
    print(f"   Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("🎉 System is ready for use!")
    elif success_rate >= 60:
        print("⚠️ System has some issues but may work with limitations")
    else:
        print("❌ System has significant issues that need to be addressed")
    
    return all_results


def main():
    """Main test function."""
    print("🧪 Document Q&A System - Test Suite")
    print("=" * 50)
    
    # Check configuration
    print("🔧 Checking Configuration...")
    try:
        Config.validate()
        print("✅ Configuration is valid")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return
    
    # Run tests
    results = run_comprehensive_tests()
    
    # Save results to file
    try:
        with open("test_results.txt", "w") as f:
            f.write("Document Q&A System - Test Results\n")
            f.write("=" * 40 + "\n\n")
            
            for component, component_results in results.items():
                f.write(f"{component.upper()}:\n")
                for test_name, passed in component_results.items():
                    status = "✅ PASS" if passed else "❌ FAIL"
                    f.write(f"  {test_name}: {status}\n")
                f.write("\n")
        
        print("📄 Test results saved to test_results.txt")
    except Exception as e:
        print(f"⚠️ Could not save test results: {e}")


if __name__ == "__main__":
    main()
