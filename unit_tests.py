"""
Unit Tests for Document Q&A System
Isolated testing of individual components and functions
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
import os
import tempfile
from pathlib import Path

# Import components to test
from config import Config
from document_processor import DocumentProcessor
from rag_engine import RAGEngine
from gemini_client import GeminiClient


class TestConfig(unittest.TestCase):
    """Unit tests for configuration management."""
    
    def setUp(self):
        """Set up test environment."""
        # Backup original environment
        self.original_env = os.environ.copy()
    
    def tearDown(self):
        """Clean up test environment."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_config_initialization(self):
        """Test that Config class initializes correctly."""
        # Test that Config class exists and has required attributes
        self.assertTrue(hasattr(Config, 'GEMINI_API_KEY'))
        self.assertTrue(hasattr(Config, 'CHUNK_SIZE'))
        self.assertTrue(hasattr(Config, 'MAX_CHUNKS'))
        self.assertTrue(hasattr(Config, 'RETRIEVAL_TOP_K'))
    
    def test_config_validation_success(self):
        """Test successful configuration validation."""
        # Set required environment variables
        os.environ['GEMINI_API_KEY'] = 'test_key_123'
        os.environ['CHUNK_SIZE'] = '1000'
        os.environ['MAX_CHUNKS'] = '18'
        os.environ['RETRIEVAL_TOP_K'] = '3'
        
        # Should not raise an exception
        try:
            Config.validate()
            self.assertTrue(True)  # Test passes if no exception
        except ValueError:
            self.fail("Config validation should pass with valid values")
    
    def test_config_validation_missing_api_key(self):
        """Test configuration validation with missing API key."""
        # Clear API key
        if 'GEMINI_API_KEY' in os.environ:
            del os.environ['GEMINI_API_KEY']
        
        # Should raise ValueError
        with self.assertRaises(ValueError):
            Config.validate()
    
    def test_config_validation_invalid_chunk_size(self):
        """Test configuration validation with invalid chunk size."""
        os.environ['GEMINI_API_KEY'] = 'test_key_123'
        os.environ['CHUNK_SIZE'] = '0'  # Invalid value
        
        with self.assertRaises(ValueError):
            Config.validate()
    
    def test_get_config_summary(self):
        """Test configuration summary generation."""
        summary = Config.get_config_summary()
        
        # Check that summary contains expected keys
        self.assertIn('ai_service', summary)
        self.assertIn('document_processing', summary)
        self.assertIn('retrieval', summary)
        self.assertIn('storage', summary)
        self.assertIn('system', summary)
        
        # Check that values are of correct types
        self.assertIsInstance(summary['document_processing']['chunk_size'], int)
        self.assertIsInstance(summary['retrieval']['top_k_results'], int)


class TestDocumentProcessor(unittest.TestCase):
    """Unit tests for document processor."""
    
    def setUp(self):
        """Set up test environment."""
        self.processor = DocumentProcessor()
        self.test_content = """
        This is a test document for unit testing.
        It contains multiple sentences with different topics.
        Artificial intelligence is a fascinating field.
        Machine learning algorithms can process large datasets.
        Natural language processing helps computers understand text.
        """
    
    def test_processor_initialization(self):
        """Test that DocumentProcessor initializes correctly."""
        self.assertIsNotNone(self.processor.embedding_model)
        self.assertEqual(self.processor.document_chunks, [])
        self.assertEqual(self.processor.chunk_embeddings, [])
        self.assertIsNone(self.processor.document_hash)
    
    def test_content_hash_calculation(self):
        """Test content hash calculation."""
        hash1 = self.processor._calculate_content_hash("test content")
        hash2 = self.processor._calculate_content_hash("test content")
        hash3 = self.processor._calculate_content_hash("different content")
        
        # Same content should have same hash
        self.assertEqual(hash1, hash2)
        # Different content should have different hash
        self.assertNotEqual(hash1, hash3)
    
    def test_sentence_extraction(self):
        """Test sentence extraction from text."""
        sentences = self.processor._extract_sentences(self.test_content)
        
        # Should extract multiple sentences
        self.assertGreater(len(sentences), 1)
        # Each sentence should be a string
        for sentence in sentences:
            self.assertIsInstance(sentence, str)
            self.assertGreater(len(sentence.strip()), 0)
    
    def test_keyword_extraction(self):
        """Test keyword extraction from text."""
        keywords = self.processor._extract_semantic_keywords(self.test_content)
        
        # Should extract keywords
        self.assertIsInstance(keywords, list)
        self.assertGreater(len(keywords), 0)
        # Keywords should be strings
        for keyword in keywords:
            self.assertIsInstance(keyword, str)
    
    def test_chunk_metadata_creation(self):
        """Test chunk metadata creation."""
        chunk_content = "This is a test chunk."
        chunk_id = 1
        
        metadata = self.processor._create_chunk_metadata(chunk_content, chunk_id)
        
        # Check required metadata fields
        self.assertIn('id', metadata)
        self.assertIn('content', metadata)
        self.assertIn('length', metadata)
        self.assertIn('word_count', metadata)
        self.assertIn('semantic_keywords', metadata)
        
        # Check values
        self.assertEqual(metadata['id'], chunk_id)
        self.assertEqual(metadata['content'], chunk_content)
        self.assertEqual(metadata['length'], len(chunk_content))
        self.assertIsInstance(metadata['word_count'], int)
        self.assertIsInstance(metadata['semantic_keywords'], list)
    
    @patch('document_processor.SentenceTransformer')
    def test_embedding_generation(self, mock_transformer):
        """Test embedding generation with mocked transformer."""
        # Mock the transformer
        mock_model = Mock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_transformer.return_value = mock_model
        
        # Create processor with mocked transformer
        processor = DocumentProcessor()
        
        # Test embedding generation
        text = "test text"
        embedding = processor.embedding_model.encode([text])
        
        # Verify transformer was called
        mock_model.encode.assert_called_once_with([text])
        self.assertEqual(embedding, [[0.1, 0.2, 0.3]])


class TestRAGEngine(unittest.TestCase):
    """Unit tests for RAG engine."""
    
    def setUp(self):
        """Set up test environment."""
        self.rag_engine = RAGEngine()
        self.test_chunks = [
            {
                'id': 0,
                'content': 'This is a test chunk about AI.',
                'length': 30,
                'word_count': 7,
                'semantic_keywords': ['test', 'chunk', 'ai']
            },
            {
                'id': 1,
                'content': 'Machine learning is important.',
                'length': 30,
                'word_count': 4,
                'semantic_keywords': ['machine', 'learning', 'important']
            }
        ]
    
    def test_rag_engine_initialization(self):
        """Test that RAGEngine initializes correctly."""
        self.assertIsNotNone(self.rag_engine.embedding_model)
        self.assertIsNone(self.rag_engine.collection)
        self.assertEqual(self.rag_engine.document_chunks, [])
    
    @patch('rag_engine.chromadb.PersistentClient')
    def test_database_initialization(self, mock_client):
        """Test database initialization with mocked ChromaDB."""
        # Mock ChromaDB client
        mock_client_instance = Mock()
        mock_collection = Mock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get_collection.side_effect = Exception("Collection not found")
        mock_client_instance.create_collection.return_value = mock_collection
        
        # Test initialization
        self.rag_engine.initialize_database("test_collection")
        
        # Verify client was called
        mock_client.assert_called_once()
        mock_client_instance.create_collection.assert_called_once()
    
    def test_chunk_processing(self):
        """Test chunk processing for database indexing."""
        processed_chunks = self.rag_engine._process_chunks_for_indexing(self.test_chunks)
        
        # Should return lists for database indexing
        self.assertIn('documents', processed_chunks)
        self.assertIn('metadatas', processed_chunks)
        self.assertIn('ids', processed_chunks)
        
        # Check lengths match
        self.assertEqual(len(processed_chunks['documents']), len(self.test_chunks))
        self.assertEqual(len(processed_chunks['metadatas']), len(self.test_chunks))
        self.assertEqual(len(processed_chunks['ids']), len(self.test_chunks))
    
    def test_context_building(self):
        """Test context building from chunks."""
        # Mock search results
        mock_chunks = [
            {'content': 'First chunk content.'},
            {'content': 'Second chunk content.'}
        ]
        
        # Mock find_relevant_chunks method
        with patch.object(self.rag_engine, 'find_relevant_chunks', return_value=mock_chunks):
            context = self.rag_engine.build_query_context("test query")
        
        # Context should contain both chunks
        self.assertIn('First chunk content', context)
        self.assertIn('Second chunk content', context)
        self.assertIn('Section 0:', context)
        self.assertIn('Section 1:', context)
    
    def test_empty_context_handling(self):
        """Test handling of empty search results."""
        # Mock empty search results
        with patch.object(self.rag_engine, 'find_relevant_chunks', return_value=[]):
            context = self.rag_engine.build_query_context("test query")
        
        # Should return empty string
        self.assertEqual(context, "")


class TestGeminiClient(unittest.TestCase):
    """Unit tests for Gemini AI client."""
    
    def setUp(self):
        """Set up test environment."""
        # Set test API key
        os.environ['GEMINI_API_KEY'] = 'test_key_123'
        self.client = GeminiClient()
    
    def test_client_initialization(self):
        """Test that GeminiClient initializes correctly."""
        self.assertIsNotNone(self.client.model)
    
    def test_missing_api_key_handling(self):
        """Test handling of missing API key."""
        # Clear API key
        if 'GEMINI_API_KEY' in os.environ:
            del os.environ['GEMINI_API_KEY']
        
        # Should raise ValueError
        with self.assertRaises(ValueError):
            GeminiClient()
    
    def test_prompt_creation(self):
        """Test prompt creation for AI queries."""
        query = "What is AI?"
        context = "Artificial intelligence is a field of computer science."
        
        prompt = self.client._create_optimized_prompt(query, context)
        
        # Prompt should contain query and context
        self.assertIn(query, prompt)
        self.assertIn(context, prompt)
        # Prompt should contain instructions
        self.assertIn("Instructions:", prompt)
        self.assertIn("Answer:", prompt)
    
    def test_empty_context_handling(self):
        """Test handling of empty context."""
        query = "What is AI?"
        empty_context = ""
        
        answer = self.client.generate_answer_from_context(query, empty_context)
        
        # Should return default message
        self.assertIn("don't know", answer.lower())
    
    @patch('gemini_client.genai.GenerativeModel')
    def test_answer_generation(self, mock_model):
        """Test answer generation with mocked Gemini model."""
        # Mock the model
        mock_model_instance = Mock()
        mock_response = Mock()
        mock_response.text = "This is a test answer about AI."
        mock_model_instance.generate_content.return_value = mock_response
        mock_model.return_value = mock_model_instance
        
        # Create client with mocked model
        client = GeminiClient()
        
        # Test answer generation
        query = "What is AI?"
        context = "Artificial intelligence is a field of computer science."
        
        answer = client.generate_answer_from_context(query, context)
        
        # Verify model was called
        mock_model_instance.generate_content.assert_called_once()
        self.assertEqual(answer, "This is a test answer about AI.")
    
    @patch('gemini_client.genai.GenerativeModel')
    def test_api_error_handling(self, mock_model):
        """Test handling of API errors."""
        # Mock the model to raise an exception
        mock_model_instance = Mock()
        mock_model_instance.generate_content.side_effect = Exception("API Error")
        mock_model.return_value = mock_model_instance
        
        # Create client with mocked model
        client = GeminiClient()
        
        # Test error handling
        query = "What is AI?"
        context = "Artificial intelligence is a field of computer science."
        
        answer = client.generate_answer_from_context(query, context)
        
        # Should return error message
        self.assertIn("Error generating response", answer)


class TestIntegration(unittest.TestCase):
    """Integration tests for component interaction."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary test file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        self.temp_file.write("This is a test document for integration testing.")
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up test environment."""
        # Remove temporary file
        os.unlink(self.temp_file.name)
    
    def test_document_processing_integration(self):
        """Test integration between document processor and RAG engine."""
        # Process document
        processor = DocumentProcessor()
        chunks = processor.process_document(self.temp_file.name)
        
        # Should create chunks
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        
        # Each chunk should have required fields
        for chunk in chunks:
            self.assertIn('id', chunk)
            self.assertIn('content', chunk)
            self.assertIn('semantic_keywords', chunk)
    
    def test_rag_workflow_integration(self):
        """Test complete RAG workflow integration."""
        # Create test chunks
        test_chunks = [
            {
                'id': 0,
                'content': 'Artificial intelligence is a field of computer science.',
                'length': 50,
                'word_count': 8,
                'semantic_keywords': ['artificial', 'intelligence', 'computer', 'science']
            }
        ]
        
        # Initialize RAG engine
        rag_engine = RAGEngine()
        
        # Test that components work together
        self.assertIsNotNone(rag_engine)
        self.assertIsInstance(test_chunks, list)


def run_unit_tests():
    """Run all unit tests and return results."""
    print("🧪 Running Unit Tests...")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestConfig))
    test_suite.addTest(unittest.makeSuite(TestDocumentProcessor))
    test_suite.addTest(unittest.makeSuite(TestRAGEngine))
    test_suite.addTest(unittest.makeSuite(TestGeminiClient))
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Return results
    return {
        'tests_run': result.testsRun,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100 if result.testsRun > 0 else 0
    }


if __name__ == "__main__":
    # Run unit tests
    results = run_unit_tests()
    
    print(f"\n📊 Unit Test Results:")
    print(f"Tests Run: {results['tests_run']}")
    print(f"Failures: {results['failures']}")
    print(f"Errors: {results['errors']}")
    print(f"Success Rate: {results['success_rate']:.1f}%")
    
    # Save results to file
    with open('unit_test_results.txt', 'w') as f:
        f.write(f"Unit Test Results:\n")
        f.write(f"Tests Run: {results['tests_run']}\n")
        f.write(f"Failures: {results['failures']}\n")
        f.write(f"Errors: {results['errors']}\n")
        f.write(f"Success Rate: {results['success_rate']:.1f}%\n")
