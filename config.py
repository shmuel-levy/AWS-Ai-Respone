"""
Configuration settings for the Document Q&A System
Centralized configuration management for all system components
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
try:
    load_dotenv()
except Exception:
    # Silently continue if .env file is not found or malformed
    pass


class Config:
    """
    Configuration class for the Document Q&A System.
    
    This class manages all configuration settings including:
    - API keys and external service configurations
    - Document processing parameters
    - Database and storage settings
    - System behavior parameters
    """
    
    # ============================================================================
    # AI Service Configuration
    # ============================================================================
    
    # Gemini AI Configuration
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    
    # ============================================================================
    # Document Processing Configuration
    # ============================================================================
    
    # Chunk size and overlap settings
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 1000))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 200))
    MAX_CHUNKS = int(os.getenv('MAX_CHUNKS', 18))
    
    # ============================================================================
    # Retrieval System Configuration
    # ============================================================================
    
    # Number of relevant chunks to retrieve per query
    RETRIEVAL_TOP_K = int(os.getenv('RETRIEVAL_TOP_K', 3))
    
    # ============================================================================
    # Storage and Database Configuration
    # ============================================================================
    
    # ChromaDB persistence directory
    CHROMA_PERSIST_DIRECTORY = os.getenv('CHROMA_PERSIST_DIRECTORY', './chroma_db')
    
    # Data directory for storing processed chunks
    DATA_DIR = './data'
    
    # ============================================================================
    # System Configuration
    # ============================================================================
    
    # Supported file types
    SUPPORTED_FILE_TYPES = ['.pdf', '.txt']
    
    # Maximum file size (in bytes) - 50MB
    MAX_FILE_SIZE = 50 * 1024 * 1024
    
    @classmethod
    def validate(cls) -> bool:
        """
        Validate required configuration settings.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If required configuration is missing
        """
        if not cls.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is required. Please set it in your .env file or "
                "environment variables."
            )
        
        # Validate numeric settings
        if cls.CHUNK_SIZE <= 0:
            raise ValueError("CHUNK_SIZE must be positive")
        
        if cls.CHUNK_OVERLAP < 0:
            raise ValueError("CHUNK_OVERLAP must be non-negative")
        
        if cls.MAX_CHUNKS <= 0:
            raise ValueError("MAX_CHUNKS must be positive")
        
        if cls.RETRIEVAL_TOP_K <= 0:
            raise ValueError("RETRIEVAL_TOP_K must be positive")
        
        return True
    
    @classmethod
    def get_config_summary(cls) -> dict:
        """
        Get a summary of current configuration settings.
        
        Returns:
            Dictionary with configuration summary
        """
        return {
            'ai_service': {
                'gemini_api_key_configured': bool(cls.GEMINI_API_KEY)
            },
            'document_processing': {
                'chunk_size': cls.CHUNK_SIZE,
                'chunk_overlap': cls.CHUNK_OVERLAP,
                'max_chunks': cls.MAX_CHUNKS
            },
            'retrieval': {
                'top_k_results': cls.RETRIEVAL_TOP_K
            },
            'storage': {
                'chroma_persist_directory': cls.CHROMA_PERSIST_DIRECTORY,
                'data_directory': cls.DATA_DIR
            },
            'system': {
                'supported_file_types': cls.SUPPORTED_FILE_TYPES,
                'max_file_size_mb': cls.MAX_FILE_SIZE / (1024 * 1024)
            }
        }
