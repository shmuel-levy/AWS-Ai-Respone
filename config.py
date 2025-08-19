"""
Configuration settings for the RAG system
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the RAG system"""
    
    # Gemini AI Configuration
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    
    # Document Processing Settings
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 1000))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 200))
    MAX_CHUNKS = int(os.getenv('MAX_CHUNKS', 18))
    
    # Retrieval Settings
    RETRIEVAL_TOP_K = int(os.getenv('RETRIEVAL_TOP_K', 3))
    
    # Database Settings
    CHROMA_PERSIST_DIRECTORY = os.getenv('CHROMA_PERSIST_DIRECTORY', './chroma_db')
    
    # File Paths
    DATA_DIR = './data'
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required. Please set it in your .env file")
        
        return True
