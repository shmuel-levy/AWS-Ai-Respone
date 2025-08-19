"""
Gemini AI Client for Document Q&A System
Handles communication with Google's Gemini AI model for generating answers
"""
import google.generativeai as genai
from typing import Dict, Any, Optional
from config import Config


class GeminiClient:
    """
    Client for interacting with Google's Gemini AI model.
    
    This class is responsible for:
    - Managing connection to Gemini AI API
    - Generating answers based on document context
    - Handling API errors and responses
    - Creating optimized prompts for better responses
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini AI client.
        
        Args:
            api_key: Gemini API key (defaults to Config.GEMINI_API_KEY)
            
        Raises:
            ValueError: If API key is not provided
        """
        if api_key is None:
            api_key = Config.GEMINI_API_KEY
            
        if not api_key:
            raise ValueError("Gemini API key is required")
            
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
    def generate_answer_from_context(self, query: str, context: str) -> str:
        """
        Generate answer based on user query and document context.
        
        Args:
            query: User's question
            context: Relevant document context
            
        Returns:
            Generated answer from Gemini AI
            
        Raises:
            Exception: If API call fails
        """
        if not context.strip():
            return "I apologize, I don't know how to answer this question."
            
        prompt = self._create_optimized_prompt(query, context)
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _create_optimized_prompt(self, query: str, context: str) -> str:
        """
        Create an optimized prompt for Gemini AI.
        
        Args:
            query: User's question
            context: Document context
            
        Returns:
            Formatted prompt for the AI model
        """
        return f"""
You are a helpful assistant that answers questions based on the provided document context.

Document Context:
{context}

Question: {query}

Instructions:
1. Answer ONLY based on the provided document context
2. If the information is not in the context, respond with: "I apologize, I don't know how to answer this question."
3. Be comprehensive and detailed in your response
4. When asked about practical examples or implementation details, provide specific information including:
   - Exact commands, code snippets, or configuration examples
   - Specific timeframes, durations, or deadlines mentioned
   - Names of services, tools, or technologies
   - Step-by-step processes or procedures
   - Specific metrics, values, or parameters
5. Highlight important information from the document
6. If the document contains Hebrew text, preserve the original Hebrew terms and provide English translations where helpful
7. Organize your response in a clear, structured format

Answer:
"""
    
    def validate_api_connection(self) -> bool:
        """
        Validate that the API connection is working.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            test_prompt = "Hello, this is a test message."
            response = self.model.generate_content(test_prompt)
            return response.text is not None
        except Exception:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': 'gemini-1.5-flash',
            'api_configured': True,
            'connection_valid': self.validate_api_connection()
        }
