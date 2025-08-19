"""
Gemini AI Client for Document Q&A System
Handles communication with Google's Gemini AI model
"""
import google.generativeai as genai
from typing import Dict, Any, Optional
from config import Config


class GeminiClient:
    """Client for interacting with Gemini AI"""
    
    def __init__(self, api_key: str = None):
        """Initialize Gemini client"""
        if api_key is None:
            api_key = Config.GEMINI_API_KEY
            
        if not api_key:
            raise ValueError("Gemini API key is required")
            
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
    def generate_answer(self, query: str, context: str) -> str:
        """
        Generate answer based on query and document context
        
        Args:
            query: User question
            context: Relevant document context
            
        Returns:
            Generated answer
        """
        if not context.strip():
            return "I apologize, I don't know how to answer this question."
            
        prompt = self._create_prompt(query, context)
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create prompt for Gemini AI"""
        return f"""
You are a helpful assistant that answers questions based on the provided document context.

Document Context:
{context}

Question: {query}

Instructions:
1. Answer ONLY based on the provided document context
2. If the information is not in the context, respond with: "I apologize, I don't know how to answer this question."
3. Be concise and accurate
4. Highlight important information from the document

Answer:
"""
