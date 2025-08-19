"""
RAG Engine for Document Q&A System
Handles semantic search and retrieval of relevant document chunks
"""
import os
import chromadb
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from config import Config


class RAGEngine:
    """
    Handles retrieval-augmented generation for document Q&A.
    
    This class is responsible for:
    - Managing vector database (ChromaDB) for document chunks
    - Performing semantic search to find relevant chunks
    - Providing context for AI model queries
    - Managing chunk metadata and embeddings
    """
    
    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize the RAG engine with vector database.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
        """
        if persist_directory is None:
            persist_directory = Config.CHROMA_PERSIST_DIRECTORY
            
        self.persist_directory = persist_directory
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = None
        self.document_chunks = []
        
    def initialize_database(self, collection_name: str = "document_chunks") -> None:
        """
        Initialize ChromaDB collection for document chunks.
        
        Args:
            collection_name: Name of the collection to create or connect to
        """
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
    
    def index_document_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Index document chunks into the vector database.
        
        Args:
            chunks: List of document chunks with metadata
        """
        if not chunks:
            return
            
        self.document_chunks = chunks
        self.initialize_database()
        
        documents = []
        metadatas = []
        ids = []
        
        for chunk in chunks:
            documents.append(chunk['content'])
            metadatas.append({
                'chunk_id': chunk['id'],
                'length': chunk['length'],
                'word_count': chunk['word_count'],
                'keywords': ', '.join(chunk['semantic_keywords'])
            })
            ids.append(f"chunk_{chunk['id']}")
        
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    def find_relevant_chunks(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Find relevant document chunks based on semantic similarity.
        
        Args:
            query: User question or search query
            top_k: Number of top results to return
            
        Returns:
            List of relevant chunks with similarity scores
        """
        if top_k is None:
            top_k = Config.RETRIEVAL_TOP_K
            
        if not self.collection:
            return []
        
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        relevant_chunks = []
        for i in range(len(results['documents'][0])):
            chunk_data = {
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'id': results['ids'][0][i],
                'similarity_score': results['distances'][0][i] if 'distances' in results else 0.0
            }
            relevant_chunks.append(chunk_data)
        
        return relevant_chunks
    
    def build_query_context(self, query: str) -> str:
        """
        Build context string from relevant chunks for AI model.
        
        Args:
            query: User question
            
        Returns:
            Formatted context string with relevant chunks
        """
        relevant_chunks = self.find_relevant_chunks(query)
        
        if not relevant_chunks:
            return ""
        
        context_parts = []
        for i, chunk in enumerate(relevant_chunks):
            context_parts.append(f"Section {i}: {chunk['content']}")
        
        return "\n\n".join(context_parts)
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve specific chunk by its ID.
        
        Args:
            chunk_id: ID of the chunk to retrieve
            
        Returns:
            Chunk data if found, None otherwise
        """
        if not self.collection:
            return None
        
        try:
            results = self.collection.get(ids=[chunk_id])
            if results['documents']:
                return {
                    'content': results['documents'][0],
                    'metadata': results['metadatas'][0] if results['metadatas'] else {},
                    'id': chunk_id
                }
        except:
            pass
        
        return None
    
    def update_chunk_index(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Update the chunk index with new document chunks.
        
        Args:
            chunks: New document chunks to index
        """
        # Clear existing collection
        if self.collection:
            self.collection.delete()
        
        # Re-index with new chunks
        self.index_document_chunks(chunks)
    
    def get_database_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database.
        
        Returns:
            Dictionary with database statistics
        """
        if not self.collection:
            return {
                'total_chunks': 0,
                'collection_name': None,
                'is_initialized': False
            }
        
        try:
            count = self.collection.count()
            return {
                'total_chunks': count,
                'collection_name': self.collection.name,
                'is_initialized': True
            }
        except:
            return {
                'total_chunks': 0,
                'collection_name': None,
                'is_initialized': False
            }
    
    def clear_database(self) -> None:
        """Clear all data from the vector database."""
        if self.collection:
            self.collection.delete()
            self.collection = None
        self.document_chunks = []
    
    def perform_semantic_search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Perform semantic search with additional metadata.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of search results with enhanced metadata
        """
        relevant_chunks = self.find_relevant_chunks(query, top_k)
        
        # Enhance results with additional information
        enhanced_results = []
        for chunk in relevant_chunks:
            enhanced_chunk = {
                **chunk,
                'relevance_score': 1.0 - chunk.get('similarity_score', 0.0),
                'content_preview': chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content']
            }
            enhanced_results.append(enhanced_chunk)
        
        return enhanced_results
    
    def find_keyword_matches(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """
        Find chunks that contain specific keywords.
        
        Args:
            keywords: List of keywords to search for
            
        Returns:
            List of chunks containing the keywords
        """
        if not self.collection:
            return []
        
        # Create a query from keywords
        query = " ".join(keywords)
        
        # Search for chunks containing these keywords
        results = self.collection.query(
            query_texts=[query],
            n_results=len(self.document_chunks)
        )
        
        keyword_matches = []
        for i in range(len(results['documents'][0])):
            chunk_content = results['documents'][0][i]
            
            # Check if all keywords are present
            if all(keyword.lower() in chunk_content.lower() for keyword in keywords):
                chunk_data = {
                    'content': chunk_content,
                    'metadata': results['metadatas'][0][i],
                    'id': results['ids'][0][i],
                    'keyword_matches': [kw for kw in keywords if kw.lower() in chunk_content.lower()]
                }
                keyword_matches.append(chunk_data)
        
        return keyword_matches
    
    def _extract_search_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text for search optimization.
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            List of extracted keywords
        """
        # Simple keyword extraction - can be enhanced
        words = text.lower().split()
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'what', 'how', 'why', 'when', 'where', 'who', 'which', 'that', 'this'
        }
        
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Return unique keywords
        return list(set(keywords))
