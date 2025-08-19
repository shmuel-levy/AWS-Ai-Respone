"""
RAG Engine for Document Q&A System
Handles semantic search and retrieval of relevant document chunks
"""
import os
import chromadb
from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from config import Config


class RAGEngine:
    """Handles retrieval-augmented generation for document Q&A"""
    
    def __init__(self, persist_directory: str = None):
        """Initialize the RAG engine"""
        if persist_directory is None:
            persist_directory = Config.CHROMA_PERSIST_DIRECTORY
            
        self.persist_directory = persist_directory
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = None
        self.chunks = []
        
    def setup_collection(self, collection_name: str = "document_chunks"):
        """Setup ChromaDB collection for document chunks"""
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
    
    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """Add document chunks to the vector database"""
        if not chunks:
            return
            
        self.chunks = chunks
        self.setup_collection()
        
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
    
    def search_relevant_chunks(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Search for relevant document chunks based on query
        
        Args:
            query: User question
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
                'distance': results['distances'][0][i] if 'distances' in results else 0.0
            }
            relevant_chunks.append(chunk_data)
        
        return relevant_chunks
    
    def get_context_for_query(self, query: str) -> str:
        """
        Get relevant context for a query
        
        Args:
            query: User question
            
        Returns:
            Combined context from relevant chunks
        """
        relevant_chunks = self.search_relevant_chunks(query)
        
        if not relevant_chunks:
            return ""
        
        context_parts = []
        for chunk in relevant_chunks:
            context_parts.append(f"Section {chunk['metadata']['chunk_id']}: {chunk['content']}")
        
        return "\n\n".join(context_parts)
    
    def get_chunk_by_id(self, chunk_id: int) -> Dict[str, Any]:
        """Get specific chunk by ID"""
        for chunk in self.chunks:
            if chunk['id'] == chunk_id:
                return chunk
        return None
    
    def update_chunks(self, new_chunks: List[Dict[str, Any]]):
        """Update chunks in the database"""
        if self.collection:
            self.client.delete_collection(name=self.collection.name)
        
        self.add_chunks(new_chunks)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        if not self.collection:
            return {}
        
        count = self.collection.count()
        return {
            'total_chunks': count,
            'collection_name': self.collection.name,
            'persist_directory': self.persist_directory
        }
    
    def clear_collection(self):
        """Clear all data from the collection"""
        if self.collection:
            self.client.delete_collection(name=self.collection.name)
            self.collection = None
        self.chunks = []
    
    def similarity_search(self, query: str, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Perform similarity search with threshold filtering
        
        Args:
            query: Search query
            threshold: Minimum similarity threshold
            
        Returns:
            List of chunks above the similarity threshold
        """
        all_chunks = self.search_relevant_chunks(query, top_k=len(self.chunks))
        
        filtered_chunks = []
        for chunk in all_chunks:
            similarity_score = 1 - chunk['distance']  # Convert distance to similarity
            if similarity_score >= threshold:
                chunk['similarity_score'] = similarity_score
                filtered_chunks.append(chunk)
        
        return filtered_chunks
    
    def get_keyword_matches(self, query: str) -> List[Dict[str, Any]]:
        """
        Find chunks that match keywords from the query
        
        Args:
            query: User query
            
        Returns:
            List of chunks with keyword matches
        """
        query_keywords = self._extract_keywords(query.lower())
        matching_chunks = []
        
        for chunk in self.chunks:
            chunk_keywords = [kw.lower() for kw in chunk['semantic_keywords']]
            matches = set(query_keywords) & set(chunk_keywords)
            
            if matches:
                chunk_copy = chunk.copy()
                chunk_copy['keyword_matches'] = list(matches)
                chunk_copy['match_count'] = len(matches)
                matching_chunks.append(chunk_copy)
        
        return sorted(matching_chunks, key=lambda x: x['match_count'], reverse=True)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        words = text.lower().split()
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'what', 'how', 'why', 'when', 'where', 'who'}
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords
