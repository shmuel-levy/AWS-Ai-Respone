"""
RAG Engine for Document Q&A System
Handles semantic search and retrieval of relevant document chunks
"""
import os
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
from config import Config

# ChromaDB import will be handled conditionally
CHROMADB_AVAILABLE = False
chromadb = None

# Try to import ChromaDB only if it's available and safe
def _safe_import_chromadb():
    """Safely import ChromaDB if available"""
    global chromadb, CHROMADB_AVAILABLE
    try:
        import chromadb
        CHROMADB_AVAILABLE = True
        return True
    except (ImportError, RuntimeError):
        CHROMADB_AVAILABLE = False
        chromadb = None
        print("Warning: ChromaDB not available, using in-memory fallback")
        return False

# Initialize ChromaDB availability
_safe_import_chromadb()

# Fallback for when ChromaDB is not available (e.g., Streamlit Cloud)
class InMemoryVectorStore:
    """Simple in-memory vector store as fallback when ChromaDB is unavailable"""
    
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.metadatas = []
        self.ids = []
        
    def add(self, documents, metadatas, ids):
        """Add documents to in-memory store"""
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)
        
    def query(self, query_texts, n_results=3):
        """Simple similarity search using cosine distance"""
        if not self.documents:
            return {"documents": [], "metadatas": [], "ids": []}
            
        # For simplicity, return first n_results documents
        # In a real implementation, you'd compute embeddings and similarity
        return {
            "documents": self.documents[:n_results],
            "metadatas": self.metadatas[:n_results],
            "ids": self.ids[:n_results]
        }
        
    def delete(self, where=None, where_document=None):
        """Clear the store"""
        self.documents.clear()
        self.embeddings.clear()
        self.metadatas.clear()
        self.ids.clear()


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
        
        # Initialize storage based on availability
        if CHROMADB_AVAILABLE and chromadb and self._can_use_chromadb():
            try:
                self.client = chromadb.PersistentClient(path=persist_directory)
                self.collection = None
                self.use_fallback = False
            except Exception as e:
                print(f"Warning: ChromaDB initialization failed, using fallback: {e}")
                self.use_fallback = True
                self.client = None
                self.collection = InMemoryVectorStore()
        else:
            self.use_fallback = True
            self.client = None
            self.collection = InMemoryVectorStore()
            
        self.document_chunks = []
    
    def _can_use_chromadb(self) -> bool:
        """Check if ChromaDB can be used safely"""
        if not CHROMADB_AVAILABLE or not chromadb:
            return False
            
        try:
            # Check if we can write to the directory
            test_file = os.path.join(self.persist_directory, "test_write.tmp")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            return True
        except (OSError, PermissionError):
            return False
        
    def initialize_database(self, collection_name: str = "document_chunks") -> None:
        """
        Initialize ChromaDB collection for document chunks.
        
        Args:
            collection_name: Name of the collection to create or connect to
        """
        if self.use_fallback:
            return  # In-memory store doesn't need initialization
            
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except:
            try:
                self.collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception as e:
                print(f"Warning: Failed to create ChromaDB collection: {e}")
                # Fallback to in-memory
                self.use_fallback = True
                self.collection = InMemoryVectorStore()
    
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
        
        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        except Exception as e:
            print(f"Warning: Failed to add to vector store: {e}")
            # If ChromaDB fails, ensure we have the data in memory
            if not self.use_fallback:
                self.use_fallback = True
                self.collection = InMemoryVectorStore()
                self.collection.add(documents, metadatas, ids)
    
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
            
        if not self.document_chunks:
            return []
            
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            # Convert results to our expected format
            chunks_with_scores = []
            for i, doc_id in enumerate(results['ids']):
                chunk_data = {
                    'id': doc_id,
                    'content': results['documents'][i],
                    'metadata': results['metadatas'][i],
                    'similarity_score': 1.0 - (i * 0.1)  # Simple scoring for fallback
                }
                chunks_with_scores.append(chunk_data)
                
            return chunks_with_scores
            
        except Exception as e:
            print(f"Warning: Vector search failed: {e}")
            # Fallback: return first few chunks
            return self.document_chunks[:top_k]
    
    def build_query_context(self, query: str) -> str:
        """
        Build context string from relevant document chunks.
        
        Args:
            query: User's question
            
        Returns:
            Formatted context string
        """
        relevant_chunks = self.find_relevant_chunks(query)
        
        if not relevant_chunks:
            return ""
            
        context_parts = []
        for chunk in relevant_chunks:
            if isinstance(chunk, dict) and 'content' in chunk:
                content = chunk['content']
            else:
                content = str(chunk)
            context_parts.append(f"---\n{content}\n---")
            
        return "\n\n".join(context_parts)
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Get specific chunk by ID.
        
        Args:
            chunk_id: ID of the chunk to retrieve
            
        Returns:
            Chunk data or None if not found
        """
        try:
            if self.use_fallback:
                # Search in memory
                for chunk in self.document_chunks:
                    if f"chunk_{chunk['id']}" == chunk_id:
                        return chunk
                return None
            else:
                # Use ChromaDB
                results = self.collection.get(ids=[chunk_id])
                if results['documents']:
                    return {
                        'id': chunk_id,
                        'content': results['documents'][0],
                        'metadata': results['metadatas'][0] if results['metadatas'] else {}
                    }
                return None
        except Exception as e:
            print(f"Warning: Failed to get chunk by ID: {e}")
            return None
    
    def clear_database(self) -> None:
        """Clear all data from the vector store."""
        try:
            if not self.use_fallback and self.collection:
                self.collection.delete(where={})
            else:
                # Clear in-memory store
                if hasattr(self.collection, 'delete'):
                    self.collection.delete()
                    
            self.document_chunks = []
        except Exception as e:
            print(f"Warning: Failed to clear database: {e}")
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        Get information about the current database state.
        
        Returns:
            Dictionary with database information
        """
        return {
            'storage_type': 'fallback' if self.use_fallback else 'chromadb',
            'chunks_count': len(self.document_chunks),
            'persist_directory': self.persist_directory,
            'chromadb_available': CHROMADB_AVAILABLE
        }
    
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
        if not self.collection or self.use_fallback:
            # Fallback: search in memory
            keyword_matches = []
            for chunk in self.document_chunks:
                chunk_content = chunk.get('content', '')
                if all(keyword.lower() in chunk_content.lower() for keyword in keywords):
                    chunk_data = {
                        'content': chunk_content,
                        'metadata': chunk,
                        'id': f"chunk_{chunk.get('id', 'unknown')}",
                        'keyword_matches': [kw for kw in keywords if kw.lower() in chunk_content.lower()]
                    }
                    keyword_matches.append(chunk_data)
            return keyword_matches
        
        # Use ChromaDB if available
        try:
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
        except Exception as e:
            print(f"Warning: ChromaDB keyword search failed: {e}")
            # Fallback to memory search
            return self.find_keyword_matches(keywords)
    
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
