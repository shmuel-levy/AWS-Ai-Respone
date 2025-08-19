"""
Document Processor for RAG System
Handles automatic document segmentation into 18 parts with semantic analysis
"""
import os
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib
from sentence_transformers import SentenceTransformer
import numpy as np
import PyPDF2
import pdfplumber
from config import Config


class DocumentProcessor:
    """
    Handles document processing, segmentation, and semantic analysis.
    
    This class is responsible for:
    - Loading documents from various formats (PDF, TXT)
    - Segmenting documents into 18 semantically meaningful chunks
    - Extracting keywords and metadata
    - Managing document change detection
    """
    
    def __init__(self):
        """Initialize the document processor with embedding model"""
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.document_chunks = []
        self.chunk_embeddings = []
        self.document_hash = None
        
    def process_document(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Main method to process a document from file to segmented chunks.
        
        Args:
            file_path: Path to the document file (PDF or TXT)
            
        Returns:
            List of document chunks with metadata
            
        Raises:
            FileNotFoundError: If document file doesn't exist
            ValueError: If document content cannot be extracted
        """
        # Load document content
        content = self._extract_document_content(file_path)
        
        # Store document hash for change detection
        self.document_hash = self._calculate_content_hash(content)
        
        # Segment document into chunks
        self.document_chunks = self._create_semantic_chunks(content)
        
        return self.document_chunks
    
    def _extract_document_content(self, file_path: str) -> str:
        """
        Extract text content from document file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Document content as string
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If content cannot be extracted
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        # Handle different file types
        if file_path.suffix.lower() == '.pdf':
            return self._extract_pdf_content(file_path)
        else:
            return self._extract_text_content(file_path)
    
    def _extract_pdf_content(self, file_path: Path) -> str:
        """
        Extract text content from PDF file using multiple methods.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            PDF content as string
            
        Raises:
            ValueError: If content cannot be extracted
        """
        content = ""
        
        # Try pdfplumber first (better text extraction)
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        content += page_text + "\n"
        except Exception as e:
            print(f"pdfplumber failed, trying PyPDF2: {e}")
            
            # Fallback to PyPDF2
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            content += page_text + "\n"
            except Exception as e2:
                raise ValueError(f"Failed to extract PDF content: {e2}")
        
        if not content.strip():
            raise ValueError(f"Could not extract text from PDF: {file_path}")
        
        return content
    
    def _extract_text_content(self, file_path: Path) -> str:
        """
        Extract content from text file.
        
        Args:
            file_path: Path to text file
            
        Returns:
            Text content as string
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def _create_semantic_chunks(self, content: str) -> List[Dict[str, Any]]:
        """
        Create semantically meaningful document chunks.
        
        Args:
            content: Document content
            
        Returns:
            List of document chunks with metadata
        """
        # Clean and preprocess content
        cleaned_content = self._clean_document_content(content)
        
        # Split into sentences
        sentences = self._extract_sentences(cleaned_content)
        
        # Create initial chunks based on size
        initial_chunks = self._create_size_based_chunks(sentences)
        
        # Optimize chunks to exactly 18 parts using semantic similarity
        optimized_chunks = self._optimize_chunk_count(initial_chunks)
        
        # Create chunk metadata
        return self._create_chunk_metadata(optimized_chunks)
    
    def _clean_document_content(self, content: str) -> str:
        """
        Clean and preprocess document content.
        
        Args:
            content: Raw document content
            
        Returns:
            Cleaned content
        """
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove special characters but keep punctuation
        content = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', content)
        
        return content.strip()
    
    def _extract_sentences(self, content: str) -> List[str]:
        """
        Extract sentences from document content.
        
        Args:
            content: Document content
            
        Returns:
            List of sentences
        """
        # Split by sentence endings
        sentences = re.split(r'[.!?]+', content)
        
        # Filter out empty sentences
        return [s.strip() for s in sentences if s.strip()]
    
    def _create_size_based_chunks(self, sentences: List[str]) -> List[str]:
        """
        Create initial chunks based on size constraints.
        
        Args:
            sentences: List of sentences
            
        Returns:
            List of initial chunks
        """
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # If adding this sentence would exceed chunk size, start new chunk
            if current_size + sentence_size > Config.CHUNK_SIZE and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Add the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _optimize_chunk_count(self, initial_chunks: List[str]) -> List[str]:
        """
        Optimize chunks to exactly 18 parts using semantic similarity.
        
        Args:
            initial_chunks: List of initial chunks
            
        Returns:
            List of optimized chunks
        """
        if len(initial_chunks) <= Config.MAX_CHUNKS:
            # If we have fewer chunks than needed, split larger chunks
            return self._expand_chunks_to_target_count(initial_chunks)
        else:
            # If we have more chunks than needed, merge similar chunks
            return self._merge_chunks_to_target_count(initial_chunks)
    
    def _expand_chunks_to_target_count(self, chunks: List[str]) -> List[str]:
        """
        Expand chunks to exactly 18 parts by splitting large chunks.
        
        Args:
            chunks: List of chunks
            
        Returns:
            List of expanded chunks
        """
        result = []
        
        for chunk in chunks:
            if len(result) >= Config.MAX_CHUNKS:
                break
                
            # If chunk is large enough, split it
            if len(chunk) > Config.CHUNK_SIZE * 1.5:
                sub_chunks = self._split_large_chunk(chunk)
                for sub_chunk in sub_chunks:
                    if len(result) < Config.MAX_CHUNKS:
                        result.append(sub_chunk)
            else:
                result.append(chunk)
        
        # If still need more chunks, split remaining large chunks
        while len(result) < Config.MAX_CHUNKS:
            # Find the largest chunk and split it
            largest_chunk_idx = max(range(len(result)), key=lambda i: len(result[i]))
            largest_chunk = result[largest_chunk_idx]
            
            if len(largest_chunk) > Config.CHUNK_SIZE:
                sub_chunks = self._split_large_chunk(largest_chunk)
                result.pop(largest_chunk_idx)
                result.extend(sub_chunks[:Config.MAX_CHUNKS - len(result)])
            else:
                break
        
        return result[:Config.MAX_CHUNKS]
    
    def _merge_chunks_to_target_count(self, chunks: List[str]) -> List[str]:
        """
        Merge chunks to exactly 18 parts using semantic similarity clustering.
        
        Args:
            chunks: List of chunks
            
        Returns:
            List of merged chunks
        """
        # Calculate embeddings for all chunks
        embeddings = self.embedding_model.encode(chunks)
        
        # Use clustering to group similar chunks
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=Config.MAX_CHUNKS, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Group chunks by cluster
        clustered_chunks = [[] for _ in range(Config.MAX_CHUNKS)]
        for i, label in enumerate(cluster_labels):
            clustered_chunks[label].append(chunks[i])
        
        # Merge chunks within each cluster
        result = []
        for cluster in clustered_chunks:
            if cluster:
                merged_chunk = ' '.join(cluster)
                result.append(merged_chunk)
        
        return result
    
    def _split_large_chunk(self, chunk: str) -> List[str]:
        """
        Split a large chunk into smaller parts.
        
        Args:
            chunk: Large chunk to split
            
        Returns:
            List of smaller chunks
        """
        sentences = self._extract_sentences(chunk)
        sub_chunks = []
        current_sub_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > Config.CHUNK_SIZE and current_sub_chunk:
                sub_chunks.append(' '.join(current_sub_chunk))
                current_sub_chunk = [sentence]
                current_size = sentence_size
            else:
                current_sub_chunk.append(sentence)
                current_size += sentence_size
        
        if current_sub_chunk:
            sub_chunks.append(' '.join(current_sub_chunk))
        
        return sub_chunks
    
    def _create_chunk_metadata(self, chunks: List[str]) -> List[Dict[str, Any]]:
        """
        Create metadata for document chunks.
        
        Args:
            chunks: List of chunk content
            
        Returns:
            List of chunks with metadata
        """
        chunk_data = []
        for i, chunk in enumerate(chunks):
            chunk_info = {
                'id': i,
                'content': chunk,
                'length': len(chunk),
                'word_count': len(chunk.split()),
                'semantic_keywords': self._extract_semantic_keywords(chunk)
            }
            chunk_data.append(chunk_info)
        
        return chunk_data
    
    def _extract_semantic_keywords(self, text: str) -> List[str]:
        """
        Extract semantic keywords from text.
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            List of keywords
        """
        # Simple keyword extraction - can be enhanced with more sophisticated methods
        words = text.lower().split()
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        }
        keywords = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Return top 5 most frequent keywords
        from collections import Counter
        return [word for word, _ in Counter(keywords).most_common(5)]
    
    def _calculate_content_hash(self, content: str) -> str:
        """
        Calculate hash of document content for change detection.
        
        Args:
            content: Document content
            
        Returns:
            MD5 hash of content
        """
        return hashlib.md5(content.encode()).hexdigest()
    
    def is_document_modified(self, content: str) -> bool:
        """
        Check if document content has been modified.
        
        Args:
            content: Current document content
            
        Returns:
            True if document has changed, False otherwise
        """
        current_hash = self._calculate_content_hash(content)
        return current_hash != self.document_hash
    
    def get_chunk_embeddings(self) -> List[List[float]]:
        """
        Get embeddings for all document chunks.
        
        Returns:
            List of chunk embeddings
        """
        if not self.document_chunks:
            return []
        
        chunk_texts = [chunk['content'] for chunk in self.document_chunks]
        return self.embedding_model.encode(chunk_texts).tolist()
    
    def save_chunks_to_files(self, output_dir: Optional[str] = None) -> None:
        """
        Save processed chunks to individual files.
        
        Args:
            output_dir: Directory to save chunks (defaults to Config.DATA_DIR)
        """
        if output_dir is None:
            output_dir = Config.DATA_DIR
        
        os.makedirs(output_dir, exist_ok=True)
        
        for chunk in self.document_chunks:
            filename = f"chunk_{chunk['id']:02d}.txt"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(chunk['content'])
    
    def load_chunks_from_files(self, input_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load processed chunks from individual files.
        
        Args:
            input_dir: Directory to load chunks from (defaults to Config.DATA_DIR)
            
        Returns:
            List of loaded chunks with metadata
        """
        if input_dir is None:
            input_dir = Config.DATA_DIR
        
        if not os.path.exists(input_dir):
            return []
        
        chunks = []
        chunk_files = sorted([f for f in os.listdir(input_dir) if f.startswith('chunk_')])
        
        for filename in chunk_files:
            filepath = os.path.join(input_dir, filename)
            chunk_id = int(filename.split('_')[1].split('.')[0])
            
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            chunk_data = {
                'id': chunk_id,
                'content': content,
                'length': len(content),
                'word_count': len(content.split()),
                'semantic_keywords': self._extract_semantic_keywords(content)
            }
            chunks.append(chunk_data)
        
        self.document_chunks = chunks
        return chunks
