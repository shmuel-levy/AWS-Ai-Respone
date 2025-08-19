"""
Document Processor for RAG System
Handles automatic document segmentation into 18 parts with semantic analysis
"""
import os
import re
from typing import List, Dict, Any
from pathlib import Path
import hashlib
from sentence_transformers import SentenceTransformer
import numpy as np
import PyPDF2
import pdfplumber
from config import Config


class DocumentProcessor:
    """Handles document processing and segmentation"""
    
    def __init__(self):
        """Initialize the document processor"""
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chunks = []
        self.chunk_embeddings = []
        self.document_hash = None
        
    def load_document(self, file_path: str) -> str:
        """
        Load document from file (supports PDF and text files)
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Document content as string
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        # Check file extension
        if file_path.suffix.lower() == '.pdf':
            content = self._load_pdf(file_path)
        else:
            # Assume text file
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
        # Store document hash for change detection
        self.document_hash = self._calculate_hash(content)
        
        return content
    
    def _load_pdf(self, file_path: Path) -> str:
        """
        Load content from PDF file
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            PDF content as string
        """
        content = ""
        
        try:
            # Try using pdfplumber first (better text extraction)
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        content += page_text + "\n"
        except Exception as e:
            print(f"pdfplumber failed, trying PyPDF2: {e}")
            
            # Fallback to PyPDF2
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        content += page_text + "\n"
        
        if not content.strip():
            raise ValueError(f"Could not extract text from PDF: {file_path}")
        
        return content
    
    def segment_document(self, content: str) -> List[Dict[str, Any]]:
        """
        Segment document into 18 parts using semantic analysis
        
        Args:
            content: Document content
            
        Returns:
            List of document chunks with metadata
        """
        # Clean and preprocess content
        cleaned_content = self._preprocess_content(content)
        
        # Split into sentences
        sentences = self._split_into_sentences(cleaned_content)
        
        # Create initial chunks based on size
        initial_chunks = self._create_initial_chunks(sentences)
        
        # Optimize chunks to exactly 18 parts using semantic similarity
        optimized_chunks = self._optimize_to_18_chunks(initial_chunks)
        
        # Create chunk metadata
        self.chunks = []
        for i, chunk in enumerate(optimized_chunks):
            chunk_data = {
                'id': i,
                'content': chunk,
                'length': len(chunk),
                'word_count': len(chunk.split()),
                'semantic_keywords': self._extract_keywords(chunk)
            }
            self.chunks.append(chunk_data)
        
        return self.chunks
    
    def _preprocess_content(self, content: str) -> str:
        """Clean and preprocess document content"""
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove special characters but keep punctuation
        content = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', content)
        
        return content.strip()
    
    def _split_into_sentences(self, content: str) -> List[str]:
        """Split content into sentences"""
        # Split by sentence endings
        sentences = re.split(r'[.!?]+', content)
        
        # Filter out empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _create_initial_chunks(self, sentences: List[str]) -> List[str]:
        """Create initial chunks based on size constraints"""
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
    
    def _optimize_to_18_chunks(self, initial_chunks: List[str]) -> List[str]:
        """Optimize chunks to exactly 18 parts using semantic similarity"""
        if len(initial_chunks) <= Config.MAX_CHUNKS:
            # If we have fewer chunks than needed, split larger chunks
            return self._expand_to_18_chunks(initial_chunks)
        else:
            # If we have more chunks than needed, merge similar chunks
            return self._merge_to_18_chunks(initial_chunks)
    
    def _expand_to_18_chunks(self, chunks: List[str]) -> List[str]:
        """Expand chunks to exactly 18 parts"""
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
    
    def _merge_to_18_chunks(self, chunks: List[str]) -> List[str]:
        """Merge chunks to exactly 18 parts using semantic similarity"""
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
        """Split a large chunk into smaller parts"""
        sentences = self._split_into_sentences(chunk)
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
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract semantic keywords from text"""
        # Simple keyword extraction - can be enhanced with more sophisticated methods
        words = text.lower().split()
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Return top 5 most frequent keywords
        from collections import Counter
        return [word for word, _ in Counter(keywords).most_common(5)]
    
    def _calculate_hash(self, content: str) -> str:
        """Calculate hash of document content for change detection"""
        return hashlib.md5(content.encode()).hexdigest()
    
    def has_changed(self, content: str) -> bool:
        """Check if document content has changed"""
        current_hash = self._calculate_hash(content)
        return current_hash != self.document_hash
    
    def get_chunk_embeddings(self) -> List[List[float]]:
        """Get embeddings for all chunks"""
        if not self.chunks:
            return []
        
        chunk_texts = [chunk['content'] for chunk in self.chunks]
        return self.embedding_model.encode(chunk_texts).tolist()
    
    def save_chunks(self, output_dir: str = None):
        """Save processed chunks to files"""
        if output_dir is None:
            output_dir = Config.DATA_DIR
        
        os.makedirs(output_dir, exist_ok=True)
        
        for chunk in self.chunks:
            filename = f"chunk_{chunk['id']:02d}.txt"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(chunk['content'])
    
    def load_chunks(self, input_dir: str = None) -> List[Dict[str, Any]]:
        """Load processed chunks from files"""
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
                'semantic_keywords': self._extract_keywords(content)
            }
            chunks.append(chunk_data)
        
        self.chunks = chunks
        return chunks
