# Update your training_data_processor.py to fix OpenAI v1.0+ compatibility

import os
import logging
from typing import List, Dict, Any, Optional
import pickle
import numpy as np
import faiss
from pathlib import Path

logger = logging.getLogger(__name__)

class TrainingDataProcessor:
    """Handles document chunking, embedding, and FAISS index creation"""
    
    def __init__(self):
        self.chunk_size = 512
        self.chunk_overlap = 50
        self.embedding_model = "text-embedding-ada-002"
        self._openai_client = None
        self._sentence_transformer = None
        
    def _get_openai_client(self):
        """Get OpenAI client using v1.0+ API"""
        if self._openai_client is None:
            try:
                import openai
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    # Use new v1.0+ client initialization
                    self._openai_client = openai.OpenAI(api_key=api_key)
                    logger.info("OpenAI client initialized (v1.0+)")
                else:
                    logger.warning("No OpenAI API key found")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
        return self._openai_client
    
    def _get_sentence_transformer(self):
        """Get sentence transformer as fallback"""
        if self._sentence_transformer is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Sentence transformer initialized as fallback")
            except Exception as e:
                logger.error(f"Failed to initialize sentence transformer: {e}")
        return self._sentence_transformer
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        if not text or not text.strip():
            return []
        
        # Simple sentence-aware chunking
        sentences = text.replace('\n', ' ').split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Add sentence to current chunk
            test_chunk = current_chunk + (". " if current_chunk else "") + sentence
            
            # If chunk would be too long, start a new one
            if len(test_chunk) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                overlap_words = current_chunk.split()[-10:]  # Last 10 words for overlap
                current_chunk = " ".join(overlap_words) + ". " + sentence
            else:
                current_chunk = test_chunk
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if len(chunk.strip()) > 20]  # Filter out very short chunks
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for text chunks using OpenAI v1.0+ API or fallback"""
        if not texts:
            return np.array([])
        
        # Try OpenAI first
        openai_client = self._get_openai_client()
        if openai_client:
            try:
                # Use new v1.0+ API format
                response = openai_client.embeddings.create(
                    input=texts,
                    model=self.embedding_model
                )
                embeddings = [item.embedding for item in response.data]
                logger.info(f"Created {len(embeddings)} OpenAI embeddings")
                return np.array(embeddings)
            except Exception as e:
                logger.error(f"OpenAI embedding failed: {e}")
        
        # Fallback to sentence transformer
        sentence_transformer = self._get_sentence_transformer()
        if sentence_transformer:
            try:
                embeddings = sentence_transformer.encode(texts)
                logger.info(f"Created {len(embeddings)} sentence transformer embeddings")
                return embeddings
            except Exception as e:
                logger.error(f"Sentence transformer embedding failed: {e}")
        
        # Last resort: random embeddings (for testing only)
        logger.warning("Using random embeddings - install sentence-transformers or set OPENAI_API_KEY")
        return np.random.random((len(texts), 384))  # 384 dimensions for compatibility
    
    def create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Create and populate FAISS index"""
        if embeddings.size == 0:
            raise ValueError("No embeddings provided")
        
        dimension = embeddings.shape[1]
        
        # Use IndexFlatL2 for small datasets, IndexIVFFlat for larger ones
        if len(embeddings) < 1000:
            index = faiss.IndexFlatL2(dimension)
        else:
            # For larger datasets, use IVF with clustering
            nlist = min(100, len(embeddings) // 10)  # Number of clusters
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            
            # Train the index
            index.train(embeddings.astype(np.float32))
        
        # Add embeddings to index
        index.add(embeddings.astype(np.float32))
        logger.info(f"Created FAISS index with {index.ntotal} vectors, dimension {dimension}")
        return index
    
    def prepare_training_data(self, texts: List[str], training_id: str) -> Dict[str, Any]:
        """
        Complete pipeline: chunk texts, create embeddings, build FAISS index
        """
        try:
            # Combine all texts
            combined_text = "\n\n".join(texts)
            logger.info(f"Processing {len(combined_text)} characters of text")
            
            # Chunk the text
            chunks = self.chunk_text(combined_text)
            if not chunks:
                raise ValueError("No valid text chunks created")
            
            logger.info(f"Created {len(chunks)} text chunks")
            
            # Create embeddings
            embeddings = self.create_embeddings(chunks)
            if embeddings.size == 0:
                raise ValueError("No embeddings created")
            
            # Create FAISS index
            index = self.create_faiss_index(embeddings)
            
            # Save to disk
            from config import MODELS_DIR
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            
            index_dir = MODELS_DIR / "indexes" / training_id
            index_dir.mkdir(parents=True, exist_ok=True)
            
            index_path = index_dir / "faiss.index"
            metadata_path = index_dir / "chunks_metadata.pkl"
            
            # Save FAISS index
            faiss.write_index(index, str(index_path))
            
            # Save chunks metadata
            with open(metadata_path, "wb") as f:
                pickle.dump(chunks, f)
            
            logger.info(f"Saved index to {index_path} and metadata to {metadata_path}")
            
            return {
                "index_path": str(index_path),
                "metadata_path": str(metadata_path),
                "num_chunks": len(chunks),
                "embedding_dimension": embeddings.shape[1],
                "training_id": training_id
            }
            
        except Exception as e:
            logger.error(f"Failed to prepare training data: {e}")
            raise
    
    def query_index(self, query: str, training_id: str, top_k: int = 5) -> List[str]:
        """Query the FAISS index for relevant chunks"""
        try:
            from config import MODELS_DIR
            base_path = Path(MODELS_DIR) / "indexes" / training_id
            index_path = base_path / "faiss.index"
            metadata_path = base_path / "chunks_metadata.pkl"
            
            if not index_path.exists() or not metadata_path.exists():
                logger.warning(f"Index or metadata not found for {training_id}")
                return []
            
            # Load index and metadata
            index = faiss.read_index(str(index_path))
            with open(metadata_path, "rb") as f:
                chunks = pickle.load(f)
            
            # Create query embedding
            query_embedding = self.create_embeddings([query])
            if query_embedding.size == 0:
                return []
            
            # Search index
            scores, indices = index.search(query_embedding.astype(np.float32), top_k)
            
            # Return relevant chunks
            relevant_chunks = []
            for idx in indices[0]:
                if 0 <= idx < len(chunks):
                    relevant_chunks.append(chunks[idx])
            
            logger.info(f"Found {len(relevant_chunks)} relevant chunks for query")
            return relevant_chunks
            
        except Exception as e:
            logger.error(f"Failed to query index: {e}")
            return []
    
    def cleanup(self):
        """Clean up resources"""
        self._openai_client = None
        self._sentence_transformer = None