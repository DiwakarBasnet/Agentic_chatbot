import os
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from transformers import AutoTokenizer
from config import config
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_DIR = config.models_dir


class Encoder:
    """Handles document embedding functionality"""
    
    def __init__(self, embedding_config):
        self.model_name = embedding_config.model_name
        self.device = embedding_config.device
        
        try:
            self.embedding_function = HuggingFaceEmbeddings(
                model_name=self.model_name,
                cache_folder=CACHE_DIR,
                model_kwargs={"device": self.device},
                encode_kwargs={"normalize_embeddings": embedding_config.normalize_embeddings}
            )
            logger.info(f"Successfully loaded embedding model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self.embedding_function.embed_documents(texts)
    
    def get_embedding(self, text: str) -> List[float]:
        return self.embedding_function.embed_query(text)


class FaissDb:
    """FAISS database with search capabilities"""
    
    def __init__(self, docs, embedding_function, distance_strategy=DistanceStrategy.COSINE):
        try:
            self.db = FAISS.from_documents(
                docs, 
                embedding_function, 
                distance_strategy=distance_strategy
            )
            logger.info(f"Successfully created FAISS database with {len(docs)} documents")
        except Exception as e:
            logger.error(f"Failed to create FAISS database: {e}")
            raise
    
    def similarity_search(self, question: str, k: int = 3) -> str:
        """
        Perform similarity search and return formatted context
        
        Args:
            question: Query string
            k: Number of context to retrieve
            
        Returns:
            Formatted context string
        """
        try:
            retrieved_docs = self.db.similarity_search(question, k=k)
            context = self._format_context(retrieved_docs)
            logger.info(f"Retrieved {len(retrieved_docs)} contexts for query")
            return context
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return ""
    
    def similarity_search_with_scores(self, question: str, k: int = 3) -> List[tuple]:
        """
        Perform similarity search with relevance scores
        
        Args:
            question: Query string
            k: Number of context to retrieve
            
        Returns:
            List of (document, score) tuples
        """
        try:
            return self.db.similarity_search_with_score(question, k=k)
        except Exception as e:
            logger.error(f"Error during similarity search with scores: {e}")
            return []
    
    def _format_context(self, docs) -> str:
        """Format retrieved documents into context string"""
        if not docs:
            return ""
        
        context_parts = []
        for i, doc in enumerate(docs, 1):
            metadata = doc.metadata
            source_info = ""
            if 'source' in metadata:
                source_info = f" (Source: {os.path.basename(metadata['source'])})"
            if 'page' in metadata:
                source_info += f" (Page: {metadata['page']})"
            context_parts.append(f"Document {i}{source_info}:\n{doc.page_content}")
        
        return "\n\n".join(context_parts)
    
    def get_database_stats(self) -> dict:
        """Get statistics about the database"""
        return {
            'total_documents': self.db.index.ntotal,
            'embedding_dimension': self.db.index.d
        }


class DocumentProcessor:  
    def __init__(self):
        rag_config = config.get_rag_config()
        self.chunk_size = rag_config.chunk_size
        self.chunk_overlap = int(self.chunk_size * rag_config.chunk_overlap_ratio)
        
    def load_and_split_pdfs(self, file_paths: List[str]) -> List:
        """
        Load and split PDF documents into chunks
        
        Args:
            file_paths: List of PDF file paths
            
        Returns:
            List of document chunks
        """
        if not file_paths:
            logger.warning("No file paths provided")
            return []
        
        # Load documents
        pages = []
        successful_loads = 0
        
        for file_path in file_paths:
            try:
                if not os.path.exists(file_path):
                    logger.warning(f"File not found: {file_path}")
                    continue
                    
                loader = PyPDFLoader(file_path)
                file_pages = loader.load()
                pages.extend(file_pages)
                successful_loads += 1
                logger.info(f"Successfully loaded {len(file_pages)} pages from {file_path}")
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
        
        if not pages:
            logger.error("No documents were successfully loaded")
            return []
        
        # Split documents
        try:
            tokenizer = AutoTokenizer.from_pretrained(config.get_embedding_config().model_name)
            text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                tokenizer=tokenizer,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                strip_whitespace=True,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            docs = text_splitter.split_documents(pages)
            logger.info(f"Successfully split {len(pages)} pages into {len(docs)} chunks")
            return docs
            
        except Exception as e:
            logger.error(f"Failed to split documents: {e}")
            return []
    
    def get_document_stats(self, docs) -> dict:
        """Get statistics about processed documents"""
        if not docs:
            return {"total_chunks": 0, "average_chunk_size": 0}
        
        total_length = sum(len(doc.page_content) for doc in docs)
        return {
            "total_chunks": len(docs),
            "average_chunk_size": total_length // len(docs),
            "total_characters": total_length
        }


class RAGPipeline:
    """Complete RAG pipeline combining all components"""
    
    def __init__(self):
        self.encoder = Encoder(embedding_config=config.get_embedding_config())
        self.processor = DocumentProcessor()
        self.db = None
        
    def setup_database(self, file_paths: List[str]) -> bool:
        """
        Set up the RAG database from PDF files
        
        Args:
            file_paths: List of PDF file paths
            
        Returns:
            True if successful, False otherwise
        """
        try:
            docs = self.processor.load_and_split_pdfs(file_paths)
            if not docs:
                logger.error("No documents to process")
                return False
            distance_strategy = getattr(DistanceStrategy, config.get_rag_config().distance_strategy.upper(), DistanceStrategy.COSINE)
            self.db = FaissDb(docs=docs, embedding_function=self.encoder.embedding_function, distance_strategy=distance_strategy)
            logger.info("RAG pipeline setup completed successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to setup RAG pipeline: {e}")
            return False
    
    def search(self, query: str, k: int = 3) -> str:
        """
        Search for relevant context
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            Formatted context string
        """
        if not self.db:
            logger.warning("Database not initialized")
            return ""
        
        return self.db.similarity_search(query, k=k)
    
    def get_pipeline_stats(self) -> dict:
        """Get statistics about the RAG pipeline"""
        stats = {
            "encoder_model": self.encoder.model_name,
            "chunk_size": self.processor.chunk_size,
            "database_ready": self.db is not None
        }
        
        if self.db:
            stats.update(self.db.get_database_stats())
            
        return stats


def load_and_split_pdfs(file_paths: List[str], chunk_size: int = 256) -> List:
    """Load and split PDFs into chunks"""
    processor = DocumentProcessor(chunk_size=chunk_size)
    return processor.load_and_split_pdfs(file_paths)
