from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from typing import List, Optional
from langchain.schema import Document
import os
import logging
from utils.logging_config import logger

logger = logging.getLogger(__name__)

class TextChunkProcessor:
    """Handles text chunking and preprocessing."""
    
    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size
        self.overlap = int(chunk_size * 0.2)  # 20% overlap
        self.logger = logging.getLogger(__name__)
    
    def create_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Create configured text splitter."""
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""]
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        try:
            splitter = self.create_text_splitter()
            texts = splitter.split_documents(documents)
            self.logger.debug(f"Created {len(texts)} text chunks")
            
            if texts:
                self.logger.debug(f"Sample chunk: {texts[0].page_content[:200]}")
                
            return texts
            
        except Exception as e:
            self.logger.error(f"Error splitting documents: {e}")
            return []

class EmbeddingBuilder:
    """Handles embedding model configuration and initialization."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def build_embeddings(self) -> Optional[OpenAIEmbeddings]:
        """Build and configure embeddings."""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
                
            return OpenAIEmbeddings(
                model="text-embedding-ada-002",
                openai_api_key=api_key
            )
        except Exception as e:
            self.logger.error(f"Error building embeddings: {e}")
            return None

class FAISSIndexManager:
    """Handles FAISS index creation and management."""
    
    def __init__(self, persist_directory: str):
        self.persist_directory = persist_directory
        self.chunk_processor = TextChunkProcessor()
        self.embedding_builder = EmbeddingBuilder()
        self.logger = logging.getLogger(__name__)
    
    def build_index(self, documents: List[Document], chunk_size: int = 1000) -> Optional[FAISS]:
        """Build an optimized FAISS index from documents."""
        try:
            self.logger.info(f"Building index from {len(documents)} documents")
            
            # Process text chunks
            texts = self.chunk_processor.split_documents(documents)
            if not texts:
                return None
            
            # Build embeddings
            embeddings = self.embedding_builder.build_embeddings()
            if not embeddings:
                return None
            
            # Create and save index
            return self._create_and_save_index(texts, embeddings)
            
        except Exception as e:
            self.logger.error(f"Error building FAISS index: {e}")
            return None
    
    def _create_and_save_index(self, texts: List[Document], embeddings: OpenAIEmbeddings) -> Optional[FAISS]:
        """Create and persist FAISS index."""
        try:
            vectorstore = FAISS.from_documents(texts, embeddings)
            
            if self.persist_directory:
                os.makedirs(self.persist_directory, exist_ok=True)
                vectorstore.save_local(self.persist_directory)
                self.logger.info(f"Saved index to {self.persist_directory}")
            
            return vectorstore
            
        except Exception as e:
            self.logger.error(f"Error creating/saving index: {e}")
            return None
    
    def load_index(self, embeddings: Optional[OpenAIEmbeddings] = None) -> Optional[FAISS]:
        """Load FAISS index from disk."""
        try:
            if not os.path.exists(self.persist_directory):
                self.logger.error(f"Index directory not found: {self.persist_directory}")
                return None
                
            if not embeddings:
                embeddings = self.embedding_builder.build_embeddings()
                if not embeddings:
                    return None
            
            return FAISS.load_local(self.persist_directory, embeddings)
            
        except Exception as e:
            self.logger.error(f"Error loading index: {e}")
            return None

def build_faiss_index(documents: List[Document], persist_directory: str) -> Optional[FAISS]:
    """Build and save FAISS index."""
    try:
        # Initialize embeddings with explicit API key
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Create and save the FAISS index
        vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=embeddings
        )
        
        # Persist index
        vectorstore.save_local(persist_directory)
        return vectorstore
        
    except Exception as e:
        logger.error(f"Error creating/saving index: {str(e)}")
        return None

def load_faiss_index(persist_directory: str) -> Optional[FAISS]:
    """Load a FAISS index from disk."""
    manager = FAISSIndexManager(persist_directory)
    return manager.load_index()

