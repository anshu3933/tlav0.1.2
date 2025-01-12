from typing import List, Optional, Dict, Any
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
import os
import logging
from utils.logging_config import logger

# Local imports
from loaders import load_documents
from chains import LLMBuilder, RAGPromptBuilder
from embeddings import build_faiss_index
from dspy_pipeline import process_iep_to_lesson_plans

class FileManager:
    """Manages file and directory operations."""
    def __init__(self, data_dir: str, index_dir: str):
        self.data_dir = data_dir
        self.index_dir = index_dir
        
    def ensure_directories(self) -> bool:
        """Create necessary directories if they don't exist."""
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            os.makedirs(self.index_dir, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Error creating directories: {e}")
            return False
            
    def get_document_paths(self) -> List[str]:
        """Get paths of all documents in data directory."""
        try:
            files = []
            for file in os.listdir(self.data_dir):
                if file.endswith(('.pdf', '.docx', '.txt')):
                    files.append(os.path.join(self.data_dir, file))
            return files
        except Exception as e:
            logger.error(f"Error getting document paths: {e}")
            return []

class RAGChainBuilder:
    """Builds RAG chain components."""
    def build_chain(self, 
                   llm: ChatOpenAI, 
                   retriever: Any,
                   prompt: Any) -> Optional[RetrievalQA]:
        """Build the RAG chain."""
        try:
            return RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": prompt}
            )
        except Exception as e:
            logger.error(f"Error building RAG chain: {e}")
            return None

class SystemInitializer:
    """Handles system initialization and setup."""
    
    def __init__(self, data_dir: str = "data", index_dir: str = "models/faiss_index"):
        self.file_manager = FileManager(data_dir, index_dir)
        self.logger = logger
        self.llm_builder = LLMBuilder()
        self.rag_builder = RAGChainBuilder()
    
    def initialize(self, use_dspy: bool = False) -> Optional[Dict[str, Any]]:
        """Initialize the system by loading documents and building the index."""
        try:
            if not self._ensure_environment():
                return None
                
            documents = self._load_and_process_documents(use_dspy)
            if not documents:
                return None
                
            return self._build_qa_system(documents)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize system: {e}")
            return None
            
    def _ensure_environment(self) -> bool:
        """Ensure all required environment variables and directories exist."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            self.logger.error("OPENAI_API_KEY environment variable is not set")
            return False
            
        # Test embeddings initialization
        try:
            embeddings = OpenAIEmbeddings(
                model="text-embedding-ada-002",
                api_key=api_key
            )
            embeddings.embed_query("test")
        except Exception as e:
            self.logger.error(f"Failed to validate OpenAI API key: {e}")
            return False
            
        return self.file_manager.ensure_directories()
        
    def _load_and_process_documents(self, use_dspy: bool) -> Optional[List[Document]]:
        """Load and process documents."""
        file_paths = self.file_manager.get_document_paths()
        if not file_paths:
            self.logger.warning(f"No documents found in {self.file_manager.data_dir}")
            return None
            
        documents = load_documents(file_paths)
        if not documents:
            return None
            
        if use_dspy:
            documents = process_iep_to_lesson_plans(documents)
            
        return documents
        
    def _build_qa_system(self, documents: List[Document]) -> Optional[Dict[str, Any]]:
        """Build the QA system components."""
        try:
            # Build vector store
            vectorstore = build_faiss_index(
                documents=documents,
                persist_directory=self.file_manager.index_dir
            )
            if not vectorstore:
                self.logger.error("Failed to build vector store")
                return None
                
            # Build LLM
            llm = self.llm_builder.build_llm(
                model_name="gpt-4",
                temperature=0.7,
                max_tokens=2000
            )
            if not llm:
                return None
                
            # Build RAG chain
            chain = self.rag_builder.build_chain(
                llm=llm,
                retriever=vectorstore.as_retriever(),
                prompt=RAGPromptBuilder().build_prompt()
            )
            if not chain:
                return None
                
            return {
                "documents": documents,
                "vectorstore": vectorstore,
                "chain": chain
            }
            
        except Exception as e:
            self.logger.error(f"Error building QA system: {e}")
            return None

def main():
    """Main entry point."""
    try:
        data_dir = os.getenv("DATA_DIR", "data")
        index_dir = os.getenv("INDEX_DIR", "models/faiss_index")
        
        initializer = SystemInitializer(
            data_dir=data_dir,
            index_dir=index_dir
        )
        
        system = initializer.initialize(use_dspy=False)
        if system:
            logger.info("System initialized successfully")
            return system
        else:
            logger.error("Failed to initialize system")
            return None
            
    except Exception as e:
        logger.error(f"Error in main: {e}")
        return None

def initialize_system(data_dir: str = "data", use_dspy: bool = False) -> Optional[Dict[str, Any]]:
    """Initialize the system components."""
    try:
        initializer = SystemInitializer(
            data_dir=data_dir,
            index_dir="models/faiss_index"
        )
        return initializer.initialize(use_dspy=use_dspy)
        
    except Exception as e:
        logger.error(f"Error initializing system: {e}")
        return None

if __name__ == "__main__":
    main()