from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import BaseRetriever
from typing import Optional, Dict, Any
import os
import logging

logger = logging.getLogger(__name__)

class RAGPromptBuilder:
    """Handles building and configuring RAG prompts."""
    
    def build_prompt(self) -> PromptTemplate:
        """Build the RAG prompt template."""
        prompt_template = """You are a helpful AI assistant specializing in special education, learning disabilities, learning design and IEPs. Use the following pieces of context to answer the question. 
If the context doesn't contain all the information needed, you can:
1. Use the relevant parts of the context that are available
2. Combine it with your general knowledge about education and IEPs
3. Clearly indicate which parts of your response are from the context and which are general knowledge."

Context:
{context}

Question: {question}

Please provide a detailed answer citing specific information from the context when available:"""


        return PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

class LLMBuilder:
    """Handles LLM configuration and initialization."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def build_llm(self,
                  model_name: str,
                  temperature: float,
                  max_tokens: int) -> Optional[ChatOpenAI]:
        """Build and configure the LLM."""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
                
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                openai_api_key=api_key
            )
        except Exception as e:
            self.logger.error(f"Error building LLM: {e}")
            return None

class RetrieverBuilder:
    """Handles retriever configuration and setup."""
    
    def build_retriever(self,
                       vectorstore: BaseRetriever,
                       k_documents: int) -> BaseRetriever:
        """Build and configure the retriever."""
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": k_documents,
                "fetch_k": k_documents * 2,
                "score_threshold": None
            }
        )

class RAGChainBuilder:
    """Handles RAG chain construction and verification."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.prompt_builder = RAGPromptBuilder()
        self.llm_builder = LLMBuilder()
        self.retriever_builder = RetrieverBuilder()
    
    def build_chain(self,
                    vectorstore: BaseRetriever,
                    model_name: str,
                    temperature: float,
                    max_tokens: int,
                    k_documents: int) -> Optional[RetrievalQA]:
        """Build and verify the RAG chain."""
        try:
            # Build components
            prompt = self.prompt_builder.build_prompt()
            llm = self.llm_builder.build_llm(model_name, temperature, max_tokens)
            retriever = self.retriever_builder.build_retriever(vectorstore, k_documents)
            
            if not llm:
                return None
            
            # Build chain
            chain = self._create_chain(llm, retriever, prompt)
            
            # Verify chain
            if self._verify_chain(chain):
                return chain
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error building RAG chain: {e}")
            return None
    
    def _create_chain(self,
                     llm: ChatOpenAI,
                     retriever: BaseRetriever,
                     prompt: PromptTemplate) -> RetrievalQA:
        """Create the RAG chain with components."""
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": prompt,
                "verbose": True,
                "document_separator": "\n\n"
            }
        )
    
    def _verify_chain(self, chain: RetrievalQA) -> bool:
        """Verify chain functionality."""
        try:
            test_response = chain({"query": "test"})
            return (
                isinstance(test_response, dict) and
                "result" in test_response
            )
        except Exception as e:
            self.logger.error(f"Chain verification failed: {e}")
            return False

def build_rag_chain(
    vectorstore: BaseRetriever,
    model_name: str = "o1-mini",
    temperature: float = 0,
    max_tokens: int = 2048,
    k_documents: int = 4
) -> Optional[RetrievalQA]:
    """Build an optimized RAG chain with configurable parameters."""
    chain_builder = RAGChainBuilder()
    return chain_builder.build_chain(
        vectorstore=vectorstore,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        k_documents=k_documents
    )
