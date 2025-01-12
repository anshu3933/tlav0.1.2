from duckduckgo_search import DDGS
from typing import List, Dict, Any
from langchain.schema import Document
import logging
import uuid

logger = logging.getLogger(__name__)

class SearchAugmentedRetrieval:
    """Augments document retrieval with internet search results."""
    
    def __init__(self, vector_store, search_limit: int = 3):
        self.vector_store = vector_store
        self.search_limit = search_limit
        self.ddgs = DDGS()
    
    def _search_ddg(self, query: str) -> List[Dict[str, str]]:
        """Perform DuckDuckGo search."""
        try:
            results = self.ddgs.text(query, max_results=self.search_limit)
            return [
                {
                    'title': r['title'],
                    'link': r['link'],
                    'snippet': r['body']
                } for r in results
            ]
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return []
    
    def augment_context(self, 
                       query: str, 
                       doc_results: List[Document], 
                       search_weight: float = 0.3) -> List[Document]:
        """Combine document results with internet search results."""
        try:
            # Get DuckDuckGo search results
            search_results = self._search_ddg(query)
            
            # Convert search results to documents with unique IDs
            search_docs = [
                Document(
                    page_content=f"{r['title']}\n{r['snippet']}",
                    metadata={
                        'source': r['link'],
                        'type': 'web_search',
                        'title': r['title'],
                        'id': str(uuid.uuid4())
                    }
                ) for r in search_results
            ]
            
            # Add unique IDs to original documents if they don't have them
            for doc in doc_results:
                if 'id' not in doc.metadata:
                    doc.metadata['id'] = str(uuid.uuid4())
            
            # Combine and rerank results
            combined_docs = self._rerank_results(
                query=query,
                doc_results=doc_results,
                search_docs=search_docs,
                search_weight=search_weight
            )
            
            return combined_docs
            
        except Exception as e:
            logger.error(f"Failed to augment context: {e}")
            return doc_results
    
    def _rerank_results(self, 
                       query: str,
                       doc_results: List[Document],
                       search_docs: List[Document],
                       search_weight: float) -> List[Document]:
        """Rerank combined results based on relevance."""
        try:
            # Get relevance scores for all documents
            all_docs = doc_results + search_docs
            
            # Use vector store to get similarity scores
            query_embedding = self.vector_store.embedding_function.embed_query(query)
            
            # Score and rank documents
            scored_docs = []
            for doc in all_docs:
                doc_embedding = self.vector_store.embedding_function.embed_documents(
                    [doc.page_content]
                )[0]
                
                # Calculate similarity score
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                
                # Apply source weights
                final_score = similarity * (
                    search_weight if doc.metadata['type'] == 'web_search'
                    else (1 - search_weight)
                )
                
                scored_docs.append((doc, final_score))
            
            # Sort by score and return documents
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in scored_docs]
            
        except Exception as e:
            logger.error(f"Failed to rerank results: {e}")
            return doc_results + search_docs
    
    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import numpy as np
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)) 