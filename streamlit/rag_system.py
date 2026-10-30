"""
Retriever + Reader QA system: each retriever's top-k passages are passed to a
local extractive QA model, which returns the answer span. No API key required.
"""
from typing import List, Dict, Any
import logging

from qa_reader import ExtractiveReader

logger = logging.getLogger(__name__)


class RAGSystem:
    """Answer questions from retrieved documents using an extractive reader."""

    def __init__(self, reader_model: str = "deepset/roberta-base-squad2"):
        """Initialize the QA system with a HuggingFace extractive QA model."""
        self.reader = ExtractiveReader(model_name=reader_model)

    def generate_answer(self, query: str, retrieved_docs: List[str]) -> Dict[str, Any]:
        """Extract an answer to the query from the retrieved documents."""
        try:
            result = self.reader.answer(query, retrieved_docs)
            return {
                'answer': result['answer'],
                'score': result['score'],
                'source_rank': result['rank'],
                'context_used': result['context'],
                'success': True,
                'error': None
            }
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                'answer': f"Sorry, I encountered an error generating the answer: {str(e)}",
                'score': 0.0,
                'source_rank': None,
                'context_used': "",
                'success': False,
                'error': str(e)
            }

    def compare_rag_methods(self, query: str, retrievers_dict: Dict[str, Any],
                           corpus_texts: List[str], k: int = 3) -> Dict[str, Dict]:
        """Compare answers produced from each retrieval method's documents."""
        results = {}

        for method_name, retriever in retrievers_dict.items():
            try:
                retrieval_results = retriever.retrieve([query], k=k)
                retrieved_doc_indices = retrieval_results[0] if retrieval_results else []
                retrieved_docs = [corpus_texts[idx] for idx in retrieved_doc_indices
                                if idx < len(corpus_texts)]

                rag_result = self.generate_answer(query, retrieved_docs)
                results[method_name] = {
                    **rag_result,
                    'retrieved_indices': retrieved_doc_indices,
                    'retrieved_docs': retrieved_docs
                }

            except Exception as e:
                logger.error(f"Error with {method_name}: {e}")
                results[method_name] = {
                    'answer': f"Error with {method_name}: {str(e)}",
                    'score': 0.0,
                    'source_rank': None,
                    'retrieved_indices': [],
                    'retrieved_docs': [],
                    'context_used': "",
                    'success': False,
                    'error': str(e)
                }

        return results
