"""
Semantic Retrieval & Open-Domain QA - Core Modules

- data_loader: Load BEIR Natural Questions with query sampling and distractor passages
- bm25_retriever: Keyword-based retrieval (BM25)
- word2vec_retriever: Static word-embedding retrieval (Word2Vec)
- transformer_retriever: Dense retrieval with sentence-transformer embeddings
- hybrid_retriever: Reciprocal Rank Fusion of keyword and dense rankings
- qa_reader: Extractive QA reader that answers from retrieved passages
- evaluator: Recall@k, Precision@k and MRR
- utils: Visualization and comparison helpers

Modules import each other by name, so add this folder to sys.path before importing.
"""
