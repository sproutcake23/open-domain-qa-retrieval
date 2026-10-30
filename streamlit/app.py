"""
RAG Retrieval Comparison Streamlit App

This app demonstrates how different retrieval methods affect question answering
using BM25, Word2Vec, and Transformer-based retrievers, each feeding a local
extractive QA reader (retriever -> reader).
"""

import streamlit as st
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import our existing retrievers
try:
    from bm25_retriever import BM25Retriever
    from transformer_retriever import TransformerRetriever
    from word2vec_retriever import Word2VecRetriever
    from evaluator import IRMetrics
except ImportError as e:
    st.error(f"Error importing retrievers: {e}")
    st.error("Please ensure you're running this from the correct directory with access to the src folder.")
    st.stop()

# Import demo data and RAG system
from demo_data import DEMO_CORPUS, CORPUS_TEXTS, GROUND_TRUTH, SAMPLE_QUERIES
from rag_system import RAGSystem

# Page configuration
st.set_page_config(
    page_title="RAG Retrieval Comparison",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.retrieval-box {
    border: 2px solid #e0e0e0;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    background-color: #f9f9f9;
}

.correct-retrieval {
    border-color: #4CAF50 !important;
    background-color: #f1f8e9 !important;
}

.incorrect-retrieval {
    border-color: #f44336 !important;
    background-color: #ffebee !important;
}

.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 10px 0;
}

.query-box {
    background-color: #e3f2fd;
    padding: 15px;
    border-radius: 10px;
    border-left: 5px solid #2196F3;
    margin: 15px 0;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_retrievers():
    """Initialize and cache all retrievers."""
    with st.spinner("🔧 Initializing retrievers..."):
        # BM25 Retriever
        bm25 = BM25Retriever()
        bm25.build_index(CORPUS_TEXTS)
        
        # Transformer Retriever - use a more powerful model for better performance
        transformer = TransformerRetriever(model_name="all-mpnet-base-v2")
        transformer.build_index(CORPUS_TEXTS)
        
        # Word2Vec Retriever - optimized for small corpus
        # Use smaller dimensions and aggressive parameters for limited data
        word2vec = Word2VecRetriever(
            vector_size=100,  # Smaller for limited data
            window=15,        # Larger window to capture more context
            min_count=1,      # Keep all words in small corpus
            epochs=100,       # More training iterations
            sg=1              # Skip-gram works better for small corpora
        )
        word2vec.build_index(CORPUS_TEXTS)
        
    return {
        "BM25 (Keywords)": bm25,
        "Word2Vec (Static Embeddings)": word2vec,
        "Transformer (MPNet-base-v2)": transformer
    }

def evaluate_retrieval(query: str, retrieved_indices: List[int], ground_truth: List[int]) -> Dict[str, float]:
    """Evaluate retrieval performance for a query."""
    if not ground_truth:
        return {
            "precision": None, 
            "recall": None, 
            "f1": None,
            "relevant_found": None,
            "total_relevant": 0
        }
    
    retrieved_set = set(retrieved_indices)
    relevant_set = set(ground_truth)
    
    # Calculate metrics
    true_positives = len(retrieved_set & relevant_set)
    precision = true_positives / len(retrieved_set) if retrieved_set else 0.0
    recall = true_positives / len(relevant_set) if relevant_set else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "relevant_found": true_positives,
        "total_relevant": len(relevant_set)
    }

@st.cache_resource
def initialize_qa_system():
    """Load and cache the extractive QA reader."""
    with st.spinner("📖 Loading QA reader..."):
        return RAGSystem()

def display_retrieved_documents(method_name: str, retrieved_indices: List[int], 
                              ground_truth: List[int], rag_answer: Dict[str, Any] = None):
    """Display retrieved documents with visual indicators and ranking."""
    st.markdown(f"### 🔍 {method_name}")
    
    if rag_answer:
        st.markdown("**🤖 Answer:**")
        if rag_answer.get('success') and rag_answer.get('answer'):
            st.info(f"{rag_answer['answer']}  \n"
                    f"_confidence {rag_answer['score']:.2f} · from rank {rag_answer['source_rank'] + 1}_")
        else:
            st.warning(rag_answer.get('answer') or "No answer found in the retrieved documents.")
    
    st.markdown("**📄 Retrieved Documents (Ranked by Similarity):**")
    
    for i, doc_idx in enumerate(retrieved_indices[:3], 1):
        if doc_idx < len(DEMO_CORPUS):
            doc = DEMO_CORPUS[doc_idx]
            
            # Only show relevance indicators if we have ground truth
            if ground_truth:
                is_relevant = doc_idx in ground_truth
                if is_relevant:
                    st.markdown(f"""
                    <div class="retrieval-box correct-retrieval">
                        <h4>✅ Rank {i}: {doc['title']}</h4>
                        <p><strong>Relevance:</strong> ✅ Correct (Doc #{doc_idx})</p>
                        <p>{doc['content'][:200]}...</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="retrieval-box incorrect-retrieval">
                        <h4>❌ Rank {i}: {doc['title']}</h4>
                        <p><strong>Relevance:</strong> ❌ Not relevant (Doc #{doc_idx})</p>
                        <p>{doc['content'][:200]}...</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                # No ground truth - just show the document without relevance indicators
                st.markdown(f"""
                <div class="retrieval-box">
                    <h4>📄 Rank {i}: {doc['title']}</h4>
                    <p><strong>Document #{doc_idx}</strong> (sorted by similarity)</p>
                    <p>{doc['content'][:200]}...</p>
                </div>
                """, unsafe_allow_html=True)

def main():
    """Main application."""
    # Title and description
    st.title("🔍 Retrieval-Augmented QA Comparison Demo")
    st.markdown("""
    **Compare how different retrieval methods affect question answering!**
    
    This demo shows how BM25 (keywords), Word2Vec (embeddings), and Transformers (semantic) 
    retrieve different documents for the same query. A local extractive QA model
    (RoBERTa fine-tuned on SQuAD 2.0) then reads each method's documents and extracts
    the answer, so retrieval quality directly determines answer quality.
    """)
    
    # Sidebar configuration
    st.sidebar.header("🛠️ Configuration")
    
    enable_rag = st.sidebar.checkbox(
        "Enable QA Answers",
        value=True,
        help="Extract answers from each method's retrieved documents with a local QA model"
    )
    
    num_docs = st.sidebar.slider(
        "Number of documents to retrieve",
        min_value=1,
        max_value=5,
        value=3,
        help="How many documents each method should retrieve"
    )
    
    show_debug = st.sidebar.checkbox(
        "Show debug info",
        value=False,
        help="Display similarity scores and ranking details"
    )
    
    # Initialize retrievers
    retrievers = initialize_retrievers()
    
    # RAG System initialization
    rag_system = None
    if enable_rag:
        try:
            rag_system = initialize_qa_system()
            st.sidebar.success("✅ QA reader loaded")
        except Exception as e:
            st.sidebar.error(f"❌ QA reader failed to load: {e}")
            enable_rag = False
    
    # Main interface
    st.header("🎯 Demo Company: TechFlow AI")
    
    # Show corpus information
    with st.expander("📚 View Demo Corpus (5 documents about TechFlow AI)"):
        for i, doc in enumerate(DEMO_CORPUS):
            st.markdown(f"**Document {i+1}: {doc['title']}**")
            st.write(doc['content'])
            st.markdown("---")
    
    # Query input
    st.header("💬 Enter Your Query")
    
    # Quick query buttons
    st.markdown("**🚀 Try these sample queries:**")
    cols = st.columns(3)
    selected_query = None
    
    for i, query in enumerate(SAMPLE_QUERIES[:6]):
        col_idx = i % 3
        with cols[col_idx]:
            if st.button(f"💡 {query}", key=f"sample_{i}"):
                selected_query = query
    
    # Text input for custom query
    query = st.text_input(
        "Or type your own query:",
        value=selected_query or "",
        placeholder="e.g., What does TechFlow AI do?"
    )
    
    if query:
        st.markdown(f"""
        <div class="query-box">
            <h3>🔍 Query: "{query}"</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Get ground truth for evaluation (if available)
        ground_truth = GROUND_TRUTH.get(query, [])
        if ground_truth:
            st.info(f"📊 Ground Truth: Documents {[i+1 for i in ground_truth]} are most relevant for this query")
        else:
            st.info("📊 No predefined ground truth for this query - answers come from whatever documents are retrieved")
        
        # Perform retrieval with all methods
        st.header("🔄 Retrieval Results Comparison")
        
        retrieval_results = {}
        eval_results = {}
        rag_results = {}
        
        # Get retrieval results
        for method_name, retriever in retrievers.items():
            with st.spinner(f"Running {method_name}..."):
                results = retriever.retrieve([query], k=num_docs)
                retrieved_indices = results[0] if results else []
                
                # Debug: Show retrieval ranking for verification
                if show_debug:
                    st.write(f"**{method_name} Rankings:** {retrieved_indices}")
                
                retrieval_results[method_name] = retrieved_indices
                
                # Evaluate performance
                eval_results[method_name] = evaluate_retrieval(query, retrieved_indices, ground_truth)
        
        # Get RAG results if enabled
        if enable_rag and rag_system:
            with st.spinner("🤖 Extracting answers..."):
                rag_results = rag_system.compare_rag_methods(query, retrievers, CORPUS_TEXTS, k=num_docs)
        
        # Display results in columns
        cols = st.columns(len(retrievers))
        
        for i, (method_name, retrieved_indices) in enumerate(retrieval_results.items()):
            with cols[i]:
                rag_answer = None
                if enable_rag and method_name in rag_results:
                    rag_answer = rag_results[method_name]
                
                display_retrieved_documents(method_name, retrieved_indices, ground_truth, rag_answer)
                
                # Display metrics (only if we have ground truth)
                eval_data = eval_results[method_name]
                if eval_data['precision'] is not None:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>📊 Performance</h4>
                        <p><strong>Precision:</strong> {eval_data['precision']:.2f}</p>
                        <p><strong>Recall:</strong> {eval_data['recall']:.2f}</p>
                        <p><strong>F1-Score:</strong> {eval_data['f1']:.2f}</p>
                        <p><strong>Relevant Found:</strong> {eval_data['relevant_found']}/{eval_data['total_relevant']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>📊 Performance</h4>
                        <p>No ground truth available</p>
                        <p>Answers use the retrieved docs</p>
                    </div>
                    """, unsafe_allow_html=True)
        
    # Footer
    st.markdown("---")
    st.markdown("""
    **💡 About this Demo:**
    This app demonstrates why retrieval is the foundation of good QA and RAG systems. 
    Different retrieval methods (keywords vs embeddings vs transformers) find different documents, 
    and the reader can only answer from what the retriever gives it.
    """)

if __name__ == "__main__":
    main()
