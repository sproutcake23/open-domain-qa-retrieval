"""
Extractive Question Answering on top of a retriever (retriever -> reader).

The retriever narrows millions of passages down to a handful of candidates;
the reader, a transformer fine-tuned on SQuAD 2.0, then extracts the exact
answer span from those candidates. Implemented directly on
AutoModelForQuestionAnswering so it works across transformers versions
(v5 removed the question-answering pipeline).
"""
from typing import List, Dict, Optional
from tqdm import tqdm


class ExtractiveReader:
    """Extract answer spans from retrieved passages with a QA transformer."""

    def __init__(self, model_name: str = "deepset/roberta-base-squad2",
                 device: Optional[str] = None, batch_size: int = 32,
                 max_seq_length: int = 384, max_answer_len: int = 30):
        """
        Initialize the reader.

        Args:
            model_name: HuggingFace extractive QA model
            device: "cuda" or "cpu". Defaults to GPU when available.
            batch_size: Number of (question, passage) pairs processed per forward pass
            max_seq_length: Max tokens for question + passage (passage is truncated)
            max_answer_len: Max answer length in tokens
        """
        import torch
        from transformers import AutoTokenizer, AutoModelForQuestionAnswering

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"📖 Loading reader model: {model_name}")
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.max_answer_len = max_answer_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(self.device).eval()
        print(f"✅ Reader loaded: {model_name} ({self.device})")

    def _read(self, questions: List[str], passages: List[str]) -> List[Dict]:
        """Find the best answer span in each aligned (question, passage) pair."""
        spans = []
        for start in range(0, len(questions), self.batch_size):
            batch_q = questions[start:start + self.batch_size]
            batch_p = passages[start:start + self.batch_size]
            enc = self.tokenizer(batch_q, batch_p, truncation="only_second", max_length=self.max_seq_length,
                                 padding=True, return_offsets_mapping=True, return_tensors="pt")
            offsets = enc.pop("offset_mapping")
            with self.torch.no_grad():
                out = self.model(**{k: v.to(self.device) for k, v in enc.items()})

            for i, passage in enumerate(batch_p):
                # Softmax over passage tokens plus the [CLS] "no answer" slot: for an irrelevant
                # passage most probability goes to [CLS], so span scores are comparable across passages
                is_context = self.torch.tensor([sid == 1 for sid in enc.sequence_ids(i)])
                allowed = is_context.clone()
                allowed[0] = True
                start_p = out.start_logits[i].float().cpu().masked_fill(~allowed, -1e4).softmax(-1)
                end_p = out.end_logits[i].float().cpu().masked_fill(~allowed, -1e4).softmax(-1)
                start_p, end_p = start_p * is_context, end_p * is_context  # answers come from the passage only
                scores = start_p[:, None] * end_p[None, :]
                scores = scores.triu().tril(self.max_answer_len - 1)  # end >= start, bounded length
                best = int(scores.argmax())
                s_tok, e_tok = divmod(best, scores.shape[1])
                char_start, char_end = int(offsets[i][s_tok][0]), int(offsets[i][e_tok][1])
                spans.append({'answer': passage[char_start:char_end], 'score': float(scores[s_tok, e_tok])})
        return spans

    def answer(self, question: str, passages: List[str],
               passage_ids: Optional[List[int]] = None) -> Dict:
        """
        Answer a question from a list of candidate passages.

        Args:
            question: Natural-language question
            passages: Candidate passages (typically the retriever's top-k)
            passage_ids: Corpus indices of the passages (defaults to 0..n-1)

        Returns:
            Dict with 'answer', 'score', 'passage_id', 'rank' (position in passages)
            and 'context' (the passage the answer came from)
        """
        if passage_ids is None:
            passage_ids = list(range(len(passages)))
        if not passages:
            return {'answer': '', 'score': 0.0, 'passage_id': None, 'rank': None, 'context': ''}

        spans = self._read([question] * len(passages), passages)
        rank = max(range(len(spans)), key=lambda i: spans[i]['score'])
        return {
            'answer': spans[rank]['answer'].strip(),
            'score': float(spans[rank]['score']),
            'passage_id': passage_ids[rank],
            'rank': rank,
            'context': passages[rank]
        }

    def evaluate(self, questions: List[str], retrieval_results: Dict[int, List[int]],
                 corpus_texts: List[str], qrels: Dict[int, Dict[int, int]],
                 query_ids: Optional[List[int]] = None, k: int = 5) -> Dict:
        """
        Answer many questions and measure how often answers come from a relevant passage.

        BEIR's NQ release has relevance judgments but no answer strings, so exact-match/F1
        cannot be computed. The gold-source rate (share of answers extracted from a passage
        that qrels marks relevant) is used as a proxy for end-to-end QA quality.

        Args:
            questions: All query texts (indexed by query id)
            retrieval_results: {query_id: [doc_ids]} from a retriever
            corpus_texts: Corpus passages
            qrels: {query_id: {doc_id: relevance}}
            query_ids: Subset of query ids to evaluate (defaults to all in retrieval_results)
            k: Number of retrieved passages given to the reader

        Returns:
            Dict with 'gold_source_rate', 'retrieval_hit_rate' (a relevant passage was among
            the k given to the reader), 'mean_score' and per-query 'predictions'
        """
        if query_ids is None:
            query_ids = list(retrieval_results.keys())

        predictions = []
        for q_id in tqdm(query_ids, desc="Reading"):
            doc_ids = retrieval_results[q_id][:k]
            pred = self.answer(questions[q_id], [corpus_texts[d] for d in doc_ids], doc_ids)
            relevant = {d for d, rel in qrels.get(q_id, {}).items() if rel > 0}
            pred.update({
                'query_id': q_id,
                'question': questions[q_id],
                'from_relevant': pred['passage_id'] in relevant,
                'relevant_in_top_k': bool(relevant & set(doc_ids))
            })
            predictions.append(pred)

        n = max(len(predictions), 1)
        return {
            'gold_source_rate': sum(p['from_relevant'] for p in predictions) / n,
            'retrieval_hit_rate': sum(p['relevant_in_top_k'] for p in predictions) / n,
            'mean_score': sum(p['score'] for p in predictions) / n,
            'predictions': predictions
        }
