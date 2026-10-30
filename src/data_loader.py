"""
Data Loading Module for the BEIR Natural Questions (NQ) retrieval benchmark.

Reads the standard BEIR on-disk layout directly:
    <data_dir>/corpus.jsonl
    <data_dir>/queries.jsonl
    <data_dir>/qrels/<split>.tsv

so it runs anywhere the files are available (locally, or from a Kaggle
input dataset) without network access. The `beir` package is only used to
download the dataset when no local copy exists.
"""
import csv
import json
import os
import pathlib
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Set

logger = logging.getLogger(__name__)

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"


def resolve_data_dir(dataset_name: str = "nq", data_dir: Optional[str] = None) -> str:
    """
    Locate the dataset folder, downloading it if it is not available locally.

    Args:
        dataset_name (str): BEIR dataset name (e.g., 'nq')
        data_dir (str, optional): Folder containing corpus.jsonl / queries.jsonl / qrels/.
            Defaults to <project>/dataset/<dataset_name>.

    Returns:
        str: Path to the dataset folder
    """
    if data_dir is not None:
        if not os.path.exists(os.path.join(data_dir, "corpus.jsonl")):
            raise FileNotFoundError(f"corpus.jsonl not found in {data_dir}")
        return data_dir

    out_dir = os.path.join(pathlib.Path.cwd().parent, "dataset")
    data_path = os.path.join(out_dir, dataset_name)
    if os.path.exists(os.path.join(data_path, "corpus.jsonl")):
        logger.info(f"Using local dataset at {data_path}")
        return data_path

    from beir import util  # only needed for downloading
    logger.info(f"Downloading {dataset_name} dataset to {out_dir}...")
    os.makedirs(out_dir, exist_ok=True)
    return util.download_and_unzip(BEIR_URL.format(name=dataset_name), out_dir)


def _read_queries(data_path: str) -> Dict[str, str]:
    queries = {}
    with open(os.path.join(data_path, "queries.jsonl"), encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            queries[row["_id"]] = row["text"]
    return queries


def _read_qrels(data_path: str, split: str) -> Dict[str, Dict[str, int]]:
    qrels: Dict[str, Dict[str, int]] = {}
    with open(os.path.join(data_path, "qrels", f"{split}.tsv"), encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header
        for query_id, doc_id, score in reader:
            qrels.setdefault(query_id, {})[doc_id] = int(score)
    return qrels


def _read_corpus_ids(data_path: str) -> List[str]:
    with open(os.path.join(data_path, "corpus.jsonl"), encoding="utf-8") as f:
        return [json.loads(line)["_id"] for line in f]


def _read_corpus(data_path: str, keep_ids: Optional[Set[str]] = None) -> Dict[str, Dict[str, str]]:
    """Read corpus.jsonl, optionally keeping only the given document IDs."""
    corpus = {}
    with open(os.path.join(data_path, "corpus.jsonl"), encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if keep_ids is None or row["_id"] in keep_ids:
                corpus[row["_id"]] = {"title": row.get("title", ""), "text": row["text"]}
    return corpus


def load_beir_dataset(dataset_name, split="test", data_dir=None):
    """
    Load a full BeIR dataset.

    Args:
        dataset_name (str): Dataset name (e.g., 'nq')
        split (str): 'train', 'dev', or 'test' (default: 'test')
        data_dir (str, optional): Local dataset folder (see resolve_data_dir)

    Returns:
        tuple: (corpus, queries, qrels)
            - corpus: dict of {doc_id: {'title': str, 'text': str}}
            - queries: dict of {query_id: str}
            - qrels: dict of {query_id: {doc_id: relevance_score}}
    """
    data_path = resolve_data_dir(dataset_name, data_dir)
    qrels = _read_qrels(data_path, split)
    queries = {q_id: text for q_id, text in _read_queries(data_path).items() if q_id in qrels}
    corpus = _read_corpus(data_path)
    return corpus, queries, qrels


def load_beir_nq(split="test", data_dir=None):
    """Load the full BeIR/nq (Natural Questions) dataset. See load_beir_dataset."""
    return load_beir_dataset("nq", split, data_dir)


class DataLoader:
    """
    Handles loading and preprocessing of datasets for semantic retrieval.
    """

    def __init__(self, dataset_name: str = "BeIR/nq", data_dir: Optional[str] = None):
        """
        Initialize DataLoader for BeIR datasets.

        Args:
            dataset_name (str): Dataset name ("BeIR/nq" for Natural Questions)
            data_dir (str, optional): Folder containing corpus.jsonl, queries.jsonl and qrels/
                (e.g. a Kaggle input dataset). Defaults to <project>/dataset/nq.
        """
        if dataset_name != "BeIR/nq":
            raise ValueError(f"Dataset {dataset_name} not supported. Supported dataset: 'BeIR/nq'")
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.corpus_dict = None
        self.queries_dict = None
        self.qrels_dict = None

    def load_dataset(self, split: str = "test", query_sample_size: Optional[int] = 500,
                     num_distractors: int = 0, random_seed: int = 42) -> Dict:
        """
        Load the dataset with query sampling and optional distractor passages.

        The corpus always contains every relevant passage for the sampled queries.
        `num_distractors` adds that many random, non-relevant passages from the
        full corpus so retrieval has to find the answer among unrelated text.

        Args:
            split (str): Dataset split ('train', 'dev', 'test')
            query_sample_size (int, optional): Number of queries to sample (None = all)
            num_distractors (int): Number of random non-relevant passages to add
            random_seed (int): Random seed for reproducible sampling

        Returns:
            Dict: Loaded dataset with corpus, queries, qrels
        """
        data_path = resolve_data_dir("nq", self.data_dir)
        logger.info(f"Loading BeIR/nq {split} split from {data_path}...")

        qrels = _read_qrels(data_path, split)
        queries = {q_id: text for q_id, text in _read_queries(data_path).items() if q_id in qrels}
        rng = np.random.default_rng(random_seed)

        if query_sample_size and len(queries) > query_sample_size:
            sampled_ids = rng.choice(sorted(queries), size=query_sample_size, replace=False)
            queries = {q_id: queries[q_id] for q_id in sampled_ids}
        qrels = {q_id: qrels[q_id] for q_id in queries}

        # Gold passages for the selected queries
        keep_ids = {doc_id for rels in qrels.values() for doc_id in rels}
        num_gold = len(keep_ids)

        if num_distractors > 0:
            logger.info(f"Sampling {num_distractors:,} distractor passages...")
            candidates = [doc_id for doc_id in _read_corpus_ids(data_path) if doc_id not in keep_ids]
            n = min(num_distractors, len(candidates))
            keep_ids.update(candidates[i] for i in rng.choice(len(candidates), size=n, replace=False))

        corpus = _read_corpus(data_path, keep_ids)

        self.corpus_dict = corpus
        self.queries_dict = queries
        self.qrels_dict = qrels

        logger.info(f"Dataset loaded: {len(queries):,} queries, {len(corpus):,} passages "
                    f"({num_gold:,} relevant + {len(corpus) - num_gold:,} distractors)")

        return {
            'corpus': corpus,
            'queries': queries,
            'qrels': qrels
        }

    def prepare_retrieval_data(self) -> Tuple[List[str], List[str], Dict]:
        """
        Convert BEIR format to retrieval format with sequential IDs.

        Returns:
            Tuple[List[str], List[str], Dict]: corpus_texts, query_texts, qrels_sequential
        """
        if self.corpus_dict is None or self.queries_dict is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        logger.info("Converting BEIR format to retrieval format...")

        # Convert corpus to list with sequential indexing
        corpus_texts = []
        doc_id_to_idx = {}

        for idx, (doc_id, doc_info) in enumerate(self.corpus_dict.items()):
            corpus_texts.append(doc_info['text'])
            doc_id_to_idx[doc_id] = idx

        # Convert queries to list with sequential indexing
        query_texts = []
        query_id_to_idx = {}

        for idx, (query_id, query_text) in enumerate(self.queries_dict.items()):
            query_texts.append(query_text)
            query_id_to_idx[query_id] = idx

        # Convert qrels to sequential format
        qrels_sequential = {}
        if self.qrels_dict:
            for query_id, relevant_docs in self.qrels_dict.items():
                if query_id in query_id_to_idx:
                    q_idx = query_id_to_idx[query_id]
                    qrels_sequential[q_idx] = {}

                    for doc_id, score in relevant_docs.items():
                        if doc_id in doc_id_to_idx:
                            doc_idx = doc_id_to_idx[doc_id]
                            qrels_sequential[q_idx][doc_idx] = score

        logger.info(f"Converted to retrieval format:")
        logger.info(f"  - Corpus texts: {len(corpus_texts)}")
        logger.info(f"  - Query texts: {len(query_texts)}")
        logger.info(f"  - QRELs with sequential IDs: {len(qrels_sequential)}")

        return corpus_texts, query_texts, qrels_sequential
