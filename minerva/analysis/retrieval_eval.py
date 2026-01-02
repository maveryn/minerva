import argparse
import csv
import json
import os
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ATTACK_PARENT_RE = re.compile(r"^(T\d{4})\.\d{3}$")
TOKEN_CLEAN_RE = re.compile(r"[^a-z0-9]+")
GOLD_FALLBACK_KEYS = (
    "id",
    "ids",
    "label",
    "labels",
    "technique_id",
    "technique_ids",
)


def parse_ks(ks_str: str) -> List[int]:
    ks: List[int] = []
    for part in ks_str.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError as exc:
            raise ValueError(f"Invalid k value '{part}' in --ks.") from exc
        if value <= 0:
            raise ValueError(f"k values must be positive; got {value}.")
        ks.append(value)
    if not ks:
        raise ValueError("--ks must contain at least one integer value.")
    return ks


def get_by_path(obj: Dict[str, Any], path: str) -> Any:
    if not path:
        return obj
    current: Any = obj
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise KeyError(f"Missing path '{path}' at '{part}'.")
    return current


def normalize_id(value: Any, mode: str) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if mode == "attack_parent":
        return ATTACK_PARENT_RE.sub(r"\1", text)
    return text


def dedupe_preserve_order(items: Iterable[Any]) -> List[Any]:
    seen = set()
    result: List[Any] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def coerce_gold(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in GOLD_FALLBACK_KEYS:
            if key in value:
                return coerce_gold(value[key])
        if len(value) == 1:
            return coerce_gold(next(iter(value.values())))
    raise ValueError(
        "Gold labels must be a string, list, or dict containing a string/list."
    )


def normalize_gold_labels(
    gold_raw: Any, mode: str
) -> Tuple[List[str], set]:
    gold_list = coerce_gold(gold_raw)
    normalized: List[str] = []
    for item in gold_list:
        norm = normalize_id(item, mode)
        if norm:
            normalized.append(norm)
    normalized = dedupe_preserve_order(normalized)
    if not normalized:
        raise ValueError("Gold labels are empty after normalization.")
    return normalized, set(normalized)


def tokenize(text: Any) -> List[str]:
    cleaned = TOKEN_CLEAN_RE.sub(" ", str(text or "").lower())
    return [tok for tok in cleaned.split() if tok]


def load_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_num} in {path}: {exc}"
                ) from exc


def load_corpus(
    corpus_path: str, doc_id_path: str, doc_text_path: str
) -> Tuple[List[str], List[str]]:
    doc_ids: List[str] = []
    doc_texts: List[str] = []
    for obj in load_jsonl(corpus_path):
        doc_id = get_by_path(obj, doc_id_path)
        doc_text = get_by_path(obj, doc_text_path)
        if doc_id is None:
            raise ValueError("Corpus document ID is null.")
        doc_ids.append(str(doc_id))
        doc_texts.append("" if doc_text is None else str(doc_text))
    if not doc_ids:
        raise ValueError("Corpus is empty.")
    return doc_ids, doc_texts


def load_queries(
    dataset_path: str,
    query_text_path: str,
    gold_label_path: str,
    normalize_mode: str,
    task_filter: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> List[Dict[str, Any]]:
    queries: List[Dict[str, Any]] = []
    for line_idx, obj in enumerate(load_jsonl(dataset_path)):
        if task_filter is not None and obj.get("task") != task_filter:
            continue
        query_text = get_by_path(obj, query_text_path)
        gold_raw = get_by_path(obj, gold_label_path)
        gold_labels, gold_set = normalize_gold_labels(gold_raw, normalize_mode)
        queries.append(
            {
                "idx": line_idx,
                "query_text": "" if query_text is None else str(query_text),
                "gold_labels": gold_labels,
                "gold_set": gold_set,
            }
        )
        if max_rows is not None and len(queries) >= max_rows:
            break
    return queries


def require_bm25():
    try:
        from rank_bm25 import BM25Okapi  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "BM25 retrieval requires rank_bm25. Install via "
            "`pip install rank_bm25`."
        ) from exc
    return BM25Okapi


def require_dense():
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Dense retrieval requires numpy. Install via `pip install numpy`."
        ) from exc
    try:
        import faiss  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Dense retrieval requires faiss. Install via "
            "`pip install faiss-cpu`."
        ) from exc
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Dense retrieval requires sentence-transformers. Install via "
            "`pip install sentence-transformers`."
        ) from exc
    return np, faiss, SentenceTransformer


def build_bm25(corpus_texts: Sequence[str]):
    BM25Okapi = require_bm25()
    tokenized = [tokenize(text) for text in corpus_texts]
    return BM25Okapi(tokenized)


def try_load_dense_cache(
    cache_dir: str,
    expected_count: int,
    corpus_ids: Sequence[str],
):
    ids_path = os.path.join(cache_dir, "corpus_ids.json")
    emb_path = os.path.join(cache_dir, "corpus_emb.npy")
    index_path = os.path.join(cache_dir, "faiss.index")
    if not (os.path.exists(ids_path) and os.path.exists(emb_path) and os.path.exists(index_path)):
        return None
    np, faiss, _ = require_dense()
    try:
        with open(ids_path, "r", encoding="utf-8") as handle:
            cached_ids = json.load(handle)
        if not isinstance(cached_ids, list) or len(cached_ids) != expected_count:
            return None
        if list(cached_ids) != list(corpus_ids):
            return None
        embeddings = np.load(emb_path)
        if embeddings.shape[0] != expected_count:
            return None
        index = faiss.read_index(index_path)
        if index.ntotal != expected_count:
            return None
    except Exception:
        return None
    return embeddings, index


def save_dense_cache(
    cache_dir: str,
    corpus_ids: Sequence[str],
    embeddings,
    index,
):
    os.makedirs(cache_dir, exist_ok=True)
    ids_path = os.path.join(cache_dir, "corpus_ids.json")
    emb_path = os.path.join(cache_dir, "corpus_emb.npy")
    index_path = os.path.join(cache_dir, "faiss.index")
    with open(ids_path, "w", encoding="utf-8") as handle:
        json.dump(list(corpus_ids), handle)
    embeddings = embeddings.astype("float32", copy=False)
    import numpy as np  # type: ignore

    np.save(emb_path, embeddings)
    import faiss  # type: ignore

    faiss.write_index(index, index_path)


def build_dense_index(
    corpus_texts: Sequence[str],
    model_name: str,
    device: str,
    batch_size: int,
    cache_dir: Optional[str],
    corpus_ids: Sequence[str],
):
    np, faiss, SentenceTransformer = require_dense()
    model = SentenceTransformer(model_name, device=device)
    cache_hit = None
    if cache_dir:
        cache_hit = try_load_dense_cache(cache_dir, len(corpus_texts), corpus_ids)
    if cache_hit is not None:
        embeddings, index = cache_hit
        return model, embeddings, index

    embeddings = model.encode(
        list(corpus_texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    if cache_dir:
        save_dense_cache(cache_dir, corpus_ids, embeddings, index)
    return model, embeddings, index


def min_max_normalize(scores):
    import numpy as np  # type: ignore

    scores = np.asarray(scores, dtype="float32")
    if scores.size == 0:
        return scores
    min_val = float(scores.min())
    max_val = float(scores.max())
    if max_val == min_val:
        return np.zeros_like(scores)
    return (scores - min_val) / (max_val - min_val)


def top_n_indices(scores: Sequence[float], n: int) -> List[int]:
    if n <= 0:
        return []
    if n >= len(scores):
        return list(range(len(scores)))
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]


def ranks_from_order(order: Sequence[int], n_docs: int) -> List[int]:
    ranks = [0] * n_docs
    for rank, idx in enumerate(order, start=1):
        ranks[idx] = rank
    return ranks


def evaluate_predictions(
    queries: Sequence[Dict[str, Any]],
    predictions: Sequence[List[str]],
    ks: Sequence[int],
):
    recall_sums = {k: 0.0 for k in ks}
    hit_sums = {k: 0.0 for k in ks}
    hit_count = 0
    per_example: List[Dict[str, Any]] = []

    for query, pred_ids in zip(queries, predictions):
        gold_set = query["gold_set"]
        gold_labels = query["gold_labels"]
        pred_ids = dedupe_preserve_order(pred_ids)
        row = {
            "idx": query["idx"],
            "gold_labels": json.dumps(gold_labels),
            "topK_ids": json.dumps(pred_ids),
        }
        is_multilabel = len(gold_set) > 1
        if is_multilabel:
            hit_count += 1
        for k in ks:
            preds_k = pred_ids[:k]
            if len(gold_set) == 1:
                recall = 1.0 if next(iter(gold_set)) in preds_k else 0.0
            else:
                recall = len(set(preds_k) & gold_set) / float(len(gold_set))
            recall_sums[k] += recall
            row[f"recall@{k}"] = f"{recall:.6f}"
            if is_multilabel:
                hit = 1.0 if set(preds_k) & gold_set else 0.0
                hit_sums[k] += hit
                row[f"hit@{k}"] = f"{hit:.6f}"
        per_example.append(row)

    n_queries = len(queries)
    recall_avg = (
        {k: recall_sums[k] / n_queries for k in ks} if n_queries else {}
    )
    hit_avg = (
        {k: hit_sums[k] / hit_count for k in ks} if hit_count else {}
    )
    return recall_avg, hit_avg, per_example, hit_count


def write_per_example_csv(
    path: str,
    per_example: Sequence[Dict[str, Any]],
    ks: Sequence[int],
    include_hit: bool,
):
    if not per_example:
        return
    fieldnames = ["idx", "gold_labels", "topK_ids"]
    fieldnames.extend([f"recall@{k}" for k in ks])
    if include_hit:
        fieldnames.extend([f"hit@{k}" for k in ks])
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_example:
            if not include_hit:
                row = {k: v for k, v in row.items() if not k.startswith("hit@")}
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval recall@k over a JSONL dataset."
    )
    parser.add_argument("--dataset_jsonl", required=True)
    parser.add_argument("--corpus_jsonl", required=True)
    parser.add_argument(
        "--method", required=True, choices=["bm25", "dense", "hybrid", "rrf"]
    )
    parser.add_argument("--query_text_path", default="input.scenario")
    parser.add_argument("--gold_label_path", default="ground_truth.technique_id")
    parser.add_argument("--doc_id_path", default="id")
    parser.add_argument("--doc_text_path", default="text")
    parser.add_argument("--ks", default="1,5,10,20")
    parser.add_argument("--task_filter")
    parser.add_argument("--cache_dir")
    parser.add_argument("--max_rows", type=int)
    parser.add_argument("--write_per_example_csv")
    parser.add_argument(
        "--normalize_id",
        default="none",
        choices=["none", "attack_parent"],
    )
    parser.add_argument(
        "--model_name",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--bm25_candidate_k", type=int, default=50)
    parser.add_argument("--dense_candidate_k", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--rrf_k0", type=int, default=60)
    args = parser.parse_args()

    try:
        ks = parse_ks(args.ks)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if args.max_rows is not None and args.max_rows <= 0:
        raise SystemExit("--max_rows must be positive.")
    if args.batch_size <= 0:
        raise SystemExit("--batch_size must be positive.")
    if not (0.0 <= args.alpha <= 1.0):
        raise SystemExit("--alpha must be in [0, 1].")
    if args.bm25_candidate_k < 0 or args.dense_candidate_k < 0:
        raise SystemExit("Candidate k values must be non-negative.")
    if args.rrf_k0 <= 0:
        raise SystemExit("--rrf_k0 must be positive.")

    try:
        doc_ids, doc_texts = load_corpus(
            args.corpus_jsonl, args.doc_id_path, args.doc_text_path
        )
    except (KeyError, ValueError) as exc:
        raise SystemExit(f"Corpus error: {exc}") from exc

    normalized_doc_ids = [normalize_id(doc_id, args.normalize_id) for doc_id in doc_ids]

    try:
        queries = load_queries(
            args.dataset_jsonl,
            args.query_text_path,
            args.gold_label_path,
            args.normalize_id,
            task_filter=args.task_filter,
            max_rows=args.max_rows,
        )
    except (KeyError, ValueError) as exc:
        raise SystemExit(f"Dataset error: {exc}") from exc

    if not queries:
        result = {
            "method": args.method,
            "ks": ks,
            "n_queries": 0,
            "recall_at_k": {},
            "config": vars(args),
        }
        sys.stdout.write(json.dumps(result) + "\n")
        return

    max_k = min(max(ks), len(doc_ids))

    bm25 = None
    dense_model = None
    dense_embeddings = None
    dense_index = None

    if args.method in ("bm25", "hybrid", "rrf"):
        bm25 = build_bm25(doc_texts)

    if args.method in ("dense", "hybrid", "rrf"):
        dense_model, dense_embeddings, dense_index = build_dense_index(
            doc_texts,
            args.model_name,
            args.device,
            args.batch_size,
            args.cache_dir,
            doc_ids,
        )

    predictions: List[List[str]] = []

    if args.method == "bm25":
        for query in queries:
            scores = bm25.get_scores(tokenize(query["query_text"]))
            top_idx = top_n_indices(scores, max_k)
            pred_ids = [normalized_doc_ids[i] for i in top_idx]
            predictions.append(pred_ids)

    elif args.method == "dense":
        np, _, _ = require_dense()
        query_texts = [q["query_text"] for q in queries]
        query_embs = dense_model.encode(
            query_texts,
            batch_size=args.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        query_embs = np.asarray(query_embs, dtype="float32")
        distances, indices = dense_index.search(query_embs, max_k)
        for row in indices:
            pred_ids = [normalized_doc_ids[i] for i in row.tolist()]
            predictions.append(pred_ids)

    elif args.method == "hybrid":
        np, _, _ = require_dense()
        query_texts = [q["query_text"] for q in queries]
        query_embs = dense_model.encode(
            query_texts,
            batch_size=args.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        query_embs = np.asarray(query_embs, dtype="float32")
        dense_k = min(args.dense_candidate_k, len(doc_ids))
        if dense_k > 0:
            _, dense_indices = dense_index.search(query_embs, dense_k)
        else:
            dense_indices = np.zeros((len(queries), 0), dtype="int64")
        bm25_k = min(args.bm25_candidate_k, len(doc_ids))

        for idx, query in enumerate(queries):
            bm25_scores = bm25.get_scores(tokenize(query["query_text"]))
            bm25_top = top_n_indices(bm25_scores, bm25_k)
            dense_top = dense_indices[idx].tolist()
            candidate_idx = dedupe_preserve_order(bm25_top + dense_top)
            if not candidate_idx:
                predictions.append([])
                continue
            bm25_cand_scores = [bm25_scores[i] for i in candidate_idx]
            cand_emb = dense_embeddings[candidate_idx]
            dense_cand_scores = np.dot(cand_emb, query_embs[idx])

            bm25_norm = min_max_normalize(bm25_cand_scores)
            dense_norm = min_max_normalize(dense_cand_scores)
            fused = (1.0 - args.alpha) * bm25_norm + args.alpha * dense_norm
            order = fused.argsort()[::-1]
            top_idx = [candidate_idx[i] for i in order[:max_k]]
            pred_ids = [normalized_doc_ids[i] for i in top_idx]
            predictions.append(pred_ids)

    elif args.method == "rrf":
        np, _, _ = require_dense()
        query_texts = [q["query_text"] for q in queries]
        query_embs = dense_model.encode(
            query_texts,
            batch_size=args.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        query_embs = np.asarray(query_embs, dtype="float32")
        _, dense_indices = dense_index.search(query_embs, len(doc_ids))

        for idx, query in enumerate(queries):
            bm25_scores = bm25.get_scores(tokenize(query["query_text"]))
            bm25_order = top_n_indices(bm25_scores, len(doc_ids))
            dense_order = dense_indices[idx].tolist()
            bm25_ranks = ranks_from_order(bm25_order, len(doc_ids))
            dense_ranks = ranks_from_order(dense_order, len(doc_ids))

            rrf_scores = [
                1.0 / (args.rrf_k0 + bm25_ranks[i])
                + 1.0 / (args.rrf_k0 + dense_ranks[i])
                for i in range(len(doc_ids))
            ]
            top_idx = top_n_indices(rrf_scores, max_k)
            pred_ids = [normalized_doc_ids[i] for i in top_idx]
            predictions.append(pred_ids)

    recall_avg, hit_avg, per_example, hit_count = evaluate_predictions(
        queries, predictions, ks
    )

    result = {
        "method": args.method,
        "ks": ks,
        "n_queries": len(queries),
        "n_multilabel": hit_count,
        "recall_at_k": {str(k): recall_avg.get(k, 0.0) for k in ks},
        "config": vars(args),
    }
    if hit_avg:
        result["hit_at_k"] = {str(k): hit_avg.get(k, 0.0) for k in ks}

    sys.stdout.write(json.dumps(result) + "\n")

    if args.write_per_example_csv:
        write_per_example_csv(
            args.write_per_example_csv, per_example, ks, include_hit=bool(hit_avg)
        )


if __name__ == "__main__":
    main()
