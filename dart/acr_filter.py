from __future__ import annotations

import json
import math
import os
from difflib import SequenceMatcher
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
RLVR_ROOT = REPO_ROOT / "rlvr"
for path in (REPO_ROOT, RLVR_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from verl.utils.reward_score import reward_acr as reward_acr_module


_TEXTCNN_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


class _TextCNNModel(torch.nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, kernel_sizes: List[int], num_filters: int):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv1d(embed_dim, num_filters, kernel_size) for kernel_size in kernel_sizes]
        )
        self.dropout = torch.nn.Dropout(0.0)
        self.fc = torch.nn.Linear(num_filters * len(kernel_sizes), 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embed(x)
        emb = emb.transpose(1, 2)
        feats = []
        for conv in self.convs:
            y = torch.relu(conv(emb))
            y = torch.max(y, dim=2).values
            feats.append(y)
        out = torch.cat(feats, dim=1)
        out = self.dropout(out)
        return self.fc(out)


def _render_prompt_payload(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if not isinstance(payload, list):
        return ""
    for msg in reversed(payload):
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "").lower()
        content = msg.get("content")
        if role == "user" and isinstance(content, str):
            return content
    contents: List[str] = []
    for msg in payload:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            contents.append(content)
    return "\n".join(contents).strip()


def _extract_prompt_text(extra_info: Any) -> str:
    if not isinstance(extra_info, dict):
        return ""
    for key in ("acr_orig_prompt", "orig_prompt", "raw_prompt", "prompt"):
        text = _render_prompt_payload(extra_info.get(key))
        if text:
            return text
    return ""


def _normalize_filter_mode(mode: str, disable_filters: bool) -> str:
    mode = str(mode or "heuristic").lower().strip()
    compact = mode.replace(" ", "")
    if compact in {"ml+heuristic", "ml+heuristics", "ml_heuristic", "ml_heuristics", "ml&heuristic", "both"}:
        mode = "ml+heuristic"
    if mode not in {"off", "heuristic", "ml", "ml+heuristic"}:
        mode = "heuristic"
    if mode == "heuristic" and disable_filters:
        mode = "off"
    return mode


def _distinct_ngram_ratio(tokens: List[int], n: int) -> float:
    total = len(tokens) - n + 1
    if total <= 0:
        return 1.0
    ngrams = {tuple(tokens[i : i + n]) for i in range(total)}
    return len(ngrams) / total


def _has_repeated_window(
    tokens: List[int],
    *,
    window_size: int,
    window_stride: int,
    window_jaccard: float,
) -> bool:
    if window_size <= 0 or len(tokens) < window_size * 2:
        return False
    stride = max(1, window_stride)
    windows: List[Tuple[int, set[Tuple[int, int, int]]]] = []
    seen_exact: set[Tuple[int, ...]] = set()
    max_start = len(tokens) - window_size
    for start in range(0, max_start + 1, stride):
        window = tokens[start : start + window_size]
        window_key = tuple(window)
        if window_key in seen_exact:
            return True
        seen_exact.add(window_key)
        if len(window) < 3:
            continue
        ngrams = {tuple(window[i : i + 3]) for i in range(len(window) - 2)}
        windows.append((start, ngrams))
    for i, (start_i, grams_i) in enumerate(windows):
        for start_j, grams_j in windows[i + 1 :]:
            if abs(start_i - start_j) < window_size:
                continue
            union = grams_i | grams_j
            if not union:
                continue
            jaccard = len(grams_i & grams_j) / len(union)
            if jaccard >= window_jaccard:
                return True
    return False


def _sentence_near_duplicate(
    text: str,
    *,
    sentence_sim: float,
    sentence_window: int,
    sentence_min_words: int,
) -> bool:
    if not text or sentence_sim <= 0.0:
        return False
    raw_parts = re.split(r"[.!?]+|\n+", text)
    sentences: List[str] = []
    for part in raw_parts:
        cleaned = re.sub(r"[^a-z0-9\s]", "", part.lower())
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            continue
        if sentence_min_words and len(cleaned.split()) < sentence_min_words:
            continue
        sentences.append(cleaned)
    if len(sentences) < 2:
        return False
    window = max(1, sentence_window)
    for i, left in enumerate(sentences):
        for j in range(i + 1, min(len(sentences), i + 1 + window)):
            if SequenceMatcher(None, left, sentences[j]).ratio() >= sentence_sim:
                return True
    return False


class AcrDistillFilter:
    def __init__(
        self,
        tokenizer,
        *,
        filter_mode: str = "ml+heuristic",
        disable_filters: bool = False,
        reward_threshold: float = 1.0,
        degenerate_filter: bool = True,
        degenerate_min_tokens: int = 30,
        degenerate_rep_3_max: float = 0.70,
        degenerate_rep_4_max: float = 0.75,
        degenerate_window_size: int = 24,
        degenerate_window_stride: int | None = None,
        degenerate_window_jaccard: float = 0.9,
        degenerate_sentence_sim: float = 0.75,
        degenerate_sentence_window: int = 6,
        degenerate_sentence_min_words: int = 6,
        filter_model: str = "xashru/textcnn-response-only-lr6e-4-k345-f384-e300-t1024-d0p25",
        filter_model_type: str = "textcnn",
        filter_threshold: float = 0.5,
        filter_batch_size: int = 128,
        filter_max_length: int = 1024,
        filter_text_mode: str = "response",
        filter_device: str = "auto",
        filter_dtype: str = "auto",
        filter_tokenizer: str | None = None,
        filter_trust_remote_code: bool = False,
        r_correct: float = 0.1,
        leak_penalty: float = 0.5,
        multilabel_match: str = "exact",
        enforce_no_id_in_reasoning: bool = False,
        max_id_mentions: int = 0,
        min_reasoning_chars: int = 100,
        min_overlap_jaccard: float = 0.05,
    ) -> None:
        self.tokenizer = tokenizer
        self.filter_mode = str(filter_mode)
        self.disable_filters = bool(disable_filters)
        self.reward_threshold = float(reward_threshold)
        self.degenerate_filter = bool(degenerate_filter)
        self.degenerate_min_tokens = int(degenerate_min_tokens)
        self.degenerate_rep_3_max = float(degenerate_rep_3_max)
        self.degenerate_rep_4_max = float(degenerate_rep_4_max)
        self.degenerate_window_size = int(degenerate_window_size)
        if degenerate_window_stride is None:
            degenerate_window_stride = max(1, self.degenerate_window_size // 2)
        self.degenerate_window_stride = int(degenerate_window_stride)
        self.degenerate_window_jaccard = float(degenerate_window_jaccard)
        self.degenerate_sentence_sim = float(degenerate_sentence_sim)
        self.degenerate_sentence_window = int(degenerate_sentence_window)
        self.degenerate_sentence_min_words = int(degenerate_sentence_min_words)
        self.filter_model = str(filter_model)
        self.filter_model_type = str(filter_model_type)
        self.filter_threshold = float(filter_threshold)
        self.filter_batch_size = int(filter_batch_size)
        self.filter_max_length = int(filter_max_length)
        self.filter_text_mode = str(filter_text_mode)
        self.filter_device = str(filter_device)
        self.filter_dtype = str(filter_dtype)
        self.filter_tokenizer = filter_tokenizer
        self.filter_trust_remote_code = bool(filter_trust_remote_code)
        self.r_correct = float(r_correct)
        self.leak_penalty = float(leak_penalty)
        self.multilabel_match = str(multilabel_match)
        self.enforce_no_id_in_reasoning = bool(enforce_no_id_in_reasoning)
        self.max_id_mentions = int(max_id_mentions)
        self.min_reasoning_chars = int(min_reasoning_chars)
        self.min_overlap_jaccard = float(min_overlap_jaccard)

        self._filter_state: Optional[dict] = None
        self._filter_disabled = False

    def _merge_extra_info(self, row: Dict[str, Any]) -> Dict[str, Any]:
        extra_info = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
        merged = dict(extra_info)
        base_messages = row.get("base_messages")
        prompt_used = row.get("prompt_used_messages")
        if isinstance(base_messages, list) and "acr_orig_prompt" not in merged:
            merged["acr_orig_prompt"] = base_messages
        if isinstance(prompt_used, list):
            merged["acr_prompt"] = prompt_used
        return merged

    def _compute_acr_reward_info(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return reward_acr_module.reward_acr(
            str(row.get("data_source") or ""),
            str(row.get("response_text") or ""),
            row.get("ground_truth"),
            extra_info=self._merge_extra_info(row),
            r_correct=self.r_correct,
            leak_penalty=self.leak_penalty,
            multilabel_match=self.multilabel_match,
            enforce_no_id_in_reasoning=self.enforce_no_id_in_reasoning,
            max_id_mentions=self.max_id_mentions,
            min_reasoning_chars=self.min_reasoning_chars,
            min_overlap_jaccard=self.min_overlap_jaccard,
        )

    def _response_ids(self, response_text: str) -> List[int]:
        return self.tokenizer.encode(response_text or "", add_special_tokens=False)

    def _is_degenerate(self, response_ids: List[int], response_text: str) -> bool:
        if not self.degenerate_filter or len(response_ids) < self.degenerate_min_tokens:
            return False
        rep_3 = 1.0 - _distinct_ngram_ratio(response_ids, 3)
        rep_4 = 1.0 - _distinct_ngram_ratio(response_ids, 4)
        if rep_3 >= self.degenerate_rep_3_max or rep_4 >= self.degenerate_rep_4_max:
            return True
        if _has_repeated_window(
            response_ids,
            window_size=self.degenerate_window_size,
            window_stride=self.degenerate_window_stride,
            window_jaccard=self.degenerate_window_jaccard,
        ):
            return True
        if _sentence_near_duplicate(
            response_text or "",
            sentence_sim=self.degenerate_sentence_sim,
            sentence_window=self.degenerate_sentence_window,
            sentence_min_words=self.degenerate_sentence_min_words,
        ):
            return True
        return False

    def _maybe_init_textcnn(self) -> None:
        if self._filter_state is not None or self._filter_disabled:
            return
        model_type = str(self.filter_model_type or "auto").lower().strip()
        if model_type == "auto":
            model_type = "textcnn" if "textcnn" in self.filter_model.lower() else "textcnn"
        if model_type != "textcnn":
            raise ValueError(f"Unsupported ACR filter model_type for DART: {self.filter_model_type}")

        device_name = str(self.filter_device or "auto").lower().strip()
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_name)

        try:
            if os.path.isdir(self.filter_model):
                model_path = os.path.join(self.filter_model, "model.pt")
                metrics_path = os.path.join(self.filter_model, "metrics.json")
            else:
                from huggingface_hub import hf_hub_download

                model_path = hf_hub_download(self.filter_model, "model.pt")
                try:
                    metrics_path = hf_hub_download(self.filter_model, "metrics.json")
                except Exception:
                    metrics_path = None

            payload = torch.load(model_path, map_location="cpu", weights_only=False)
            state_dict = payload.get("state_dict")
            vocab = payload.get("vocab") or {}
            if state_dict is None or not vocab:
                raise ValueError("textcnn model.pt missing state_dict or vocab")

            metrics = {}
            if metrics_path and os.path.exists(metrics_path):
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)

            embed_dim = int(metrics.get("embed_dim", 200))
            kernel_sizes = metrics.get("kernel_sizes", [3, 4, 5])
            if isinstance(kernel_sizes, str):
                kernel_sizes = [int(x) for x in kernel_sizes.split(",") if x.strip()]
            num_filters = int(metrics.get("num_filters", 256))
            model = _TextCNNModel(len(vocab), embed_dim, kernel_sizes, num_filters)
            model.load_state_dict(state_dict)
            model.eval()
            model.to(device)

            max_tokens = self.filter_max_length if self.filter_max_length > 0 else None
            self._filter_state = {
                "type": "textcnn",
                "model": model,
                "device": device,
                "vocab": vocab,
                "max_tokens": max_tokens,
                "text_mode": self.filter_text_mode.lower().strip(),
                "batch_size": self.filter_batch_size,
            }
        except Exception:
            self._filter_disabled = True
            self._filter_state = None

    def _textcnn_build_tokens(
        self,
        prompt: str,
        response: str,
        text_mode: str,
        max_tokens: Optional[int],
    ) -> List[str]:
        prompt_tokens = _TEXTCNN_TOKEN_RE.findall((prompt or "").lower())
        response_tokens = _TEXTCNN_TOKEN_RE.findall((response or "").lower())
        if text_mode == "response":
            tokens = response_tokens
            return tokens[:max_tokens] if max_tokens else tokens
        if text_mode == "prompt":
            tokens = prompt_tokens
            return tokens[:max_tokens] if max_tokens else tokens
        if max_tokens is not None:
            if len(response_tokens) >= max_tokens:
                response_tokens = response_tokens[:max_tokens]
                prompt_tokens = []
            else:
                budget = max_tokens - len(response_tokens)
                prompt_tokens = prompt_tokens[:budget]
        return response_tokens + ["sep"] + prompt_tokens

    def _score_textcnn(self, prompt_texts: List[str], response_texts: List[str]) -> Optional[List[float]]:
        self._maybe_init_textcnn()
        state = self._filter_state
        if state is None:
            return None
        model = state["model"]
        device = state["device"]
        vocab = state["vocab"]
        max_tokens = state.get("max_tokens")
        text_mode = state.get("text_mode", "response")
        batch_size = int(state.get("batch_size") or 64)
        scores: List[float] = []
        model.eval()
        for start in range(0, len(prompt_texts), batch_size):
            batch_prompts = prompt_texts[start : start + batch_size]
            batch_responses = response_texts[start : start + batch_size]
            seqs: List[List[int]] = []
            for prompt, response in zip(batch_prompts, batch_responses, strict=True):
                tokens = self._textcnn_build_tokens(prompt, response, text_mode, max_tokens)
                if not tokens:
                    tokens = ["<unk>"]
                seqs.append([vocab.get(tok, 1) for tok in tokens])
            if not seqs:
                continue
            max_len = max(len(seq) for seq in seqs)
            if max_tokens:
                max_len = min(max_len, int(max_tokens))
            if max_len <= 0:
                max_len = 1
            padded = []
            for seq in seqs:
                if len(seq) >= max_len:
                    padded.append(seq[:max_len])
                else:
                    padded.append(seq + [0] * (max_len - len(seq)))
            x = torch.tensor(padded, dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=-1)
                scores.extend(probs[:, 1].detach().cpu().tolist())
        return scores

    def apply(self, rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        mode = _normalize_filter_mode(self.filter_mode, self.disable_filters)
        if mode in {"ml", "ml+heuristic"}:
            self._maybe_init_textcnn()
            if self._filter_state is None:
                mode = "off" if self.disable_filters else "heuristic"

        use_ml = mode in {"ml", "ml+heuristic"}
        use_heuristics = mode in {"heuristic", "ml+heuristic"}
        degenerate_filter = self.degenerate_filter if use_heuristics else False

        metrics = {
            "filter_mode": mode,
            "rows_scored": len(rows),
            "heuristic_input": 0,
            "heuristic_passed": 0,
            "degenerate_checked": 0,
            "degenerate_filtered": 0,
            "ml_filter_input": 0,
            "ml_filter_passed": 0,
            "filter_threshold": float(self.filter_threshold),
            "reward_threshold": float(self.reward_threshold),
        }

        base_ok_by_idx: Dict[int, bool] = {}
        heuristic_ok_by_idx: Dict[int, bool] = {}
        degenerate_hit_by_idx: Dict[int, bool] = {}

        for idx, row in enumerate(rows):
            row = dict(row)
            rows[idx] = row
            acr_info = self._compute_acr_reward_info(row)
            row["acr_reward_info"] = acr_info
            row["acr_filter_reward"] = float(acr_info.get("score", 0.0))
            for key, value in acr_info.items():
                if key == "score":
                    continue
                row[key] = value

            base_score = float(acr_info.get("acr_base_score", 0.0) or 0.0)
            base_ok = math.isfinite(base_score) and base_score >= self.reward_threshold
            base_ok_by_idx[idx] = base_ok
            row["acr_base_pass"] = bool(base_ok)
            row["acr_degenerate_hit"] = False
            row["acr_heuristic_pass"] = False
            row["acr_ml_score"] = None
            row["acr_ml_pass"] = False
            row["acr_filter_pass"] = False

            if not base_ok:
                continue
            metrics["heuristic_input"] += 1
            if use_heuristics:
                if bool(acr_info.get("acr_leak_hit", False)):
                    continue
                degenerate_hit = False
                if degenerate_filter:
                    metrics["degenerate_checked"] += 1
                    response_text = str(row.get("response_text") or "")
                    response_ids = self._response_ids(response_text)
                    degenerate_hit = (not response_ids) or self._is_degenerate(response_ids, response_text)
                    row["acr_degenerate_hit"] = bool(degenerate_hit)
                    if degenerate_hit:
                        metrics["degenerate_filtered"] += 1
                        degenerate_hit_by_idx[idx] = True
                        continue
                heuristic_ok_by_idx[idx] = True
                row["acr_heuristic_pass"] = True
                metrics["heuristic_passed"] += 1
            else:
                heuristic_ok_by_idx[idx] = True

        if use_ml:
            idxs_to_score: List[int] = []
            prompt_texts: List[str] = []
            response_texts: List[str] = []
            for idx, row in enumerate(rows):
                if not base_ok_by_idx.get(idx, False):
                    continue
                if use_heuristics and not heuristic_ok_by_idx.get(idx, False):
                    continue
                response_text = str(row.get("response_text") or "")
                if not response_text:
                    continue
                idxs_to_score.append(idx)
                prompt_texts.append(_extract_prompt_text(self._merge_extra_info(row)))
                response_texts.append(response_text)
            if idxs_to_score:
                scores = self._score_textcnn(prompt_texts, response_texts)
                if scores is None or len(scores) != len(idxs_to_score):
                    mode = "off" if self.disable_filters else "heuristic"
                    use_ml = False
                    use_heuristics = mode in {"heuristic", "ml+heuristic"}
                    metrics["filter_mode"] = mode
                else:
                    for idx, score in zip(idxs_to_score, scores, strict=True):
                        score_val = float(score) if math.isfinite(float(score)) else -1.0
                        rows[idx]["acr_ml_score"] = score_val
                        metrics["ml_filter_input"] += 1
                        if score_val >= self.filter_threshold:
                            rows[idx]["acr_ml_pass"] = True
                            metrics["ml_filter_passed"] += 1

        accepted = 0
        for idx, row in enumerate(rows):
            if not base_ok_by_idx.get(idx, False):
                row["acr_filter_pass"] = False
                continue
            if use_heuristics and not heuristic_ok_by_idx.get(idx, False):
                row["acr_filter_pass"] = False
                continue
            if use_ml and not bool(row.get("acr_ml_pass", False)):
                row["acr_filter_pass"] = False
                continue
            row["acr_filter_pass"] = True
            accepted += 1
        metrics["accepted_after_filter"] = accepted
        return rows, metrics


def acr_filter_from_config(tokenizer, cfg: Optional[Dict[str, Any]]) -> Optional[AcrDistillFilter]:
    if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
        return None
    reward_kwargs = cfg.get("reward_kwargs", {}) if isinstance(cfg.get("reward_kwargs"), dict) else {}
    return AcrDistillFilter(
        tokenizer,
        filter_mode=str(cfg.get("filter_mode", "ml+heuristic")),
        disable_filters=bool(cfg.get("disable_filters", False)),
        reward_threshold=float(cfg.get("reward_threshold", 1.0)),
        degenerate_filter=bool(cfg.get("degenerate_filter", True)),
        degenerate_min_tokens=int(cfg.get("degenerate_min_tokens", 30)),
        degenerate_rep_3_max=float(cfg.get("degenerate_rep_3_max", 0.70)),
        degenerate_rep_4_max=float(cfg.get("degenerate_rep_4_max", 0.75)),
        degenerate_window_size=int(cfg.get("degenerate_window_size", 24)),
        degenerate_window_stride=cfg.get("degenerate_window_stride"),
        degenerate_window_jaccard=float(cfg.get("degenerate_window_jaccard", 0.9)),
        degenerate_sentence_sim=float(cfg.get("degenerate_sentence_sim", 0.75)),
        degenerate_sentence_window=int(cfg.get("degenerate_sentence_window", 6)),
        degenerate_sentence_min_words=int(cfg.get("degenerate_sentence_min_words", 6)),
        filter_model=str(
            cfg.get("filter_model", "xashru/textcnn-response-only-lr6e-4-k345-f384-e300-t1024-d0p25")
        ),
        filter_model_type=str(cfg.get("filter_model_type", "textcnn")),
        filter_threshold=float(cfg.get("filter_threshold", 0.5)),
        filter_batch_size=int(cfg.get("filter_batch_size", 128)),
        filter_max_length=int(cfg.get("filter_max_length", 1024)),
        filter_text_mode=str(cfg.get("filter_text_mode", "response")),
        filter_device=str(cfg.get("filter_device", "auto")),
        filter_dtype=str(cfg.get("filter_dtype", "auto")),
        filter_tokenizer=cfg.get("filter_tokenizer"),
        filter_trust_remote_code=bool(cfg.get("filter_trust_remote_code", False)),
        r_correct=float(reward_kwargs.get("r_correct", 0.1)),
        leak_penalty=float(reward_kwargs.get("leak_penalty", 0.5)),
        multilabel_match=str(reward_kwargs.get("multilabel_match", "exact")),
        enforce_no_id_in_reasoning=bool(reward_kwargs.get("enforce_no_id_in_reasoning", False)),
        max_id_mentions=int(reward_kwargs.get("max_id_mentions", 0)),
        min_reasoning_chars=int(reward_kwargs.get("min_reasoning_chars", 100)),
        min_overlap_jaccard=float(reward_kwargs.get("min_overlap_jaccard", 0.05)),
    )


__all__ = ["AcrDistillFilter", "acr_filter_from_config"]
