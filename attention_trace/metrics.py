"""
Purpose
-------
Aggregate attention probabilities into interpretable token-group metrics.

Parameters
----------
attn_probs: attention tensor with shape [B, H, Q, K], [H, Q, K], or [Q, K].
token_map: list of token metadata dictionaries.
token_spans: token groups from TokenSpanBuilder.
query_indices: query positions to summarize.
sink_tokens: sink-token dictionary from detect_sink_tokens.
target_object: object word expected by the prompt.
distractor_objects: object labels or words used as contrast classes.

Usage
-----
rows = compute_attention_summary(attn, token_map, token_spans, action_indices, sinks, "bowl", ["cup"])

Outputs
-------
Returns rows suitable for attention_summary.parquet in each run directory under
outputs/attention_trace/.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from .token_map import find_subword_indices


def _to_numpy(array: Any) -> np.ndarray:
    if hasattr(array, "detach"):
        array = array.detach()
    if hasattr(array, "cpu"):
        array = array.cpu()
    if hasattr(array, "float"):
        array = array.float()
    return np.asarray(array)


def flatten_indices(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, dict):
        out: set[int] = set()
        for nested in value.values():
            out.update(flatten_indices(nested))
        return out
    if isinstance(value, range):
        return set(value)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return {int(v) for v in value}
    return {int(value)}


def _valid(indices: Iterable[int], length: int) -> list[int]:
    return sorted({int(i) for i in indices if 0 <= int(i) < length})


def _mass(row: np.ndarray, indices: Iterable[int]) -> float:
    valid = _valid(indices, row.shape[-1])
    if not valid:
        return 0.0
    return float(row[valid].sum())


def _entropy(row: np.ndarray, indices: Iterable[int], eps: float = 1e-12) -> float:
    valid = _valid(indices, row.shape[-1])
    if not valid:
        return 0.0
    probs = row[valid].astype(np.float64)
    total = probs.sum()
    if total <= eps:
        return 0.0
    probs = probs / total
    return float(-(probs * np.log(probs + eps)).sum())


def _type_lookup(token_map: list[dict[str, Any]], key_len: int) -> list[dict[str, Any]]:
    lookup = [{"index": idx, "type": "unknown"} for idx in range(key_len)]
    for entry in token_map:
        idx = int(entry.get("index", -1))
        if 0 <= idx < key_len:
            lookup[idx] = entry
    return lookup


def _position_lookup(token_map: list[dict[str, Any]], length: int) -> list[dict[str, Any]]:
    lookup = [{"index": idx, "type": "unknown"} for idx in range(length)]
    for pos, entry in enumerate(token_map[:length]):
        lookup[pos] = entry
    return lookup


def object_patch_indices(token_map: list[dict[str, Any]], object_name: str) -> set[int]:
    target = object_name.lower().replace(" ", "_")
    out: set[int] = set()
    for entry in token_map:
        if entry.get("type") != "image":
            continue
        label = entry.get("object_label")
        if label is not None and str(label).lower().replace(" ", "_") == target:
            out.add(int(entry["index"]))
    return out


def compute_attention_summary(
    attn_probs: Any,
    token_map: list[dict[str, Any]],
    token_spans: dict[str, Any],
    query_indices: Iterable[int] | None,
    sink_tokens: dict[str, Any] | None,
    target_object: str | None,
    distractor_objects: Iterable[str] | None = None,
    *,
    denoise_step: int | None = None,
    layer: int | None = None,
    module_name: str | None = None,
    inference_phase: str | None = None,
    query_group: str = "continuous_action",
    query_token_map: list[dict[str, Any]] | None = None,
    key_token_map: list[dict[str, Any]] | None = None,
    eps: float = 1e-12,
) -> list[dict[str, Any]]:
    """Compute group attention mass for selected query rows."""

    attn = _to_numpy(attn_probs)
    if attn.ndim == 2:
        attn = attn[None, None, :, :]
    elif attn.ndim == 3:
        attn = attn[None, :, :, :]
    elif attn.ndim != 4:
        raise ValueError(f"attn_probs must be [Q,K], [H,Q,K], or [B,H,Q,K], got {attn.shape}")

    _, num_heads, query_len, key_len = attn.shape
    if query_indices is None:
        queries = list(range(query_len))
    else:
        queries = _valid(query_indices, query_len)

    image_indices = flatten_indices(token_spans.get("image"))
    text_indices = flatten_indices(token_spans.get("text"))
    prompt_indices = flatten_indices(token_spans.get("prompt"))
    state_indices = flatten_indices(token_spans.get("state"))
    special_indices = flatten_indices(token_spans.get("special"))
    proprio_indices = flatten_indices(token_spans.get("proprio"))
    action_indices = flatten_indices(token_spans.get("continuous_action")) | flatten_indices(token_spans.get("fast_action"))
    sink_tokens = sink_tokens or {}
    sink_tokens_by_head = sink_tokens.get("by_head", {}) if isinstance(sink_tokens, dict) else {}

    target_word_indices = set(find_subword_indices(target_object, token_map)) if target_object else set()
    target_patch_indices = object_patch_indices(token_map, target_object) if target_object else set()
    distractor_patch_indices: set[int] = set()
    distractor_word_indices: set[int] = set()
    for obj in distractor_objects or []:
        distractor_patch_indices.update(object_patch_indices(token_map, obj))
        distractor_word_indices.update(find_subword_indices(obj, token_map))

    key_lookup = _position_lookup(key_token_map or token_map, key_len)
    query_lookup = _position_lookup(query_token_map or token_map, query_len)
    rows: list[dict[str, Any]] = []
    mean_attn = attn.mean(axis=0)
    for head_idx in range(num_heads):
        head_sink_tokens = sink_tokens_by_head.get(str(head_idx), sink_tokens)
        text_sink_indices = flatten_indices(head_sink_tokens.get("text"))
        prompt_sink_indices = flatten_indices(head_sink_tokens.get("prompt"))
        state_sink_indices = flatten_indices(head_sink_tokens.get("state"))
        special_sink_indices = flatten_indices(head_sink_tokens.get("special"))
        visual_sink_indices = flatten_indices(head_sink_tokens.get("visual"))
        non_sink_text_indices = text_indices - text_sink_indices
        non_sink_visual_indices = image_indices - visual_sink_indices
        for query_idx in queries:
            row = mean_attn[head_idx, query_idx]
            top1_key = int(row.argmax()) if row.size else -1
            top1_entry = key_lookup[top1_key] if top1_key >= 0 else {"type": "unknown"}
            query_entry = query_lookup[query_idx]
            rows.append(
                {
                    "inference_phase": inference_phase,
                    "denoise_step": denoise_step,
                    "layer": layer,
                    "module_name": module_name,
                    "head": int(head_idx),
                    "query_group": query_group,
                    "query_position": int(query_idx),
                    "query_index": int(query_entry.get("index", query_idx)),
                    "query_token_index": int(query_entry.get("index", query_idx)),
                    "query_type": query_entry.get("type", "unknown"),
                    "query_len": int(query_len),
                    "key_len": int(key_len),
                    "text_mass": _mass(row, text_indices),
                    "prompt_mass": _mass(row, prompt_indices),
                    "state_mass": _mass(row, state_indices),
                    "special_mass": _mass(row, special_indices),
                    "text_sink_mass": _mass(row, text_sink_indices),
                    "prompt_sink_mass": _mass(row, prompt_sink_indices),
                    "state_sink_mass": _mass(row, state_sink_indices),
                    "special_sink_mass": _mass(row, special_sink_indices),
                    "non_sink_text_mass": _mass(row, non_sink_text_indices),
                    "object_word_mass": _mass(row, target_word_indices),
                    "target_object_word_mass": _mass(row, target_word_indices),
                    "distractor_object_word_mass": _mass(row, distractor_word_indices),
                    "visual_mass": _mass(row, image_indices),
                    "visual_sink_mass": _mass(row, visual_sink_indices),
                    "non_sink_visual_mass": _mass(row, non_sink_visual_indices),
                    "target_object_patch_mass": _mass(row, target_patch_indices),
                    "distractor_object_patch_mass": _mass(row, distractor_patch_indices),
                    "proprio_mass": _mass(row, proprio_indices),
                    "action_mass": _mass(row, action_indices),
                    "visual_entropy": _entropy(row, image_indices, eps=eps),
                    "text_entropy": _entropy(row, text_indices, eps=eps),
                    "top1_key_position": top1_key,
                    "top1_key_index": int(top1_entry.get("index", top1_key)) if top1_key >= 0 else -1,
                    "top1_key_token_index": int(top1_entry.get("index", top1_key)) if top1_key >= 0 else -1,
                    "top1_key_type": top1_entry.get("type", "unknown"),
                    "top1_attention": float(row[top1_key]) if top1_key >= 0 else 0.0,
                }
            )
    return rows


def attention_topk(
    attn_probs: Any,
    token_map: list[dict[str, Any]],
    query_indices: Iterable[int],
    *,
    denoise_step: int | None,
    layer: int | None,
    inference_phase: str | None = None,
    query_token_map: list[dict[str, Any]] | None = None,
    key_token_map: list[dict[str, Any]] | None = None,
    query_type: str | None = None,
    topk: int = 6,
) -> list[dict[str, Any]]:
    """Return top-k key metadata for selected query rows."""

    if topk <= 0:
        return []

    attn = _to_numpy(attn_probs)
    if attn.ndim == 2:
        attn = attn[None, None, :, :]
    elif attn.ndim == 3:
        attn = attn[None, :, :, :]
    mean_attn = attn.mean(axis=0)
    _, query_len, key_len = mean_attn.shape
    queries = _valid(query_indices, query_len)
    key_lookup = _position_lookup(key_token_map or token_map, key_len)
    query_lookup = _position_lookup(query_token_map or token_map, query_len)

    records: list[dict[str, Any]] = []
    for head_idx in range(mean_attn.shape[0]):
        for query_idx in queries:
            row = mean_attn[head_idx, query_idx]
            k = min(topk, row.shape[-1])
            if k <= 0:
                continue
            top_indices = np.argsort(row)[-k:][::-1]
            query_entry = query_lookup[query_idx]
            records.append(
                {
                    "inference_phase": inference_phase,
                    "denoise_step": denoise_step,
                    "layer": layer,
                    "head": int(head_idx),
                    "query_position": int(query_idx),
                    "query_index": int(query_entry.get("index", query_idx)),
                    "query_token_index": int(query_entry.get("index", query_idx)),
                    "query_type": query_type or query_entry.get("type", "unknown"),
                    "topk": [
                        {
                            "key_position": int(idx),
                            "key_index": int(key_lookup[int(idx)].get("index", idx)),
                            "key_token_index": int(key_lookup[int(idx)].get("index", idx)),
                            "key_type": key_lookup[int(idx)].get("type", "unknown"),
                            "token_str": key_lookup[int(idx)].get("token_str"),
                            "token_text": key_lookup[int(idx)].get("token_text"),
                            "special_role": key_lookup[int(idx)].get("special_role"),
                            "camera": key_lookup[int(idx)].get("camera"),
                            "patch_row": key_lookup[int(idx)].get("patch_row"),
                            "patch_col": key_lookup[int(idx)].get("patch_col"),
                            "patch_box_xyxy": key_lookup[int(idx)].get("patch_box_xyxy"),
                            "raw_image_path": key_lookup[int(idx)].get("raw_image_path"),
                            "raw_image_height": key_lookup[int(idx)].get("raw_image_height"),
                            "raw_image_width": key_lookup[int(idx)].get("raw_image_width"),
                            "object_label": key_lookup[int(idx)].get("object_label"),
                            "attention": float(row[int(idx)]),
                        }
                        for idx in top_indices
                    ],
                }
            )
    return records
