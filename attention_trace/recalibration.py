"""
Purpose
-------
Apply IGAR-style text attention recalibration to attention probabilities.

Parameters
----------
attn_row: one attention row with shape [key_len].
attn_probs: attention tensor with shape [B, H, Q, K], [H, Q, K], or [Q, K].
text_sink_indices: key indices of text sink tokens.
non_sink_text_indices: key indices of non-sink text tokens.
p: retention factor for original sink-token attention.
eps: numerical epsilon.

Usage
-----
from attention_trace.recalibration import recalibrate_attention_row
new_row = recalibrate_attention_row(row, text_sink_indices, non_sink_text_indices)

Outputs
-------
Returns recalibrated attention tensors in memory. The caller should use the
returned probabilities before `context = attn_probs @ value_states` and may
save before/after summaries under outputs/attention_trace/.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch


def _valid_indices(indices: Iterable[int], length: int, device: torch.device) -> torch.Tensor:
    clean = sorted({int(i) for i in indices if 0 <= int(i) < length})
    if not clean:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.tensor(clean, dtype=torch.long, device=device)


def recalibrate_attention_row(
    attn_row: torch.Tensor,
    text_sink_indices: Iterable[int],
    non_sink_text_indices: Iterable[int],
    p: float = 0.6,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Move released attention mass from text sinks to non-sink text tokens."""

    if attn_row.ndim != 1:
        raise ValueError(f"attn_row must be one-dimensional, got shape {tuple(attn_row.shape)}")

    out = attn_row.clone()
    original = attn_row
    sink = _valid_indices(text_sink_indices, out.shape[-1], out.device)
    non_sink = _valid_indices(non_sink_text_indices, out.shape[-1], out.device)
    if sink.numel() == 0 or non_sink.numel() == 0:
        return out

    released = (1.0 - p) * original.index_select(0, sink).sum()
    out.index_copy_(0, sink, p * original.index_select(0, sink))

    denom = original.index_select(0, non_sink).sum().clamp_min(eps)
    updated_non_sink = original.index_select(0, non_sink) + released * original.index_select(0, non_sink) / denom
    out.index_copy_(0, non_sink, updated_non_sink)

    drift = out.sum() - original.sum()
    if torch.abs(drift) > 1e-4:
        out = out / out.sum().clamp_min(eps) * original.sum()
    return out


def recalibrate_attention_probs(
    attn_probs: torch.Tensor,
    text_sink_indices: Iterable[int],
    non_sink_text_indices: Iterable[int],
    query_indices: Iterable[int] | None = None,
    head_indices: Iterable[int] | None = None,
    p: float = 0.6,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Recalibrate selected query/head rows in an attention tensor."""

    out = attn_probs.clone()
    original_ndim = out.ndim
    if original_ndim == 2:
        out = out.unsqueeze(0).unsqueeze(0)
    elif original_ndim == 3:
        out = out.unsqueeze(0)
    elif original_ndim != 4:
        raise ValueError(f"attn_probs must have shape [Q,K], [H,Q,K], or [B,H,Q,K], got {tuple(out.shape)}")

    _, num_heads, query_len, _ = out.shape
    heads = list(range(num_heads)) if head_indices is None else [int(h) for h in head_indices if 0 <= int(h) < num_heads]
    queries = (
        list(range(query_len))
        if query_indices is None
        else [int(q) for q in query_indices if 0 <= int(q) < query_len]
    )

    for batch_idx in range(out.shape[0]):
        for head_idx in heads:
            for query_idx in queries:
                out[batch_idx, head_idx, query_idx] = recalibrate_attention_row(
                    out[batch_idx, head_idx, query_idx],
                    text_sink_indices,
                    non_sink_text_indices,
                    p=p,
                    eps=eps,
                )

    if original_ndim == 2:
        return out[0, 0]
    if original_ndim == 3:
        return out[0]
    return out


class AttentionRecalibrationController:
    """Small state object for attention modules that expose attn_probs before V multiplication."""

    def __init__(
        self,
        text_sink_indices: Iterable[int],
        non_sink_text_indices: Iterable[int],
        query_indices: Iterable[int] | None = None,
        head_indices: Iterable[int] | None = None,
        p: float = 0.6,
    ) -> None:
        self.text_sink_indices = list(text_sink_indices)
        self.non_sink_text_indices = list(non_sink_text_indices)
        self.query_indices = None if query_indices is None else list(query_indices)
        self.head_indices = None if head_indices is None else list(head_indices)
        self.p = p

    def __call__(self, attn_probs: torch.Tensor, _: Any = None) -> torch.Tensor:
        return recalibrate_attention_probs(
            attn_probs,
            self.text_sink_indices,
            self.non_sink_text_indices,
            query_indices=self.query_indices,
            head_indices=self.head_indices,
            p=self.p,
        )
