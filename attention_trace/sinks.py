"""
Purpose
-------
Detect layer-specific sink tokens from hidden states for attention tracing.

Parameters
----------
hidden_states: tensor-like array with shape [N, D] or [B, N, D].
token_spans: token index groups produced by TokenSpanBuilder.
gamma: spike-ratio threshold used to select hidden dimensions.
tau: absolute activation threshold used to mark sink tokens.
top_k_dims: fallback number of largest spike dimensions when no dimension
            passes gamma.

Usage
-----
from attention_trace.sinks import detect_sink_tokens
sinks = detect_sink_tokens(hidden_states, token_spans)

Outputs
-------
Returns a dictionary with sink-token indices grouped as all, visual, text,
prompt, state, special, proprio, and action. Callers usually save it as sink_tokens.json inside one
run directory, for example:
outputs/attention_trace/scene_id=0/seed=0/prompt=pick_up_cup/baseline/
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


def _to_numpy(array: Any) -> np.ndarray:
    if hasattr(array, "detach"):
        array = array.detach()
    if hasattr(array, "cpu"):
        array = array.cpu()
    if hasattr(array, "float"):
        array = array.float()
    return np.asarray(array)


def _flatten_indices(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, dict):
        out: set[int] = set()
        for nested in value.values():
            out.update(_flatten_indices(nested))
        return out
    if isinstance(value, range):
        return set(value)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return {int(v) for v in value}
    return {int(value)}


def detect_sink_tokens(
    hidden_states: Any,
    token_spans: dict[str, Any],
    gamma: float = 3.0,
    tau: float = 20.0,
    top_k_dims: int = 5,
    max_spike_dims: int = 20,
    max_token_spike_dims: int = 20,
    eps: float = 1e-8,
) -> dict[str, Any]:
    """Detect sink tokens using hidden-state spike dimensions."""

    hidden = _to_numpy(hidden_states)
    if hidden.ndim == 3:
        hidden = hidden[0]
    if hidden.ndim != 2:
        raise ValueError(f"hidden_states must have shape [N, D] or [B, N, D], got {hidden.shape}")

    abs_hidden = np.abs(hidden)
    spike_ratio = abs_hidden.max(axis=0) / (abs_hidden.mean(axis=0) + eps)
    candidate_spike_dims = np.flatnonzero(spike_ratio > gamma)
    if candidate_spike_dims.size == 0 and top_k_dims > 0:
        candidate_spike_dims = np.argsort(spike_ratio)[-top_k_dims:]

    if candidate_spike_dims.size == 0:
        sink_indices: set[int] = set()
    else:
        sink_indices = set(np.flatnonzero(abs_hidden[:, candidate_spike_dims].max(axis=1) > tau).astype(int).tolist())

    spike_dims = candidate_spike_dims
    if max_spike_dims > 0 and spike_dims.size > max_spike_dims:
        order = np.argsort(spike_ratio[spike_dims])[::-1]
        spike_dims = spike_dims[order[:max_spike_dims]]

    sink_token_spike_dims: dict[str, list[int]] = {}
    for token_idx in sorted(sink_indices):
        token_dims = candidate_spike_dims[abs_hidden[token_idx, candidate_spike_dims] > tau]
        if max_token_spike_dims > 0 and token_dims.size > max_token_spike_dims:
            order = np.argsort(abs_hidden[token_idx, token_dims])[::-1]
            token_dims = token_dims[order[:max_token_spike_dims]]
        sink_token_spike_dims[str(int(token_idx))] = sorted(int(d) for d in token_dims.tolist())

    visual_indices = _flatten_indices(token_spans.get("image"))
    text_indices = _flatten_indices(token_spans.get("text"))
    prompt_indices = _flatten_indices(token_spans.get("prompt"))
    state_indices = _flatten_indices(token_spans.get("state"))
    special_indices = _flatten_indices(token_spans.get("special"))
    proprio_indices = _flatten_indices(token_spans.get("proprio"))
    action_indices = _flatten_indices(token_spans.get("continuous_action")) | _flatten_indices(
        token_spans.get("fast_action")
    )

    return {
        "all": sorted(sink_indices),
        "visual": sorted(sink_indices & visual_indices),
        "text": sorted(sink_indices & text_indices),
        "prompt": sorted(sink_indices & prompt_indices),
        "state": sorted(sink_indices & state_indices),
        "special": sorted(sink_indices & special_indices),
        "proprio": sorted(sink_indices & proprio_indices),
        "action": sorted(sink_indices & action_indices),
        "spike_dims": sorted(int(d) for d in spike_dims.tolist()),
        "sink_token_spike_dims": sink_token_spike_dims,
    }
