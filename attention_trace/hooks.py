"""
Purpose
-------
Capture attention probabilities and hidden-state diagnostics during pi0.5
action-chunk inference.

Parameters
----------
token_map: list of key-token metadata dictionaries.
token_spans: token groups produced by TokenSpanBuilder.
output_dir: run directory receiving token maps, summaries, top-k records, and
            optional compressed attention tensors.
save_full: whether to save full attention tensors as compressed npz files.
topk: number of top attended keys to save for each selected query. Use 0 to
      skip attention_topk.jsonl files.

Usage
-----
tracer = AttentionTracer(token_map, token_spans, output_dir)
tracer.start_run(prompt="pick up the cup", seed=0, mode="baseline")
tracer.capture(0, 3, "model.layers.3.self_attn", attn_probs)
tracer.finish_run()

Outputs
-------
Writes attention_summary.parquet, attention_topk.jsonl, hidden_state_norms.parquet,
config.json, token_map.json, sink_tokens.json, and optional_full_attention/*.npz
inside the provided run directory.
"""

from __future__ import annotations

import json
import pathlib
import re
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from .metrics import attention_topk, compute_attention_summary, flatten_indices
from .sinks import detect_sink_tokens


def _to_cpu_tensor(value: Any) -> torch.Tensor | None:
    if value is None or not hasattr(value, "detach"):
        return None
    return value.detach().to("cpu")


def _extract_attention(output: Any) -> torch.Tensor | None:
    if hasattr(output, "attentions"):
        return _to_cpu_tensor(output.attentions)
    if isinstance(output, dict):
        for key in ("attn_probs", "attentions", "attention_probs"):
            attn = _to_cpu_tensor(output.get(key))
            if attn is not None:
                return attn
    if isinstance(output, (tuple, list)):
        for item in output:
            attn = _to_cpu_tensor(item)
            if attn is not None and attn.ndim in {3, 4}:
                return attn
    attn = _to_cpu_tensor(output)
    if attn is not None and attn.ndim in {3, 4}:
        return attn
    return None


def _layer_from_name(name: str) -> int | None:
    matches = re.findall(r"(?:layers?|blocks?)\.(\d+)", name)
    if matches:
        return int(matches[-1])
    matches = re.findall(r"\.(\d+)\.", name)
    return int(matches[-1]) if matches else None


class AttentionTracer:
    """Aggregate attention tensors as soon as hooks expose them."""

    def __init__(
        self,
        token_map: list[dict[str, Any]],
        token_spans: dict[str, Any],
        output_dir: str | pathlib.Path,
        save_full: bool = False,
        topk: int = 6,
        layers_to_save: list[int] | None = None,
        heads_to_save: list[int] | str = "all",
        target_object: str | None = None,
        distractor_objects: list[str] | None = None,
        query_indices: list[int] | None = None,
    ) -> None:
        self.token_map = token_map
        self.token_spans = token_spans
        self.output_dir = pathlib.Path(output_dir)
        self.save_full = save_full
        self.topk = topk
        self.layers_to_save = None if layers_to_save is None else set(int(v) for v in layers_to_save)
        self.heads_to_save = heads_to_save
        self.target_object = target_object
        self.distractor_objects = distractor_objects or []
        self.query_indices = query_indices or sorted(flatten_indices(token_spans.get("continuous_action")))
        self.summary_rows: list[dict[str, Any]] = []
        self.hidden_rows: list[dict[str, Any]] = []
        self.sink_tokens_by_layer: dict[str, Any] = {}
        self.handles: list[Any] = []
        self.prompt: str | None = None
        self.seed: int | None = None
        self.mode: str | None = None

    def start_run(self, prompt: str, seed: int, mode: str) -> None:
        self.prompt = prompt
        self.seed = seed
        self.mode = mode
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.save_full:
            (self.output_dir / "optional_full_attention").mkdir(exist_ok=True)
        if self.topk > 0:
            (self.output_dir / "attention_topk.jsonl").write_text("", encoding="utf-8")
        (self.output_dir / "token_map.json").write_text(
            json.dumps(
                {
                    "token_map": self.token_map,
                    "token_spans": self.token_spans,
                    "query_token_map": self.token_map,
                    "key_token_map": self.token_map,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (self.output_dir / "config.json").write_text(
            json.dumps(
                {
                    "prompt": prompt,
                    "seed": seed,
                    "mode": mode,
                    "save_full": self.save_full,
                    "topk": self.topk,
                    "target_object": self.target_object,
                    "distractor_objects": self.distractor_objects,
                    "query_indices": self.query_indices,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def register_hooks(
        self,
        model: Any,
        module_filter: Callable[[str, Any], bool] | None = None,
    ) -> list[Any]:
        """Register forward hooks on modules whose outputs include attention."""

        if module_filter is None:
            module_filter = lambda name, _: "attn" in name.lower() or "attention" in name.lower()

        for name, module in model.named_modules():
            if not module_filter(name, module) or not hasattr(module, "register_forward_hook"):
                continue

            def hook(_module: Any, inputs: tuple[Any, ...], output: Any, module_name: str = name) -> None:
                attn = _extract_attention(output)
                if attn is None:
                    return
                layer = _layer_from_name(module_name)
                hidden = inputs[0] if inputs else None
                self.capture(
                    denoise_step=getattr(self, "current_denoise_step", 0),
                    layer_idx=-1 if layer is None else layer,
                    module_name=module_name,
                    attn_probs=attn,
                    hidden_states=hidden,
                )

            self.handles.append(module.register_forward_hook(hook))
        return self.handles

    def remove_hooks(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def capture(
        self,
        denoise_step: int,
        layer_idx: int,
        module_name: str,
        attn_probs: Any,
        hidden_states: Any | None = None,
        inference_phase: str | None = None,
        query_indices: list[int] | None = None,
        query_group: str = "continuous_action",
        query_token_map: list[dict[str, Any]] | None = None,
        key_token_map: list[dict[str, Any]] | None = None,
    ) -> None:
        attn = _to_cpu_tensor(attn_probs)
        if attn is None:
            return

        sinks = {}
        hidden = _to_cpu_tensor(hidden_states)
        if hidden is not None:
            try:
                hidden_np = hidden.float().numpy()
                sinks_by_head: dict[str, Any] = {}
                if hidden_np.ndim == 4:
                    for head_idx in range(hidden_np.shape[1]):
                        head_sinks = detect_sink_tokens(hidden_np[:, head_idx, :, :], self.token_spans)
                        if head_sinks.get("all"):
                            head_sinks["head"] = int(head_idx)
                            key = f"phase={inference_phase}/step={denoise_step}/layer={layer_idx}/head={head_idx}"
                            self.sink_tokens_by_layer[key] = head_sinks
                            sinks_by_head[str(head_idx)] = head_sinks
                    sinks = {"by_head": sinks_by_head}
                else:
                    sinks = detect_sink_tokens(hidden_np, self.token_spans)
                    if sinks.get("all"):
                        self.sink_tokens_by_layer[f"phase={inference_phase}/step={denoise_step}/layer={layer_idx}"] = sinks
                if hidden_np.ndim == 4:
                    if hidden_np.shape[0] == 1:
                        hidden_np = hidden_np[0]
                    rms = np.sqrt(np.mean(np.square(hidden_np), axis=-1))
                    for head_idx, head_values in enumerate(rms.tolist()):
                        self.hidden_rows.extend(
                            {
                                "denoise_step": int(denoise_step),
                                "inference_phase": inference_phase,
                                "layer": int(layer_idx),
                                "head": int(head_idx),
                                "module_name": module_name,
                                "token_index": int(token_idx),
                                "rms_norm": float(value),
                            }
                            for token_idx, value in enumerate(head_values)
                        )
                else:
                    if hidden_np.ndim == 3:
                        hidden_np = hidden_np[0]
                    rms = np.sqrt(np.mean(np.square(hidden_np), axis=-1))
                    self.hidden_rows.extend(
                        {
                            "denoise_step": int(denoise_step),
                            "inference_phase": inference_phase,
                            "layer": int(layer_idx),
                            "head": -1,
                            "module_name": module_name,
                            "token_index": int(idx),
                            "rms_norm": float(value),
                        }
                        for idx, value in enumerate(rms.tolist())
                    )
            except ValueError:
                sinks = {}

        self.summary_rows.extend(
            compute_attention_summary(
                attn,
                self.token_map,
                self.token_spans,
                query_indices or self.query_indices,
                sinks,
                self.target_object,
                self.distractor_objects,
                denoise_step=denoise_step,
                layer=layer_idx,
                module_name=module_name,
                inference_phase=inference_phase,
                query_group=query_group,
                query_token_map=query_token_map,
                key_token_map=key_token_map,
            )
        )

        if self.topk > 0:
            with (self.output_dir / "attention_topk.jsonl").open("a", encoding="utf-8") as handle:
                for record in attention_topk(
                    attn,
                    self.token_map,
                    query_indices or self.query_indices,
                    denoise_step=denoise_step,
                    layer=layer_idx,
                    inference_phase=inference_phase,
                    query_token_map=query_token_map,
                    key_token_map=key_token_map,
                    query_type=query_group,
                    topk=self.topk,
                ):
                    handle.write(json.dumps(record, ensure_ascii=True) + "\n")

        if self.save_full and (self.layers_to_save is None or layer_idx in self.layers_to_save):
            self._save_full_attention(attn, denoise_step, layer_idx, inference_phase=inference_phase)

    def _save_full_attention(
        self,
        attn: torch.Tensor,
        denoise_step: int,
        layer_idx: int,
        *,
        inference_phase: str | None = None,
    ) -> None:
        array = attn.to(torch.float16).numpy()
        if self.heads_to_save != "all":
            array = array[:, [int(h) for h in self.heads_to_save], :, :]
        phase = inference_phase or "unknown"
        path = (
            self.output_dir
            / "optional_full_attention"
            / f"attention_phase={phase}_step={denoise_step}_layer={layer_idx}.npz"
        )
        np.savez_compressed(path, attention=array)

    def finish_run(self, action_outputs: Any | None = None) -> None:
        pd.DataFrame(self.summary_rows).to_parquet(self.output_dir / "attention_summary.parquet", index=False)
        pd.DataFrame(self.hidden_rows).to_parquet(self.output_dir / "hidden_state_norms.parquet", index=False)
        (self.output_dir / "sink_tokens.json").write_text(
            json.dumps(self.sink_tokens_by_layer, indent=2),
            encoding="utf-8",
        )
        if action_outputs is not None:
            if isinstance(action_outputs, dict):
                arrays = {str(k): np.asarray(v) for k, v in action_outputs.items()}
            else:
                arrays = {"actions": np.asarray(action_outputs)}
            np.savez_compressed(self.output_dir / "action_outputs.npz", **arrays)
        self.remove_hooks()
