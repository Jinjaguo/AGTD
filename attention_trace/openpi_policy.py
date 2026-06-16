"""
Purpose
-------
Connect attention tracing utilities to the existing OpenPI websocket inference
path used by LIBERO rollouts.

Parameters
----------
AttentionTracingPolicy(policy, record_root, mode, ...)
  policy: PyTorch OpenPI policy created by `start_igar_attention_server.py`.
  record_root: root directory for attention trace outputs.
  mode: baseline, recalibrated, random_text, or visual_uniform.
  save_full_attention: whether to save compressed full attention tensors.
  layers_to_save: optional layer ids for full attention dumps.
  topk: number of top attended keys saved per action query. Use 0 to skip
        attention_topk.jsonl files.
  sink_strategy: value_projection detects sink keys from value-state norms.

Usage
-----
from attention_trace.openpi_policy import AttentionTracingPolicy

policy = AttentionTracingPolicy(
    policy=policy,
    record_root="attention_analysis/outputs/attention_trace",
    mode="baseline",
)
result = policy.infer(obs)

Outputs
-------
For each websocket action request, files are saved under:
<record_root>/trial_<trial_id>/chunk_<chunk_id>/

The run directory contains token_map.json, config.json,
attention_summary.parquet, attention_topk.jsonl, hidden_state_norms.parquet,
sink_tokens.json, and action_outputs.npz.
"""

from __future__ import annotations

import dataclasses
import json
import math
import pathlib
import random
import types
from typing import Any, Literal

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from .hooks import AttentionTracer
from .metrics import flatten_indices
from .recalibration import recalibrate_attention_probs
from .sinks import detect_sink_tokens


Mode = Literal["baseline", "recalibrated", "random_text", "visual_uniform"]
SinkStrategy = Literal["value_projection", "none"]


def _to_numpy(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=torch.float32).cpu().numpy()
    return np.asarray(value)


def _sanitize_name(value: str | None, fallback: str) -> str:
    raw = (value or fallback).strip()
    safe = raw.replace("/", "_").replace("\\", "_").replace(" ", "_")
    return safe or fallback


def _infer_target_object(prompt: str | None) -> str | None:
    if not prompt:
        return None
    words = [w.lower() for w in prompt.replace("_", " ").split()]
    stops = {"the", "a", "an", "up", "pick", "move", "put", "to", "on", "in", "front", "of", "and"}
    content = [w for w in words if w not in stops]
    return content[-1] if content else None


def _flatten_for_sink(value_states: torch.Tensor) -> torch.Tensor:
    """Convert [B, H, K, D] value states into [B, K, H*D] token features."""

    return value_states.transpose(1, 2).contiguous().flatten(start_dim=2)


def _image_tensor_to_uint8_hwc(image: Any) -> np.ndarray:
    array = _to_numpy(image)
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"Expected image with shape [H,W,C] or [C,H,W], got {array.shape}")
    if array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
        array = np.moveaxis(array, 0, -1)
    array = array[..., :3]
    if array.dtype != np.uint8:
        array = array.astype(np.float32)
        min_value = float(np.nanmin(array))
        max_value = float(np.nanmax(array))
        if min_value < -0.01:
            array = (array + 1.0) / 2.0
        elif max_value > 1.5:
            array = array / 255.0
        array = np.clip(array, 0.0, 1.0)
        array = (array * 255.0).round().astype(np.uint8)
    return np.ascontiguousarray(array)


def _save_input_images(
    images: Any,
    camera_names: tuple[str, ...],
    run_dir: pathlib.Path,
) -> dict[str, dict[str, Any]]:
    image_dir = run_dir / "input_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, dict[str, Any]] = {}
    for idx, image in enumerate(images):
        camera = camera_names[idx] if idx < len(camera_names) else f"camera_{idx}"
        array = _image_tensor_to_uint8_hwc(image)
        filename = f"{camera}.png"
        imageio.imwrite(image_dir / filename, array)
        metadata[camera] = {
            "camera": camera,
            "path": str(pathlib.Path("input_images") / filename),
            "height": int(array.shape[0]),
            "width": int(array.shape[1]),
            "channels": int(array.shape[2]),
        }
    (image_dir / "image_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


@dataclasses.dataclass
class _PrefixInfo:
    image_token_counts: dict[str, int] = dataclasses.field(default_factory=dict)
    image_metadata: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)
    text_token_ids: list[int] = dataclasses.field(default_factory=list)
    text_token_strings: list[str | None] = dataclasses.field(default_factory=list)
    text_token_texts: list[str | None] = dataclasses.field(default_factory=list)
    text_token_types: list[str] = dataclasses.field(default_factory=list)
    text_token_special_roles: list[str | None] = dataclasses.field(default_factory=list)
    text_mask: list[bool] = dataclasses.field(default_factory=list)
    prefix_len: int = 0


def _iter_transforms(transform: Any):
    if transform is None:
        return
    children = getattr(transform, "transforms", None)
    if children is not None:
        for child in children:
            yield from _iter_transforms(child)
        return
    yield transform


def _find_text_tokenizer(policy: Any) -> Any | None:
    for transform in _iter_transforms(getattr(policy, "_input_transform", None)):
        tokenizer = getattr(transform, "tokenizer", None)
        if tokenizer is not None:
            return tokenizer
    return None


def _token_piece(tokenizer: Any | None, token_id: int) -> str | None:
    if tokenizer is None:
        return None
    inner = getattr(tokenizer, "_tokenizer", None)
    if inner is not None:
        return _token_piece(inner, token_id)
    inner = getattr(tokenizer, "_paligemma_tokenizer", None)
    if inner is not None:
        return _token_piece(inner, token_id)
    for method_name in ("id_to_piece", "IdToPiece"):
        method = getattr(tokenizer, method_name, None)
        if method is not None:
            try:
                return str(method(int(token_id)))
            except Exception:
                pass
    method = getattr(tokenizer, "convert_ids_to_tokens", None)
    if method is not None:
        try:
            return str(method([int(token_id)])[0])
        except Exception:
            pass
    return _token_text(tokenizer, token_id)


def _token_text(tokenizer: Any | None, token_id: int) -> str | None:
    if tokenizer is None:
        return None
    inner = getattr(tokenizer, "_tokenizer", None)
    if inner is not None:
        return _token_text(inner, token_id)
    inner = getattr(tokenizer, "_paligemma_tokenizer", None)
    if inner is not None:
        return _token_text(inner, token_id)
    method = getattr(tokenizer, "decode", None)
    if method is not None:
        try:
            return str(method([int(token_id)]))
        except Exception:
            try:
                return str(method(int(token_id)))
            except Exception:
                pass
    return None


def _tokenizer_int_attr(tokenizer: Any | None, name: str) -> int | None:
    if tokenizer is None:
        return None
    inner = getattr(tokenizer, "_tokenizer", None)
    if inner is not None:
        return _tokenizer_int_attr(inner, name)
    inner = getattr(tokenizer, "_paligemma_tokenizer", None)
    if inner is not None:
        return _tokenizer_int_attr(inner, name)
    value = getattr(tokenizer, name, None)
    if value is None:
        return None
    try:
        return int(value() if callable(value) else value)
    except Exception:
        return None


def _special_token_role(tokenizer: Any | None, token_id: int) -> str | None:
    special_methods = (
        ("bos", "bos_id"),
        ("eos", "eos_id"),
        ("pad", "pad_id"),
        ("unk", "unk_id"),
    )
    for role, attr_name in special_methods:
        special_id = _tokenizer_int_attr(tokenizer, attr_name)
        if special_id is not None and special_id >= 0 and int(token_id) == special_id:
            return role

    piece = (_token_piece(tokenizer, token_id) or "").strip().lower()
    text = (_token_text(tokenizer, token_id) or "").strip().lower()
    marker = piece or text
    if marker in {"<s>", "<bos>", "[bos]", "<start_of_text>"}:
        return "bos"
    if marker in {"</s>", "<eos>", "[eos]", "<end_of_text>"}:
        return "eos"
    if marker in {"<pad>", "[pad]", "<padding>"}:
        return "pad"
    if marker in {"<unk>", "[unk]"}:
        return "unk"
    return None


def _normalized_token_text(token_piece: str | None, token_text: str | None) -> str:
    value = token_text if token_text is not None else token_piece
    value = (value or "").replace("▁", " ").replace("Ġ", " ")
    return value.strip().lower()


def _is_template_delimiter(value: str) -> bool:
    return value in {"", ":", ",", ";", "\\n", "\n", "action:", "state:", "task:"}


def _classify_text_tokens(
    token_ids: list[int],
    token_pieces: list[str | None],
    token_texts: list[str | None],
    tokenizer: Any | None,
) -> tuple[list[str], list[str | None]]:
    """Classify prefix language tokens as tokenizer/template special, prompt, or state."""

    special_roles = [_special_token_role(tokenizer, token_id) for token_id in token_ids]
    normalized = [_normalized_token_text(piece, text) for piece, text in zip(token_pieces, token_texts)]

    def find_marker(marker: str) -> int | None:
        for idx, value in enumerate(normalized):
            if value == marker:
                return idx
        return None

    task_pos = find_marker("task")
    state_pos = find_marker("state")
    action_pos = find_marker("action")

    token_types: list[str] = []
    updated_special_roles: list[str | None] = []
    for idx, value in enumerate(normalized):
        special_role = special_roles[idx]
        token_type = "special"

        if special_role is not None:
            token_type = "special"
        elif idx in {task_pos, state_pos, action_pos} or _is_template_delimiter(value):
            special_role = "template"
            token_type = "special"
        elif state_pos is not None and idx > state_pos and (action_pos is None or idx < action_pos):
            token_type = "state"
        elif task_pos is not None and idx > task_pos and (state_pos is None or idx < state_pos):
            token_type = "prompt"
        elif state_pos is None and action_pos is None:
            token_type = "prompt"
        else:
            special_role = "template"
            token_type = "special"

        token_types.append(token_type)
        updated_special_roles.append(special_role)

    return token_types, updated_special_roles


class _OpenPIAttentionCollector:
    """Patch eager OpenPI attention so captured probabilities are the ones used for V multiplication."""

    def __init__(
        self,
        pi_model: Any,
        *,
        record_root: pathlib.Path,
        mode: Mode,
        save_full_attention: bool,
        layers_to_save: list[int] | None,
        topk: int,
        sink_strategy: SinkStrategy,
        p: float,
        camera_names: tuple[str, ...],
        text_tokenizer: Any | None,
        disable_torch_compile: bool,
    ) -> None:
        self._pi_model = pi_model
        self._record_root = record_root
        self._mode = mode
        self._save_full_attention = save_full_attention
        self._layers_to_save = layers_to_save
        self._topk = topk
        self._sink_strategy = sink_strategy
        self._p = p
        self._camera_names = camera_names
        self._text_tokenizer = text_tokenizer
        self._disable_torch_compile = disable_torch_compile

        self._active = False
        self._phase: str | None = None
        self._denoise_step = -1
        self._tracer: AttentionTracer | None = None
        self._run_context: dict[str, Any] = {}
        self._prefix_info = _PrefixInfo()
        self._original_embed_prefix = None
        self._original_denoise_step = None
        self._original_eager_attention_forward = None
        self._modeling_gemma = None
        self._install()

    def _install(self) -> None:
        if self._disable_torch_compile:
            self._pi_model.sample_actions = types.MethodType(type(self._pi_model).sample_actions, self._pi_model)
        self._patch_embed_prefix()
        self._patch_denoise_step()
        self._patch_eager_attention()

    def _patch_embed_prefix(self) -> None:
        self._original_embed_prefix = self._pi_model.embed_prefix

        def wrapped(images, img_masks, lang_tokens, lang_masks):
            prefix_embs, prefix_pad_masks, prefix_att_masks = self._original_embed_prefix(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
            )
            if self._active:
                self._record_prefix_info(images, lang_tokens, lang_masks, prefix_embs)
                self._ensure_tracer(prefix_embs.device)
            return prefix_embs, prefix_pad_masks, prefix_att_masks

        self._pi_model.embed_prefix = wrapped

    def _record_prefix_info(self, images, lang_tokens, lang_masks, prefix_embs) -> None:
        text_len = int(lang_tokens.shape[1])
        prefix_len = int(prefix_embs.shape[1])
        num_images = max(1, len(images))
        image_total = max(0, prefix_len - text_len)
        base_count = image_total // num_images
        remainder = image_total % num_images
        counts = {}
        for idx in range(num_images):
            name = self._camera_names[idx] if idx < len(self._camera_names) else f"camera_{idx}"
            counts[name] = base_count + (1 if idx < remainder else 0)

        image_metadata = _save_input_images(
            images,
            self._camera_names,
            pathlib.Path(self._run_context["run_dir"]),
        )
        token_ids = [int(v) for v in lang_tokens[0].detach().cpu().tolist()]
        token_pieces = [_token_piece(self._text_tokenizer, token_id) for token_id in token_ids]
        token_texts = [_token_text(self._text_tokenizer, token_id) for token_id in token_ids]
        token_types, special_roles = _classify_text_tokens(
            token_ids,
            token_pieces,
            token_texts,
            self._text_tokenizer,
        )
        self._prefix_info = _PrefixInfo(
            image_token_counts=counts,
            image_metadata=image_metadata,
            text_token_ids=token_ids,
            text_token_strings=token_pieces,
            text_token_texts=token_texts,
            text_token_types=token_types,
            text_token_special_roles=special_roles,
            text_mask=[bool(v) for v in lang_masks[0].detach().cpu().tolist()],
            prefix_len=prefix_len,
        )

    def _patch_denoise_step(self) -> None:
        self._original_denoise_step = self._pi_model.denoise_step

        def wrapped(state, prefix_pad_masks, past_key_values, x_t, timestep):
            if not self._active:
                return self._original_denoise_step(state, prefix_pad_masks, past_key_values, x_t, timestep)
            self._phase = "denoise"
            self._denoise_step += 1
            try:
                return self._original_denoise_step(state, prefix_pad_masks, past_key_values, x_t, timestep)
            finally:
                self._phase = None

        self._pi_model.denoise_step = wrapped

    def _patch_eager_attention(self) -> None:
        from transformers.models.gemma import modeling_gemma

        self._modeling_gemma = modeling_gemma
        self._original_eager_attention_forward = modeling_gemma.eager_attention_forward

        def wrapped(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
            if not self._active or self._tracer is None or self._phase != "denoise":
                return self._original_eager_attention_forward(
                    module,
                    query,
                    key,
                    value,
                    attention_mask,
                    scaling,
                    dropout=dropout,
                    **kwargs,
                )

            key_states = modeling_gemma.repeat_kv(key, module.num_key_value_groups)
            value_states = modeling_gemma.repeat_kv(value, module.num_key_value_groups)
            attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
            attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
            layer_idx = int(getattr(module, "layer_idx", -1))
            phase, query_map, key_map, query_group = self._attention_layout(
                query_len=int(attn_weights.shape[-2]),
                key_len=int(attn_weights.shape[-1]),
            )

            value_proxy = _flatten_for_sink(value_states)
            sink_tokens = self._detect_sinks(value_proxy)
            attn_for_model = self._apply_mode(attn_weights, sink_tokens)

            self._tracer.capture(
                denoise_step=self._denoise_step if phase == "action_denoise" else -1,
                layer_idx=layer_idx,
                module_name=f"{phase}.layers.{layer_idx}.self_attn",
                attn_probs=attn_for_model,
                hidden_states=value_states,
                inference_phase=phase,
                query_indices=list(range(attn_for_model.shape[-2])),
                query_group=query_group,
                query_token_map=query_map,
                key_token_map=key_map,
            )

            attn_output = torch.matmul(attn_for_model, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
            return attn_output, attn_for_model

        modeling_gemma.eager_attention_forward = wrapped

    def _attention_layout(
        self,
        *,
        query_len: int,
        key_len: int,
    ) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], str]:
        token_map = self._tracer.token_map if self._tracer is not None else []
        prefix_len = int(self._prefix_info.prefix_len)
        action_indices = sorted(flatten_indices(self._tracer.token_spans.get("continuous_action"))) if self._tracer else []

        if query_len == prefix_len and key_len == prefix_len:
            return (
                "prefix_prefill",
                token_map[:query_len],
                token_map[:key_len],
                "prefix_token",
            )

        action_query_map = [token_map[idx] for idx in action_indices[:query_len] if idx < len(token_map)]
        if len(action_query_map) < query_len:
            action_query_map.extend({"index": prefix_len + i, "type": "continuous_action"} for i in range(len(action_query_map), query_len))
        return (
            "action_denoise",
            action_query_map,
            token_map[:key_len],
            "continuous_action",
        )

    def _detect_sinks(self, value_proxy: torch.Tensor) -> dict[str, Any]:
        if self._sink_strategy == "none":
            return {
                "all": [],
                "visual": [],
                "text": [],
                "prompt": [],
                "state": [],
                "special": [],
                "proprio": [],
                "action": [],
                "spike_dims": [],
                "sink_token_spike_dims": {},
            }
        if self._tracer is None:
            return {
                "all": [],
                "visual": [],
                "text": [],
                "prompt": [],
                "state": [],
                "special": [],
                "proprio": [],
                "action": [],
                "spike_dims": [],
                "sink_token_spike_dims": {},
            }
        try:
            return detect_sink_tokens(value_proxy, self._tracer.token_spans)
        except ValueError:
            return {
                "all": [],
                "visual": [],
                "text": [],
                "prompt": [],
                "state": [],
                "special": [],
                "proprio": [],
                "action": [],
                "spike_dims": [],
                "sink_token_spike_dims": {},
            }

    def _apply_mode(self, attn_weights: torch.Tensor, sink_tokens: dict[str, list[int]]) -> torch.Tensor:
        if self._tracer is None:
            return attn_weights

        text_indices = flatten_indices(self._tracer.token_spans.get("text"))
        text_sink_indices = sink_tokens.get("text", [])
        non_sink_text_indices = sorted(text_indices - set(text_sink_indices))
        query_indices = list(range(attn_weights.shape[-2]))

        if self._mode == "baseline":
            return attn_weights
        if self._mode == "recalibrated":
            return recalibrate_attention_probs(
                attn_weights,
                text_sink_indices,
                non_sink_text_indices,
                query_indices=query_indices,
                p=self._p,
            )
        if self._mode == "random_text":
            out = attn_weights.clone()
            valid_text = sorted(i for i in non_sink_text_indices if i < out.shape[-1])
            if not valid_text:
                return out
            shuffled = valid_text[:]
            random.shuffle(shuffled)
            out[..., valid_text] = out[..., shuffled]
            return out
        if self._mode == "visual_uniform":
            out = attn_weights.clone()
            visual = sorted(i for i in flatten_indices(self._tracer.token_spans.get("image")) if i < out.shape[-1])
            if not visual:
                return out
            visual_mass = out[..., visual].sum(dim=-1, keepdim=True)
            out[..., visual] = visual_mass / max(1, len(visual))
            return out
        return attn_weights

    def start(self, obs: dict[str, Any], run_dir: pathlib.Path) -> None:
        self._active = True
        self._phase = None
        self._denoise_step = -1
        self._tracer = None
        self._prefix_info = _PrefixInfo()
        self._run_context = {
            "prompt": str(obs.get("prompt") or ""),
            "task_name": _sanitize_name(obs.get("task_name"), "default_task"),
            "trial_id": str(obs.get("trial_id")),
            "chunk_id": int(obs.get("chunk_id")),
            "run_dir": run_dir,
        }

    def _ensure_tracer(self, device: torch.device) -> None:
        if self._tracer is not None:
            return
        token_map, token_spans = self._build_token_map()
        query_indices = list(range(int(self._pi_model.config.action_horizon)))
        target_object = _infer_target_object(self._run_context.get("prompt"))
        self._tracer = AttentionTracer(
            token_map,
            token_spans,
            self._run_context["run_dir"],
            save_full=self._save_full_attention,
            topk=self._topk,
            layers_to_save=self._layers_to_save,
            target_object=target_object,
            distractor_objects=[],
            query_indices=query_indices,
        )
        self._tracer.start_run(
            prompt=self._run_context["prompt"],
            seed=int(self._run_context["trial_id"]) if str(self._run_context["trial_id"]).isdigit() else 0,
            mode=self._mode,
        )
        config_path = pathlib.Path(self._run_context["run_dir"]) / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config.update(
            {
                "task_name": self._run_context["task_name"],
                "trial_id": self._run_context["trial_id"],
                "chunk_id": self._run_context["chunk_id"],
                "query_layout": {
                    "action_denoise": "query_position is local action token order; query_token_index maps to full token order",
                },
                "key_layout": {
                    "action_denoise": "key_position follows full prefix image/text plus suffix action token order",
                },
                "captured_phases": ["action_denoise"],
                "mode": self._mode,
                "sink_strategy": self._sink_strategy,
                "device": str(device),
            }
        )
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    def _build_token_map(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        token_map: list[dict[str, Any]] = []
        token_spans: dict[str, Any] = {
            "image": {},
            "text": [],
            "prompt": [],
            "state": [],
            "special": [],
            "proprio": [],
            "continuous_action": [],
            "fast_action": [],
        }
        index = 0

        for camera, count in self._prefix_info.image_token_counts.items():
            cols = int(math.sqrt(count)) if int(math.sqrt(count)) ** 2 == count else int(math.ceil(math.sqrt(count)))
            rows = int(math.ceil(count / max(cols, 1)))
            image_info = self._prefix_info.image_metadata.get(camera, {})
            image_height = int(image_info.get("height", rows))
            image_width = int(image_info.get("width", cols))
            patch_width = float(image_width) / max(cols, 1)
            patch_height = float(image_height) / max(rows, 1)
            span = []
            for patch_id in range(count):
                patch_row = patch_id // cols
                patch_col = patch_id % cols
                x0 = patch_col * patch_width
                y0 = patch_row * patch_height
                x1 = min(image_width, (patch_col + 1) * patch_width)
                y1 = min(image_height, (patch_row + 1) * patch_height)
                token_map.append(
                    {
                        "index": index,
                        "type": "image",
                        "camera": camera,
                        "patch_row": patch_row,
                        "patch_col": patch_col,
                        "patch_id": patch_id,
                        "patch_grid_rows": rows,
                        "patch_grid_cols": cols,
                        "center_x": float((x0 + x1) / 2.0),
                        "center_y": float((y0 + y1) / 2.0),
                        "patch_box_xyxy": [float(x0), float(y0), float(x1), float(y1)],
                        "patch_width": float(patch_width),
                        "patch_height": float(patch_height),
                        "raw_image_path": image_info.get("path"),
                        "raw_image_height": image_height,
                        "raw_image_width": image_width,
                        "object_label": None,
                        "raw_span": None,
                    }
                )
                span.append(index)
                index += 1
            token_spans["image"][camera] = span

        for text_pos, token_id in enumerate(self._prefix_info.text_token_ids):
            is_valid_text = text_pos >= len(self._prefix_info.text_mask) or self._prefix_info.text_mask[text_pos]
            token_str = self._prefix_info.text_token_strings[text_pos] if text_pos < len(self._prefix_info.text_token_strings) else None
            token_text = self._prefix_info.text_token_texts[text_pos] if text_pos < len(self._prefix_info.text_token_texts) else None
            special_role = (
                self._prefix_info.text_token_special_roles[text_pos]
                if text_pos < len(self._prefix_info.text_token_special_roles)
                else None
            )
            recorded_type = (
                self._prefix_info.text_token_types[text_pos]
                if text_pos < len(self._prefix_info.text_token_types)
                else "prompt"
            )
            token_type = "text_padding"
            if is_valid_text:
                token_type = recorded_type
            token_map.append(
                {
                    "index": index,
                    "type": token_type,
                    "token_position": text_pos,
                    "token_id": int(token_id),
                    "token_str": token_str,
                    "token_text": token_text,
                    "special_role": special_role,
                    "char_start": None,
                    "char_end": None,
                    "valid": is_valid_text,
                }
            )
            if is_valid_text and token_type == "special":
                token_spans["special"].append(index)
            elif is_valid_text and token_type == "state":
                token_spans["state"].append(index)
                token_spans["text"].append(index)
            elif is_valid_text:
                token_spans["prompt"].append(index)
                token_spans["text"].append(index)
            index += 1

        if not getattr(self._pi_model, "pi05", False):
            token_map.append({"index": index, "type": "proprio", "name": "state_embedding", "value": None, "bin": None})
            token_spans["proprio"].append(index)
            index += 1

        for timestep in range(int(self._pi_model.config.action_horizon)):
            token_map.append({"index": index, "type": "continuous_action", "timestep": timestep})
            token_spans["continuous_action"].append(index)
            index += 1

        return token_map, token_spans

    def finish(self, action_chunk: Any) -> None:
        if self._tracer is not None:
            self._tracer.finish_run({"actions": _to_numpy(action_chunk)})
        self._active = False
        self._phase = None

    def reset(self) -> None:
        self._active = False
        self._phase = None
        self._tracer = None


class AttentionTracingPolicy:
    """Transparent websocket policy wrapper that records attention traces per action chunk."""

    def __init__(
        self,
        policy: Any,
        record_root: str | pathlib.Path,
        *,
        mode: Mode = "baseline",
        save_full_attention: bool = False,
        layers_to_save: list[int] | None = None,
        topk: int = 6,
        sink_strategy: SinkStrategy = "value_projection",
        p: float = 0.6,
        default_task_name: str | None = None,
        camera_names: tuple[str, ...] = ("front", "wrist"),
        disable_torch_compile: bool = True,
    ) -> None:
        self._policy = policy
        self._record_root = pathlib.Path(record_root).expanduser().resolve()
        self._record_root.mkdir(parents=True, exist_ok=True)
        self._mode = mode
        self._default_task_name = _sanitize_name(default_task_name, "default_task")

        if not getattr(policy, "_is_pytorch_model", False):
            raise TypeError("AttentionTracingPolicy requires a PyTorch OpenPI policy with model.safetensors.")

        pi_model = getattr(policy, "_model", None)
        if pi_model is None or not hasattr(pi_model, "paligemma_with_expert"):
            raise TypeError("Wrapped policy does not expose the expected PI0 PyTorch model structure.")

        text_tokenizer = _find_text_tokenizer(policy)
        self._collector = _OpenPIAttentionCollector(
            pi_model,
            record_root=self._record_root,
            mode=mode,
            save_full_attention=save_full_attention,
            layers_to_save=layers_to_save,
            topk=topk,
            sink_strategy=sink_strategy,
            p=p,
            camera_names=camera_names,
            text_tokenizer=text_tokenizer,
            disable_torch_compile=disable_torch_compile,
        )
        if disable_torch_compile and hasattr(policy, "_sample_actions"):
            policy._sample_actions = pi_model.sample_actions

    @property
    def metadata(self) -> dict[str, Any]:
        return self._policy.metadata

    def infer(self, obs: dict[str, Any], *, noise: np.ndarray | None = None) -> dict[str, Any]:
        self._validate_request_ids(obs)
        trial_id = str(obs.get("trial_id"))
        chunk_id = int(obs.get("chunk_id"))
        run_dir = self._record_root / f"trial_{trial_id}" / f"chunk_{chunk_id:05d}"

        self._collector.start(obs, run_dir)
        try:
            result = self._policy.infer(obs, noise=noise)
            self._collector.finish(result.get("actions"))
            return result
        except Exception:
            self._collector.reset()
            raise

    @staticmethod
    def _validate_request_ids(obs: dict[str, Any]) -> None:
        missing = [key for key in ("task_name", "trial_id", "chunk_id") if key not in obs]
        if missing:
            raise KeyError(f"Policy request is missing required attention tracing ids: {missing}")
