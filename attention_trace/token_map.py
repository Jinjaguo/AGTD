"""
Purpose
-------
Build token-index metadata for LIBERO / pi0.5 attention interpretation.

Parameters
----------
processor_outputs: dictionary-like model inputs containing text ids, optional
                   offset mappings, and optional precomputed token spans.
tokenizer: model tokenizer used to convert input ids to token strings.
images: mapping from camera name to original or processed images.
proprio: robot proprioceptive vector.
action_tokens: optional discrete action tokens used by pretraining paths.
continuous_action_tokens: optional continuous action/noise chunk tokens.

Usage
-----
builder = TokenSpanBuilder(patch_size=16)
result = builder.build(processor_outputs, tokenizer, images, proprio, continuous_action_tokens=noise)
token_map = result["token_map"]
token_spans = result["token_spans"]

Outputs
-------
The returned token_map and token_spans are normally saved as token_map.json
inside each attention-trace run directory under outputs/attention_trace/.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


def _as_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([])
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _first_batch(value: Any) -> np.ndarray:
    array = _as_numpy(value)
    if array.ndim >= 2 and array.shape[0] == 1:
        return array[0]
    return array


def _get(mapping: Any, key: str, default: Any = None) -> Any:
    if mapping is None:
        return default
    if isinstance(mapping, Mapping):
        return mapping.get(key, default)
    return getattr(mapping, key, default)


def _shape_hw(image: Any) -> tuple[int | None, int | None]:
    arr = _as_numpy(image)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3:
        if arr.shape[0] in {1, 3, 4}:
            return int(arr.shape[1]), int(arr.shape[2])
        return int(arr.shape[0]), int(arr.shape[1])
    if arr.ndim == 2:
        return int(arr.shape[0]), int(arr.shape[1])
    return None, None


def _token_count(value: Any) -> int:
    arr = _as_numpy(value)
    if arr.size == 0:
        return 0
    if arr.ndim == 0:
        return 1
    if arr.ndim >= 2 and arr.shape[0] == 1:
        return int(arr.shape[1])
    return int(arr.shape[0])


def _token_string(tokenizer: Any, token_id: int) -> str | None:
    if tokenizer is None:
        return None
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        return str(tokenizer.convert_ids_to_tokens([int(token_id)])[0])
    if hasattr(tokenizer, "decode"):
        return str(tokenizer.decode([int(token_id)]))
    return None


def _sanitize_offset(offset: Any) -> tuple[int | None, int | None]:
    if offset is None:
        return None, None
    try:
        start, end = offset
    except (TypeError, ValueError):
        return None, None
    return int(start), int(end)


@dataclass
class TokenSpanBuilder:
    """Construct an interpretable token map for concatenated multimodal inputs."""

    patch_size: int = 16
    default_camera_names: tuple[str, ...] = ("front", "wrist")

    def build(
        self,
        processor_outputs: Any,
        tokenizer: Any,
        images: Mapping[str, Any] | Any,
        proprio: Any,
        action_tokens: Any = None,
        continuous_action_tokens: Any = None,
    ) -> dict[str, Any]:
        token_map: list[dict[str, Any]] = []
        token_spans: dict[str, Any] = {
            "image": {},
            "text": [],
            "proprio": [],
            "fast_action": [],
            "continuous_action": [],
        }
        index = 0

        precomputed_spans = _get(processor_outputs, "token_spans")
        if precomputed_spans:
            token_spans.update(precomputed_spans)

        image_mapping = images if isinstance(images, Mapping) else {"image": images}
        image_token_counts = _get(processor_outputs, "image_token_counts", {}) or {}
        for camera_name, image in image_mapping.items():
            height, width = _shape_hw(image)
            default_rows = int(np.ceil(height / self.patch_size)) if height else 0
            default_cols = int(np.ceil(width / self.patch_size)) if width else 0
            count = int(image_token_counts.get(camera_name, default_rows * default_cols))
            if count <= 0:
                continue
            cols = default_cols or int(np.ceil(np.sqrt(count)))
            rows = int(np.ceil(count / max(cols, 1)))
            span: list[int] = []
            for patch_id in range(count):
                patch_row = patch_id // cols
                patch_col = patch_id % cols
                center_x = (patch_col + 0.5) * self.patch_size
                center_y = (patch_row + 0.5) * self.patch_size
                entry = {
                    "index": index,
                    "type": "image",
                    "camera": str(camera_name),
                    "patch_row": int(patch_row),
                    "patch_col": int(patch_col),
                    "patch_id": int(patch_id),
                    "patch_grid_rows": int(rows),
                    "patch_grid_cols": int(cols),
                    "center_x": float(center_x),
                    "center_y": float(center_y),
                    "processed_height": height,
                    "processed_width": width,
                    "object_label": None,
                    "raw_span": None,
                }
                token_map.append(entry)
                span.append(index)
                index += 1
            token_spans["image"][str(camera_name)] = span

        input_ids = _first_batch(_get(processor_outputs, "input_ids", []))
        offsets = _first_batch(_get(processor_outputs, "offset_mapping", []))
        text_span: list[int] = []
        for pos, token_id in enumerate(input_ids.tolist() if input_ids.size else []):
            if int(token_id) < 0:
                continue
            char_start, char_end = _sanitize_offset(offsets[pos] if len(offsets) > pos else None)
            token_map.append(
                {
                    "index": index,
                    "type": "text",
                    "token_position": int(pos),
                    "token_id": int(token_id),
                    "token_str": _token_string(tokenizer, int(token_id)),
                    "char_start": char_start,
                    "char_end": char_end,
                }
            )
            text_span.append(index)
            index += 1
        token_spans["text"] = text_span

        proprio_array = _as_numpy(proprio).reshape(-1)
        for proprio_idx, value in enumerate(proprio_array.tolist()):
            token_map.append(
                {
                    "index": index,
                    "type": "proprio",
                    "name": f"proprio_{proprio_idx}",
                    "value": float(value),
                    "bin": None,
                }
            )
            token_spans["proprio"].append(index)
            index += 1

        for timestep in range(_token_count(action_tokens)):
            token_map.append({"index": index, "type": "fast_action", "timestep": int(timestep)})
            token_spans["fast_action"].append(index)
            index += 1

        for timestep in range(_token_count(continuous_action_tokens)):
            token_map.append({"index": index, "type": "continuous_action", "timestep": int(timestep)})
            token_spans["continuous_action"].append(index)
            index += 1

        return {"token_map": token_map, "token_spans": token_spans}


def find_subword_indices(target_text: str, token_map: list[dict[str, Any]]) -> list[int]:
    """Find text-token indices that overlap a target word or its subword pieces."""

    target = target_text.lower().replace(" ", "")
    matches: list[int] = []
    for entry in token_map:
        if entry.get("type") != "text":
            continue
        token = str(entry.get("token_str") or "").lower()
        compact = token.replace(" ", "").replace("_", "").replace("#", "").replace("▁", "")
        if compact and (compact in target or target in compact):
            matches.append(int(entry["index"]))
    return matches
