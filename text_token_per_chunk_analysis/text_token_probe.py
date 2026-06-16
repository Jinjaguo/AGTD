"""
Purpose
-------
Probe what text the pi0.5 PaliGemma language branch generates after a query
inserted into a captured LIBERO inference prefix.

Sequence
--------
1. Load a captured simulation observation NPZ from capture_libero_observation.py.
2. Load the PyTorch pi0.5 LIBERO policy checkpoint.
3. Run the original OpenPI policy input transform on image/state/prompt.
4. Keep the transformed inference prefix tokens unchanged.
5. Insert a query, such as " Target object:", after the valid prefix tokens.
6. Run the PaliGemma language branch without KV cache.
7. Convert the current generated-token hidden state to vocabulary logits with lm_head.
8. Autoregressively generate text and save the generated answer.

Arguments
---------
--observation_npz: Captured observation from capture_libero_observation.py.
--checkpoint_dir: PyTorch OpenPI checkpoint containing model.safetensors.
--policy_config: OpenPI policy config name, normally pi05_libero.
--query: Text query appended after the valid inference prefix tokens.
--output_dir: Directory receiving JSON results.
--prompt: Optional prompt override. If omitted, use the captured prompt.
--top_k: Number of first-step next-token probabilities to save.
--max_new_tokens: Maximum number of generated tokens after the query.
--temperature: Sampling temperature. Use 0 for greedy decoding.
--stop_on_newline: Stop generation once a generated token contains a newline.
--device: Optional torch device override. If omitted, use the policy device.

Usage
-----
/home/jinjaguo/openpi/.venv/bin/python \\
    /home/jinjaguo/AGTD/text_token_per_chunk_analysis/text_token_probe.py \\
    --observation_npz /home/jinjaguo/AGTD/text_token_per_chunk_analysis/outputs/put_the_cream_cheese_on_the_plate/observation.npz \\
    --checkpoint_dir /home/jinjaguo/.cache/openpi/pytorch_checkpoints/pi05_libero \\
    --query " Target object:" \\
    --output_dir /home/jinjaguo/AGTD/text_token_per_chunk_analysis/outputs/put_the_cream_cheese_on_the_plate

Outputs
-------
The output directory contains:
  top_next_tokens.json
  generated_answer.json
  probe_metadata.json
"""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch


BH_MOE_ROOT = pathlib.Path("/home/jinjaguo/BH_MOE")
OPENPI_SRC = pathlib.Path("/home/jinjaguo/openpi/src")
if str(BH_MOE_ROOT) not in sys.path:
    sys.path.insert(0, str(BH_MOE_ROOT))
if str(OPENPI_SRC) not in sys.path:
    sys.path.insert(0, str(OPENPI_SRC))

from openpi.models import model as openpi_model  # noqa: E402
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks  # noqa: E402
from start_server_record import Args as PolicyArgs  # noqa: E402
from start_server_record import Checkpoint, EnvMode, create_policy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation_npz", type=pathlib.Path, required=True)
    parser.add_argument(
        "--checkpoint_dir",
        type=pathlib.Path,
        default=pathlib.Path.home() / ".cache" / "openpi" / "pytorch_checkpoints" / "pi05_libero",
    )
    parser.add_argument("--policy_config", type=str, default="pi05_libero")
    parser.add_argument("--query", type=str, default=" Target object:")
    parser.add_argument("--output_dir", type=pathlib.Path, required=True)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--stop_on_newline", action="store_true", default=True)
    parser.add_argument("--no-stop_on_newline", dest="stop_on_newline", action="store_false")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def tree_to_torch(value: Any, device: str) -> Any:
    if isinstance(value, dict):
        return {key: tree_to_torch(nested, device) for key, nested in value.items()}
    array = np.asarray(value)
    return torch.from_numpy(array).to(device)[None, ...]


def find_tokenizer(policy: Any) -> Any:
    transforms = getattr(policy._input_transform, "transforms", ())
    stack = list(transforms)
    while stack:
        transform = stack.pop(0)
        tokenizer = getattr(transform, "tokenizer", None)
        if tokenizer is not None:
            return tokenizer
        stack.extend(getattr(transform, "transforms", ()) or ())
    raise RuntimeError("Could not find tokenizer in policy input transforms.")


def sentencepiece(tokenizer: Any) -> Any:
    inner = getattr(tokenizer, "_tokenizer", None)
    if inner is not None:
        return inner
    inner = getattr(tokenizer, "_paligemma_tokenizer", None)
    if inner is not None:
        return inner
    raise RuntimeError(f"Unsupported tokenizer object: {type(tokenizer)}")


def encode_text(tokenizer: Any, text: str) -> list[int]:
    return [int(token_id) for token_id in sentencepiece(tokenizer).encode(text, add_bos=False)]


def decode_piece(tokenizer: Any, token_id: int) -> dict[str, Any]:
    sp = sentencepiece(tokenizer)
    piece = sp.id_to_piece(int(token_id)) if 0 <= int(token_id) < sp.vocab_size() else None
    try:
        text = sp.decode([int(token_id)])
    except Exception:
        text = None
    return {"token_id": int(token_id), "piece": piece, "text": text}


def insert_suffix_tokens(
    tokenized_prompt: np.ndarray,
    tokenized_prompt_mask: np.ndarray,
    suffix_tokens: list[int],
) -> tuple[np.ndarray, np.ndarray, int, int]:
    tokens = np.asarray(tokenized_prompt).copy()
    mask = np.asarray(tokenized_prompt_mask).astype(bool).copy()
    valid_len = int(mask.sum())
    max_len = int(tokens.shape[-1])
    end = valid_len + len(suffix_tokens)
    if end > max_len:
        raise ValueError(
            f"Token sequence would exceed max length {max_len}: "
            f"valid_len={valid_len}, suffix_len={len(suffix_tokens)}"
        )
    if suffix_tokens:
        tokens[valid_len:end] = np.asarray(suffix_tokens, dtype=tokens.dtype)
        mask[valid_len:end] = True
    if end < max_len:
        tokens[end:] = 0
        mask[end:] = False
    return tokens, mask, valid_len, end


def prepare_inputs_with_suffix(
    transformed_inputs: dict[str, Any],
    suffix_tokens: list[int],
) -> tuple[dict[str, Any], dict[str, int]]:
    out = copy.deepcopy(transformed_inputs)
    tokens, mask, original_valid_len, new_valid_len = insert_suffix_tokens(
        out["tokenized_prompt"],
        out["tokenized_prompt_mask"],
        suffix_tokens,
    )
    out["tokenized_prompt"] = tokens
    out["tokenized_prompt_mask"] = mask
    return out, {"original_valid_len": original_valid_len, "new_valid_len": new_valid_len}


@torch.no_grad()
def language_hidden_for_inputs(
    *,
    pi_model: Any,
    transformed_inputs: dict[str, Any],
    device: str,
) -> torch.Tensor:
    torch_inputs = tree_to_torch(transformed_inputs, device)
    observation = openpi_model.Observation.from_dict(torch_inputs)
    images, img_masks, lang_tokens, lang_masks, _state = pi_model._preprocess_observation(observation, train=False)
    prefix_embs, prefix_pad_masks, prefix_att_masks = pi_model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    language_dtype = pi_model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
    prefix_embs = prefix_embs.to(dtype=language_dtype)
    att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    att_2d_masks_4d = pi_model._prepare_attention_masks_4d(att_2d_masks).to(dtype=language_dtype)
    position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    pi_model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
    (prefix_out, _suffix_out), _past_key_values = pi_model.paligemma_with_expert.forward(
        attention_mask=att_2d_masks_4d,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=False,
    )
    return prefix_out


def logits_at_language_position(
    *,
    pi_model: Any,
    transformed_inputs: dict[str, Any],
    lang_position: int,
    device: str,
) -> torch.Tensor:
    hidden = language_hidden_for_inputs(pi_model=pi_model, transformed_inputs=transformed_inputs, device=device)
    image_token_count = int(hidden.shape[1] - transformed_inputs["tokenized_prompt"].shape[-1])
    full_position = image_token_count + int(lang_position)
    if full_position < 0 or full_position >= hidden.shape[1]:
        raise IndexError(f"Requested full token position {full_position}, hidden length {hidden.shape[1]}")
    last_hidden = hidden[:, full_position, :]
    logits = pi_model.paligemma_with_expert.paligemma.lm_head(last_hidden)
    return logits.float()[0]


def top_next_tokens(
    *,
    pi_model: Any,
    transformed_inputs: dict[str, Any],
    last_query_lang_position: int,
    tokenizer: Any,
    top_k: int,
    device: str,
) -> list[dict[str, Any]]:
    logits = logits_at_language_position(
        pi_model=pi_model,
        transformed_inputs=transformed_inputs,
        lang_position=last_query_lang_position,
        device=device,
    )
    probs = torch.softmax(logits, dim=-1)
    values, indices = torch.topk(probs, k=min(int(top_k), probs.numel()))
    rows = []
    for rank, (prob, token_id) in enumerate(zip(values.tolist(), indices.tolist(), strict=True), start=1):
        item = decode_piece(tokenizer, int(token_id))
        item.update({"rank": rank, "probability": float(prob), "logit": float(logits[int(token_id)].item())})
        rows.append(item)
    return rows



def choose_next_token(logits: torch.Tensor, *, temperature: float) -> tuple[int, float]:
    if temperature <= 0:
        probs = torch.softmax(logits, dim=-1)
        token_id = int(torch.argmax(probs).item())
        return token_id, float(probs[token_id].item())
    scaled = logits / float(temperature)
    probs = torch.softmax(scaled, dim=-1)
    token_id = int(torch.multinomial(probs, num_samples=1).item())
    return token_id, float(probs[token_id].item())


def resolve_eos_token_id(pi_model: Any, tokenizer: Any) -> int | None:
    eos_id = getattr(pi_model.paligemma_with_expert.paligemma.config, "eos_token_id", None)
    if eos_id is not None:
        return int(eos_id)
    sp = sentencepiece(tokenizer)
    if hasattr(sp, "eos_id"):
        sp_eos_id = int(sp.eos_id())
        if sp_eos_id >= 0:
            return sp_eos_id
    return None


def generate_answer(
    *,
    pi_model: Any,
    transformed_inputs: dict[str, Any],
    query_tokens: list[int],
    query_meta: dict[str, int],
    tokenizer: Any,
    max_new_tokens: int,
    temperature: float,
    stop_on_newline: bool,
    device: str,
) -> dict[str, Any]:
    generated_tokens: list[int] = []
    generated_steps: list[dict[str, Any]] = []
    eos_id = resolve_eos_token_id(pi_model, tokenizer)

    for step in range(max(0, int(max_new_tokens))):
        current_suffix = query_tokens + generated_tokens
        current_inputs, current_meta = prepare_inputs_with_suffix(transformed_inputs, current_suffix)
        predictor_lang_position = current_meta["new_valid_len"] - 1
        logits = logits_at_language_position(
            pi_model=pi_model,
            transformed_inputs=current_inputs,
            lang_position=predictor_lang_position,
            device=device,
        )
        token_id, probability = choose_next_token(logits, temperature=temperature)
        token_info = decode_piece(tokenizer, token_id)
        token_info.update(
            {
                "step": int(step),
                "probability": probability,
                "logit": float(logits[token_id].item()),
                "predictor_lang_position": int(predictor_lang_position),
            }
        )
        generated_tokens.append(token_id)
        generated_steps.append(token_info)
        if eos_id is not None and token_id == eos_id:
            break
        if stop_on_newline and "\n" in str(token_info.get("text") or ""):
            break

    sp = sentencepiece(tokenizer)
    return {
        "query": sp.decode(query_tokens),
        "generated_token_ids": generated_tokens,
        "generated_tokens": generated_steps,
        "generated_text": sp.decode(generated_tokens) if generated_tokens else "",
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "stop_on_newline": bool(stop_on_newline),
        "eos_token_id": eos_id,
    }

def save_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def scalar_string(value: np.ndarray) -> str:
    return str(np.asarray(value).item())


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    observation_npz = args.observation_npz.expanduser().resolve()
    if not observation_npz.exists():
        raise FileNotFoundError(f"Captured observation does not exist: {observation_npz}")

    captured = np.load(observation_npz, allow_pickle=False)
    prompt_text = args.prompt if args.prompt is not None else scalar_string(captured["prompt"])
    raw_payload = {
        "observation/image": np.asarray(captured["policy_image"]),
        "observation/wrist_image": np.asarray(captured["policy_wrist_image"]),
        "observation/state": np.asarray(captured["state"], dtype=np.float32),
        "prompt": prompt_text,
    }
    if "front_raw" in captured:
        imageio.imwrite(output_dir / "front.png", np.asarray(captured["front_raw"]))
    if "wrist_raw" in captured:
        imageio.imwrite(output_dir / "wrist.png", np.asarray(captured["wrist_raw"]))

    policy_args = PolicyArgs(
        env=EnvMode.pi05_libero,
        default_prompt=None,
        policy=Checkpoint(config=args.policy_config, dir=str(args.checkpoint_dir.expanduser().resolve())),
    )
    policy = create_policy(policy_args)
    if not getattr(policy, "_is_pytorch_model", False):
        raise RuntimeError("This probe requires a PyTorch policy loaded from model.safetensors.")
    pi_model = getattr(policy, "_model")
    device = args.device or getattr(policy, "_pytorch_device", "cpu")
    tokenizer = find_tokenizer(policy)

    transformed_inputs = policy._input_transform(copy.deepcopy(raw_payload))
    original_tokens = [int(v) for v in np.asarray(transformed_inputs["tokenized_prompt"]).tolist()]
    original_mask = np.asarray(transformed_inputs["tokenized_prompt_mask"]).astype(bool)
    original_valid_len = int(original_mask.sum())
    original_valid_tokens = original_tokens[:original_valid_len]

    query_tokens = encode_text(tokenizer, args.query)
    query_inputs, query_meta = prepare_inputs_with_suffix(transformed_inputs, query_tokens)
    last_query_lang_position = query_meta["new_valid_len"] - 1

    top_rows = top_next_tokens(
        pi_model=pi_model,
        transformed_inputs=query_inputs,
        last_query_lang_position=last_query_lang_position,
        tokenizer=tokenizer,
        top_k=args.top_k,
        device=device,
    )
    save_json(output_dir / "top_next_tokens.json", {"query": args.query, "top_tokens": top_rows})

    generated = generate_answer(
        pi_model=pi_model,
        transformed_inputs=transformed_inputs,
        query_tokens=query_tokens,
        query_meta=query_meta,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        stop_on_newline=args.stop_on_newline,
        device=device,
    )
    save_json(output_dir / "generated_answer.json", generated)

    sp = sentencepiece(tokenizer)
    metadata = {
        "observation_npz": str(observation_npz),
        "checkpoint_dir": str(args.checkpoint_dir.expanduser().resolve()),
        "policy_config": args.policy_config,
        "prompt": prompt_text,
        "query": args.query,
        "query_tokens": [decode_piece(tokenizer, token_id) for token_id in query_tokens],
        "original_valid_len": original_valid_len,
        "query_valid_len": query_meta["new_valid_len"],
        "tokenizer_max_len": int(np.asarray(transformed_inputs["tokenized_prompt"]).shape[-1]),
        "prefix_decoded": sp.decode(original_valid_tokens),
        "query_decoded": sp.decode(query_tokens),
        "max_new_tokens": int(args.max_new_tokens),
        "temperature": float(args.temperature),
        "stop_on_newline": bool(args.stop_on_newline),
        "device": str(device),
        "state_shape": list(np.asarray(captured["state"]).shape),
        "policy_image_shape": list(np.asarray(captured["policy_image"]).shape),
        "policy_wrist_image_shape": list(np.asarray(captured["policy_wrist_image"]).shape),
    }
    for key in ("bddl_path", "seed", "wait_steps"):
        if key in captured:
            value = captured[key]
            metadata[key] = scalar_string(value) if value.shape == () else np.asarray(value).tolist()
    save_json(output_dir / "probe_metadata.json", metadata)
    print(json.dumps({"output_dir": str(output_dir), "generated_text": generated["generated_text"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
