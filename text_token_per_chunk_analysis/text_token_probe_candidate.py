"""
Purpose
-------
Score candidate text answers after a fixed query appended to a captured LIBERO
pi0.5 inference prefix. This is intended for prompt grounding tests such as:
given the original task prompt, image, and robot state, append " Target object:"
and compare whether the task-relevant object has lower negative log likelihood
than distractor objects.

Sequence
--------
1. Load a captured simulation observation NPZ from capture_libero_observation.py.
2. Load the PyTorch pi0.5 LIBERO policy checkpoint.
3. Run the original OpenPI policy input transform on image/state/prompt.
4. Append a query, such as " Target object:", after the valid prefix tokens.
5. For each candidate, teacher-force candidate tokens after the query.
6. Compute token-level log probability and candidate-level NLL.
7. Save candidates ranked by lower NLL.

Arguments
---------
--observation_npz: Captured observation from capture_libero_observation.py.
--checkpoint_dir: PyTorch OpenPI checkpoint containing model.safetensors.
--policy_config: OpenPI policy config name, normally pi05_libero.
QUERY: Code-level text query appended after the valid inference prefix tokens.
CANDIDATES: Code-level candidate answer strings. Quote multi-word candidates in
  the Python tuple.
--candidate_prefix: Prefix inserted before a candidate if it does not already
  start with whitespace. The default single space matches natural text after a
  colon query.
--output_dir: Directory receiving JSON results.
--prompt: Optional prompt override. If omitted, use the captured prompt.
--device: Optional torch device override. If omitted, use the policy device.

Usage
-----
/home/jinjaguo/openpi/.venv/bin/python \\
    /home/jinjaguo/AGTD/text_token_per_chunk_analysis/text_token_probe_candidate.py \\
    --observation_npz /home/jinjaguo/AGTD/text_token_per_chunk_analysis/outputs/put_the_bowl_on_the_rack/observation.npz \\
    --checkpoint_dir /home/jinjaguo/.cache/openpi/pytorch_checkpoints/pi05_libero \\
    --policy_config pi05_libero \\
    --output_dir /home/jinjaguo/AGTD/text_token_per_chunk_analysis/outputs/put_the_bowl_on_the_rack/candidate_probe

Outputs
-------
The output directory contains:
  candidate_nll.json           Candidate-level and token-level NLL scores.
  candidate_probe_metadata.json  Run settings and decoded prefix/query text.
  front.png, wrist.png         Copied raw images if present in the observation.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import pathlib
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch

from text_token_probe import (  # noqa: E402
    Checkpoint,
    EnvMode,
    PolicyArgs,
    create_policy,
    decode_piece,
    encode_text,
    find_tokenizer,
    logits_at_language_position,
    prepare_inputs_with_suffix,
    save_json,
    scalar_string,
    sentencepiece,
)

QUERY = " Target object:"
CANDIDATES = ("bowl", "rack","table","cabinet","stove","bottle","plate","microwave")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation_npz", type=pathlib.Path, required=True)
    parser.add_argument(
        "--checkpoint_dir",
        type=pathlib.Path,
        default=pathlib.Path.home() / ".cache" / "openpi" / "pytorch_checkpoints" / "pi05_libero",
    )
    parser.add_argument("--policy_config", type=str, default="pi05_libero")
    parser.add_argument("--candidate_prefix", type=str, default=" ")
    parser.add_argument("--output_dir", type=pathlib.Path, required=True)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_candidates() -> list[str]:
    deduped = list(dict.fromkeys(str(candidate) for candidate in CANDIDATES))
    if not deduped:
        raise ValueError("CANDIDATES must contain at least one candidate string.")
    return deduped


def candidate_text_for_model(candidate: str, candidate_prefix: str) -> str:
    if not candidate:
        return candidate
    if candidate[:1].isspace():
        return candidate
    return f"{candidate_prefix}{candidate}" if candidate_prefix else candidate


def score_candidate(
    *,
    pi_model: Any,
    transformed_inputs: dict[str, Any],
    query_tokens: list[int],
    candidate: str,
    candidate_prefix: str,
    tokenizer: Any,
    device: str,
) -> dict[str, Any]:
    model_text = candidate_text_for_model(candidate, candidate_prefix)
    candidate_tokens = encode_text(tokenizer, model_text)
    if not candidate_tokens:
        raise ValueError(f"Candidate produced no tokens: {candidate!r}")

    token_rows = []
    total_nll = 0.0
    total_logprob = 0.0
    for step, target_token_id in enumerate(candidate_tokens):
        context_suffix = query_tokens + candidate_tokens[:step]
        current_inputs, current_meta = prepare_inputs_with_suffix(transformed_inputs, context_suffix)
        predictor_lang_position = current_meta["new_valid_len"] - 1
        logits = logits_at_language_position(
            pi_model=pi_model,
            transformed_inputs=current_inputs,
            lang_position=predictor_lang_position,
            device=device,
        )
        log_probs = torch.log_softmax(logits, dim=-1)
        logprob = float(log_probs[int(target_token_id)].item())
        probability = float(math.exp(logprob))
        nll = -logprob
        total_logprob += logprob
        total_nll += nll
        token_info = decode_piece(tokenizer, int(target_token_id))
        token_info.update(
            {
                "step": int(step),
                "logprob": logprob,
                "probability": probability,
                "nll": nll,
                "predictor_lang_position": int(predictor_lang_position),
            }
        )
        token_rows.append(token_info)

    return {
        "candidate": candidate,
        "candidate_text_for_model": model_text,
        "candidate_token_ids": [int(token_id) for token_id in candidate_tokens],
        "candidate_decoded": sentencepiece(tokenizer).decode(candidate_tokens),
        "num_tokens": len(candidate_tokens),
        "total_logprob": total_logprob,
        "nll": total_nll,
        "mean_nll": total_nll / len(candidate_tokens),
        "token_scores": token_rows,
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    observation_npz = args.observation_npz.expanduser().resolve()
    if not observation_npz.exists():
        raise FileNotFoundError(f"Captured observation does not exist: {observation_npz}")

    candidates = load_candidates()
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
    query_tokens = encode_text(tokenizer, QUERY)

    scored = [
        score_candidate(
            pi_model=pi_model,
            transformed_inputs=transformed_inputs,
            query_tokens=query_tokens,
            candidate=candidate,
            candidate_prefix=args.candidate_prefix,
            tokenizer=tokenizer,
            device=device,
        )
        for candidate in candidates
    ]
    ranked_by_nll = sorted(scored, key=lambda row: row["nll"])
    ranked_by_mean_nll = sorted(scored, key=lambda row: row["mean_nll"])
    save_json(
        output_dir / "candidate_nll.json",
        {
            "query": QUERY,
            "candidate_prefix": args.candidate_prefix,
            "best_by_nll": ranked_by_nll[0]["candidate"],
            "best_by_mean_nll": ranked_by_mean_nll[0]["candidate"],
            "ranked_by_nll": ranked_by_nll,
            "ranked_by_mean_nll": ranked_by_mean_nll,
        },
    )

    sp = sentencepiece(tokenizer)
    metadata = {
        "observation_npz": str(observation_npz),
        "checkpoint_dir": str(args.checkpoint_dir.expanduser().resolve()),
        "policy_config": args.policy_config,
        "prompt": prompt_text,
        "query": QUERY,
        "query_tokens": [decode_piece(tokenizer, token_id) for token_id in query_tokens],
        "candidates": candidates,
        "candidate_prefix": args.candidate_prefix,
        "original_valid_len": original_valid_len,
        "tokenizer_max_len": int(np.asarray(transformed_inputs["tokenized_prompt"]).shape[-1]),
        "prefix_decoded": sp.decode(original_valid_tokens),
        "query_decoded": sp.decode(query_tokens),
        "device": str(device),
        "state_shape": list(np.asarray(captured["state"]).shape),
        "policy_image_shape": list(np.asarray(captured["policy_image"]).shape),
        "policy_wrist_image_shape": list(np.asarray(captured["policy_wrist_image"]).shape),
    }
    for key in ("bddl_path", "seed", "wait_steps"):
        if key in captured:
            value = captured[key]
            metadata[key] = scalar_string(value) if value.shape == () else np.asarray(value).tolist()
    save_json(output_dir / "candidate_probe_metadata.json", metadata)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "best_by_nll": ranked_by_nll[0]["candidate"],
                "best_by_mean_nll": ranked_by_mean_nll[0]["candidate"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
