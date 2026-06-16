"""
Purpose
-------
Capture one LIBERO simulation observation for pi0.5 text-token probing.

Sequence
--------
1. Load a BDDL task in LIBERO.
2. Reset the environment and optionally step dummy actions for stabilization.
3. Extract the same front image, wrist image, and proprio state used by rollout
   inference.
4. Save a compressed NPZ payload that can be consumed by text_token_probe.py.

Arguments
---------
--bddl_path: BDDL task file to instantiate.
--libero_root: Path to the LIBERO checkout.
--output_npz: Path receiving the captured observation payload.
--prompt: Optional prompt override. If omitted, read the prompt from the BDDL.
--seed, --resolution, --resize, --wait_steps, --placement_retries:
  Environment and image preprocessing settings.

Usage
-----
/home/jinjaguo/anaconda3/envs/libero/bin/python \\
    /home/jinjaguo/AGTD/text_token_per_chunk_analysis/capture_libero_observation.py \\
    --bddl_path /home/jinjaguo/BH_MOE/custom_bddl/libero_goal/dif_start_end_loc/put_the_cream_cheese_on_the_plate.bddl \\
    --libero_root /home/jinjaguo/LIBERO \\
    --output_npz /home/jinjaguo/AGTD/text_token_per_chunk_analysis/outputs/put_the_cream_cheese_on_the_plate/observation.npz

Outputs
-------
The output NPZ stores:
  front_raw, wrist_raw, policy_image, policy_wrist_image, state, prompt,
  bddl_path, seed, wait_steps.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import Any

import imageio.v2 as imageio
import numpy as np
from openpi_client import image_tools


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bddl_path", type=pathlib.Path, required=True)
    parser.add_argument("--libero_root", type=pathlib.Path, default=pathlib.Path.home() / "LIBERO")
    parser.add_argument("--output_npz", type=pathlib.Path, required=True)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--wait_steps", type=int, default=10)
    parser.add_argument("--placement_retries", type=int, default=20)
    return parser.parse_args()


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    if quat.shape[-1] != 4:
        raise ValueError(f"Expected quat shape (..., 4), got {quat.shape}")
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float32)
    return ((quat[:3] * 2.0 * math.acos(quat[3])) / den).astype(np.float32)


def import_libero_modules(libero_root: pathlib.Path):
    libero_root = libero_root.expanduser().resolve()
    if not libero_root.exists():
        raise FileNotFoundError(f"LIBERO root does not exist: {libero_root}")
    if str(libero_root) not in sys.path:
        sys.path.insert(0, str(libero_root))
    from libero.libero.envs import OffScreenRenderEnv

    return OffScreenRenderEnv


def build_env(
    *,
    task_bddl_file: pathlib.Path,
    resolution: int,
    seed: int,
    offscreen_env_cls,
):
    env = offscreen_env_cls(
        bddl_file_name=str(task_bddl_file),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env


def build_env_with_placement_retries(
    *,
    task_bddl_file: pathlib.Path,
    resolution: int,
    seed: int,
    offscreen_env_cls,
    placement_retries: int,
):
    from robosuite.utils.errors import RandomizationError

    last_error = None
    for offset in range(max(1, int(placement_retries))):
        current_seed = seed + offset
        try:
            return build_env(
                task_bddl_file=task_bddl_file,
                resolution=resolution,
                seed=current_seed,
                offscreen_env_cls=offscreen_env_cls,
            )
        except RandomizationError as exc:
            last_error = exc
            print(f"Placement sampling failed with seed={current_seed}; retrying...", flush=True)
    raise RuntimeError(f"Failed to create LIBERO env after {placement_retries} retries") from last_error


def preprocess_rgb(rgb: np.ndarray, resize_size: int) -> np.ndarray:
    rgb = np.ascontiguousarray(rgb)
    rgb = image_tools.resize_with_pad(rgb, resize_size, resize_size)
    return image_tools.convert_to_uint8(rgb)


def make_policy_observation(obs: dict[str, Any], resize: int):
    agentview = obs["agentview_image"][::-1, ::-1]
    wrist = obs["robot0_eye_in_hand_image"][::-1, ::-1]
    img = preprocess_rgb(agentview, resize)
    wrist_img = preprocess_rgb(wrist, resize)
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)
    return agentview, wrist, img, wrist_img, state


def read_prompt_for_bddl(bddl_path: pathlib.Path) -> str:
    sibling_prompt = bddl_path.with_suffix(".txt")
    if sibling_prompt.exists():
        return sibling_prompt.read_text(encoding="utf-8").strip()
    prompt_txt = bddl_path.parent / "prompt.txt"
    if prompt_txt.exists():
        return prompt_txt.read_text(encoding="utf-8").strip()
    notes_txt = bddl_path.parent / "notes.txt"
    if notes_txt.exists():
        lines = notes_txt.read_text(encoding="utf-8").strip().splitlines()
        if lines:
            return lines[0].strip()
    text = bddl_path.read_text(encoding="utf-8")
    language_marker = "(:language"
    if language_marker in text:
        start = text.index(language_marker) + len(language_marker)
        end = text.find(")", start)
        if end != -1:
            return text[start:end].strip()
    return bddl_path.stem.replace("_", " ")


def save_json(path: pathlib.Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    bddl_path = args.bddl_path.expanduser().resolve()
    output_npz = args.output_npz.expanduser().resolve()
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    if not bddl_path.exists():
        raise FileNotFoundError(f"BDDL file does not exist: {bddl_path}")

    prompt = args.prompt if args.prompt is not None else read_prompt_for_bddl(bddl_path)
    offscreen_env_cls = import_libero_modules(args.libero_root)
    env = build_env_with_placement_retries(
        task_bddl_file=bddl_path,
        resolution=args.resolution,
        seed=args.seed,
        offscreen_env_cls=offscreen_env_cls,
        placement_retries=args.placement_retries,
    )
    try:
        obs = env.reset()
        done = False
        for _ in range(args.wait_steps):
            obs, _, done, _info = env.step(LIBERO_DUMMY_ACTION)
            if done:
                break
        front_raw, wrist_raw, policy_image, policy_wrist_image, state = make_policy_observation(obs, args.resize)
    finally:
        env.close()

    imageio.imwrite(output_npz.parent / "front.png", np.asarray(front_raw))
    imageio.imwrite(output_npz.parent / "wrist.png", np.asarray(wrist_raw))
    np.savez_compressed(
        output_npz,
        front_raw=np.asarray(front_raw),
        wrist_raw=np.asarray(wrist_raw),
        policy_image=np.asarray(policy_image),
        policy_wrist_image=np.asarray(policy_wrist_image),
        state=np.asarray(state, dtype=np.float32),
        prompt=np.asarray(prompt),
        bddl_path=np.asarray(str(bddl_path)),
        seed=np.asarray(args.seed),
        wait_steps=np.asarray(args.wait_steps),
    )
    save_json(
        output_npz.parent / "capture_metadata.json",
        {
            "bddl_path": str(bddl_path),
            "libero_root": str(args.libero_root.expanduser().resolve()),
            "output_npz": str(output_npz),
            "prompt": prompt,
            "seed": args.seed,
            "wait_steps": args.wait_steps,
            "resolution": args.resolution,
            "resize": args.resize,
            "state_shape": list(np.asarray(state).shape),
            "policy_image_shape": list(np.asarray(policy_image).shape),
        },
    )
    print(json.dumps({"output_npz": str(output_npz), "prompt": prompt}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
