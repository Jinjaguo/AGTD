"""
Purpose
-------
Run OOD LIBERO BDDL tasks with the OpenPI websocket policy and save rollout
videos.

This is the main rollout entry point for OOD LIBERO evaluation. For each BDDL
task, it:
1. Repeatedly rolls out the task.
2. Sends explicit `task_name`, `trial_id`, and `chunk_id` with every websocket
   request so the attention server can save chunk-aligned traces.
3. Saves both success and failure rollout videos.
4. Stops a task once it has collected the target number of successful rollouts,
   or when the trial cap is reached.

Arguments
---------
`--input_dir`
  Root directory containing custom BDDL tasks.
`--tasks_info`
  Optional task list file containing relative BDDL paths.
`--libero_root`
  Path to the LIBERO checkout.
`--target_successes`
  Minimum number of successful trials to collect per task.
`--max_trials`
  Maximum number of trials to run per task.
`--host/--port`
  OpenPI websocket server address.
`--experiment_name`
  Optional experiment folder under `OOD_exp`. Defaults to the parent folder of
  `--tasks_info`, for example `change_pos`.
`--output_root`
  Optional final video directory. If omitted, videos are saved under
  `OOD_exp/<experiment_name>/<task_name>/videos`.
Examples
--------
python /home/jinjaguo/AGTD/ood_libero_rollouts.py \
  --input_dir /home/jinjaguo/AGTD/custom_bddl/libero_goal \
  --tasks_info /home/jinjaguo/AGTD/custom_bddl/libero_goal/dif_start_end_loc/cream_cheese_plate_tasks_info.txt \
  --libero_root /home/jinjaguo/LIBERO \
  --host localhost \
  --port 8000 \
  --target_successes 5 \
  --max_trials 20 \
  --experiment_name dif_start_end_loc \
  --output_root /home/jinjaguo/AGTD/OOD_exp/dif_start_end_loc/put_the_cream_cheese_on_the_plate/videos

Outputs
-------
Videos are saved under:
`OOD_exp/<experiment_name>/<task_name>/videos/`

The compact aggregate summary is saved as:
`OOD_exp/<experiment_name>/<task_name>/rollout_summary.json`

No rollout-side chunk-wise JSONL metadata is written.
"""

import argparse
import faulthandler
import gc
import json
import math
import os
import pathlib
import sys
from typing import Optional, Tuple

import imageio
import numpy as np
from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from robosuite.utils.errors import RandomizationError


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
DEFAULT_LIBERO_ROOT = pathlib.Path.home() / "LIBERO"
REPO_ROOT = pathlib.Path(__file__).resolve().parent


def sanitize_task_name(task_name: str) -> str:
    safe = str(task_name).strip().replace("/", "_").replace("\\", "_").replace(" ", "_")
    return safe or "default_task"


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    if quat.shape[-1] != 4:
        raise ValueError(f"Expected quat shape (..., 4), got {quat.shape}")

    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float32)

    aa = (quat[:3] * 2.0 * math.acos(quat[3])) / den
    return aa.astype(np.float32)


def configure_libero_import(libero_root: pathlib.Path) -> pathlib.Path:
    libero_root = libero_root.expanduser().resolve()
    if not libero_root.exists():
        raise FileNotFoundError(f"LIBERO root does not exist: {libero_root}")
    if str(libero_root) not in sys.path:
        sys.path.insert(0, str(libero_root))
    return libero_root


def import_libero_modules(libero_root: pathlib.Path):
    configure_libero_import(libero_root)
    from libero.libero.envs import OffScreenRenderEnv

    return OffScreenRenderEnv


def preprocess_rgb(rgb: np.ndarray, resize_size: int) -> np.ndarray:
    rgb = np.ascontiguousarray(rgb)
    rgb = image_tools.resize_with_pad(rgb, resize_size, resize_size)
    rgb = image_tools.convert_to_uint8(rgb)
    return rgb


def read_prompt_for_bddl(bddl_path: pathlib.Path) -> str:
    sibling_prompt = bddl_path.with_suffix(".txt")
    if sibling_prompt.exists():
        return sibling_prompt.read_text().strip()

    prompt_txt = bddl_path.parent / "prompt.txt"
    if prompt_txt.exists():
        return prompt_txt.read_text().strip()

    notes_txt = bddl_path.parent / "notes.txt"
    if notes_txt.exists():
        lines = notes_txt.read_text().strip().splitlines()
        if lines:
            return lines[0].strip()

    text = bddl_path.read_text()
    language_marker = "(:language"
    if language_marker in text:
        start = text.index(language_marker) + len(language_marker)
        end = text.find(")", start)
        if end != -1:
            return text[start:end].strip()

    return bddl_path.stem.replace("_", " ")


def build_env(task_bddl_file: pathlib.Path, resolution: int, seed: int, offscreen_env_cls):
    env = offscreen_env_cls(
        bddl_file_name=task_bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env


def build_env_with_placement_retries(
    task_bddl_file: pathlib.Path,
    resolution: int,
    seed: int,
    offscreen_env_cls,
    placement_retries: int,
):
    last_error: Optional[RandomizationError] = None
    for retry_i in range(max(1, placement_retries)):
        retry_seed = seed + retry_i
        try:
            return build_env(
                task_bddl_file=task_bddl_file,
                resolution=resolution,
                seed=retry_seed,
                offscreen_env_cls=offscreen_env_cls,
            )
        except RandomizationError as exc:
            last_error = exc
            gc.collect()
            print(
                f"    placement sample failed for {task_bddl_file.name} "
                f"with seed={retry_seed} ({retry_i + 1}/{placement_retries}): {exc}",
                flush=True,
            )

    raise RuntimeError(
        f"Could not place all objects for {task_bddl_file} after {placement_retries} retries. "
        "Check BDDL regions for overlap, too-small ranges, object sizes, and yaw_rotation."
    ) from last_error


def make_policy_observation(obs: dict, resize: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    agentview = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
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


def run_single_trial(
    *,
    env,
    prompt_text: str,
    task_name: str,
    trial_id: int,
    ood_type: str,
    policy: WebsocketClientPolicy,
    resize: int,
    wait_steps: int,
    max_steps: int,
    replan_steps: int,
    save_wrist: bool,
):
    print(f"  [trial {trial_id}] calling env.reset()", flush=True)
    obs = env.reset()
    print(f"  [trial {trial_id}] env.reset() done", flush=True)

    done = False
    t = 0
    frames = []
    wrist_frames = []
    action_plan = []
    plan_i = 0
    chunk_id = 0
    reset_server = True
    info = None

    while t < max_steps + wait_steps and not done:
        if t < wait_steps:
            print(f"  [trial {trial_id}] warmup step {t + 1}/{wait_steps}", flush=True)
            obs, _, done, info = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue

        if not isinstance(obs, dict):
            raise RuntimeError(f"Expected dict obs, got {type(obs)}")

        agentview, wrist, img, wrist_img, state = make_policy_observation(obs, resize)
        frames.append(agentview)
        if save_wrist:
            wrist_frames.append(wrist)

        if plan_i >= len(action_plan):
            print(f"  [trial {trial_id}] requesting chunk at env step {t}", flush=True)
            base_payload = {
                "done": reset_server,
                "observation/image": img,
                "observation/wrist_image": wrist_img,
                "observation/state": state,
                "prompt": str(prompt_text),
                "task_name": task_name,
                "trial_id": int(trial_id),
                "chunk_id": int(chunk_id),
            }
            received = policy.infer(base_payload)
            action_chunk = np.asarray(received["actions"], dtype=np.float32)
            if action_chunk.ndim != 2 or action_chunk.shape[1] != 7:
                raise ValueError(f"Expected actions shape (T, 7), got {action_chunk.shape}")
            action_plan = action_chunk[:replan_steps].tolist()
            chunk_id += 1
            plan_i = 0
            reset_server = False
            print(
                f"  [trial {trial_id}] received chunk with horizon={len(action_plan)} at env step {t}",
                flush=True,
            )

        action = action_plan[plan_i]
        plan_i += 1
        obs, _, done, info = env.step(action)
        t += 1

    print(f"  [trial {trial_id}] rollout done, total_steps={t}, success={done}", flush=True)
    return done, t, frames, wrist_frames


def collect_task_trials(
    *,
    bddl_path: pathlib.Path,
    prompt_text: str,
    task_name: str,
    output_dir: pathlib.Path,
    summary_root: pathlib.Path,
    policy: WebsocketClientPolicy,
    offscreen_env_cls,
    resolution: int,
    resize: int,
    wait_steps: int,
    max_steps: int,
    replan_steps: int,
    target_successes: int,
    max_trials: int,
    seed: int,
    placement_retries: int,
    save_wrist: bool,
    ood_type: str,
) -> Tuple[int, int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running task: {bddl_path}")
    print(f"Prompt text: {prompt_text}")
    print(f"Output dir: {output_dir}")
    print(f"Summary root: {summary_root}")
    print(f"Target successes: {target_successes}")
    print(f"Max trials: {max_trials}")

    successes = 0
    failures = 0
    trials_run = 0

    for trial_id in range(max_trials):
        print(f"  [trial {trial_id}] building env for {bddl_path.name}", flush=True)
        env = build_env_with_placement_retries(
            task_bddl_file=bddl_path,
            resolution=resolution,
            seed=seed + trial_id,
            offscreen_env_cls=offscreen_env_cls,
            placement_retries=placement_retries,
        )
        print(f"  [trial {trial_id}] env created", flush=True)
        try:
            done, t, frames, wrist_frames = run_single_trial(
                env=env,
                prompt_text=prompt_text,
                task_name=task_name,
                trial_id=trial_id,
                ood_type=ood_type,
                policy=policy,
                resize=resize,
                wait_steps=wait_steps,
                max_steps=max_steps,
                replan_steps=replan_steps,
                save_wrist=save_wrist,
            )
        finally:
            print(f"  [trial {trial_id}] closing env", flush=True)
            env.close()

        trials_run += 1
        suffix = "success" if done else "failure"
        video_path = output_dir / f"trial{trial_id}_{suffix}.mp4"
        print(f"  [trial {trial_id}] writing {suffix} video to {video_path}", flush=True)
        imageio.mimwrite(str(video_path), [np.asarray(x) for x in frames], fps=30)

        if save_wrist:
            wrist_path = output_dir / f"trial{trial_id}_{suffix}_wrist.mp4"
            imageio.mimwrite(str(wrist_path), [np.asarray(x) for x in wrist_frames], fps=30)

        if done:
            successes += 1
        else:
            failures += 1

        print(
            f"  trial={trial_id} done={done} steps={t} video={video_path} "
            f"progress: success_saved={successes}/{target_successes}, failure_saved={failures}, "
            f"trials={trials_run}/{max_trials}"
        )

        if successes >= target_successes:
            print(f"  reached success target for {bddl_path.name}; moving to next task")
            break

    summary_root.mkdir(parents=True, exist_ok=True)
    summary_path = summary_root / "rollout_summary.json"
    summary = {
        "task_name": task_name,
        "bddl_path": str(bddl_path),
        "prompt_text": prompt_text,
        "success_saved": successes,
        "failure_saved": failures,
        "target_successes": target_successes,
        "trials_run": trials_run,
        "max_trials": max_trials,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved rollout summary to {summary_path}", flush=True)
    return successes, failures, trials_run


def iter_bddl_files(input_dir: pathlib.Path):
    yield from sorted(input_dir.rglob("*.bddl"))


def iter_bddl_files_from_tasks_info(tasks_info: pathlib.Path, input_dir: pathlib.Path):
    for raw_line in tasks_info.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        rel_path = pathlib.Path(line)
        if rel_path.is_absolute():
            yield rel_path.resolve()
            continue

        candidates = []
        rel_variants = [rel_path]
        parts = rel_path.parts
        if parts and parts[0] == input_dir.name:
            rel_variants.append(pathlib.Path(*parts[1:]))
        if len(parts) >= 3 and parts[0] == "libero" and parts[1] == "bddl_files":
            libero_rel = pathlib.Path(*parts[2:])
            rel_variants.append(libero_rel)
            if libero_rel.parts and libero_rel.parts[0] == input_dir.name:
                rel_variants.append(pathlib.Path(*libero_rel.parts[1:]))

        for variant in rel_variants:
            candidates.append((tasks_info.parent / variant).resolve())
            candidates.append((input_dir / variant).resolve())
            candidates.append((tasks_info.parent / variant.name).resolve())
            candidates.append((input_dir / variant.name).resolve())

        seen_candidates = []
        for candidate in candidates:
            if candidate in seen_candidates:
                continue
            seen_candidates.append(candidate)
            if candidate.exists():
                yield candidate
                break
        else:
            candidate_lines = "\n".join(f"  - {path}" for path in seen_candidates)
            raise FileNotFoundError(
                f"Could not resolve BDDL path from tasks_info line: {line}\n"
                f"tasks_info: {tasks_info}\n"
                f"input_dir: {input_dir}\n"
                f"Tried:\n{candidate_lines}"
            )


def infer_experiment_name(input_dir: pathlib.Path, tasks_info: Optional[pathlib.Path], explicit_name: Optional[str]) -> str:
    if explicit_name:
        return sanitize_task_name(explicit_name)
    if tasks_info is not None:
        return sanitize_task_name(tasks_info.parent.name)
    return sanitize_task_name(input_dir.name)


def resolve_experiment_root(
    *,
    input_dir: pathlib.Path,
    tasks_info: Optional[pathlib.Path],
    experiment_name: Optional[str],
) -> Tuple[str, pathlib.Path]:
    resolved_experiment_name = infer_experiment_name(input_dir, tasks_info, experiment_name)
    experiment_root = REPO_ROOT / "OOD_exp" / resolved_experiment_name
    return resolved_experiment_name, experiment_root


def resolve_task_output_roots(
    *,
    experiment_root: pathlib.Path,
    task_name: str,
    output_root: Optional[pathlib.Path],
) -> Tuple[pathlib.Path, pathlib.Path]:
    if output_root is not None:
        output_dir = output_root.expanduser().resolve()
        summary_root = output_dir.parent
        return output_dir, summary_root

    summary_root = experiment_root / task_name
    output_dir = summary_root / "videos"
    return output_dir, summary_root


def task_has_existing_results(*, output_dir: pathlib.Path) -> bool:
    if output_dir.exists():
        if any(output_dir.glob("trial*_success.mp4")) or any(output_dir.glob("trial*_failure.mp4")):
            return True

    return False


def main():
    faulthandler.enable(all_threads=True)
    os.environ.setdefault("PYTHONFAULTHANDLER", "1")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=pathlib.Path, required=True)
    parser.add_argument("--libero_root", type=pathlib.Path, default=DEFAULT_LIBERO_ROOT)
    parser.add_argument("--tasks_info", type=pathlib.Path, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--wait_steps", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--replan_steps", type=int, default=5)
    parser.add_argument("--target_successes", type=int, default=5)
    parser.add_argument("--max_trials", type=int, default=20)
    parser.add_argument(
        "--placement_retries",
        type=int,
        default=20,
        help="Retry environment construction with different seeds when BDDL placement sampling fails.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Experiment folder under OOD_exp. Defaults to tasks_info parent name, or input_dir name.",
    )
    parser.add_argument("--output_root", type=pathlib.Path, default=None)
    parser.add_argument(
        "--skip_existing",
        dest="skip_existing",
        action="store_true",
        default=True,
        help="Skip tasks that already have rollout outputs. Enabled by default.",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Force rerunning tasks even if outputs already exist.",
    )
    parser.add_argument("--save_wrist", action="store_true")
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    print(f"input dir path: {input_dir}")
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    offscreen_env_cls = import_libero_modules(args.libero_root)
    policy = WebsocketClientPolicy(host=args.host, port=args.port)

    tasks_info = None
    if args.tasks_info is not None:
        tasks_info = args.tasks_info.expanduser().resolve()
        if not tasks_info.exists():
            raise FileNotFoundError(f"tasks_info file does not exist: {tasks_info}")
        bddl_files = list(iter_bddl_files_from_tasks_info(tasks_info, input_dir))
    else:
        bddl_files = list(iter_bddl_files(input_dir))

    if not bddl_files:
        raise FileNotFoundError(f"No .bddl files found under: {input_dir}")

    experiment_name, experiment_root = resolve_experiment_root(
        input_dir=input_dir,
        tasks_info=tasks_info,
        experiment_name=args.experiment_name,
    )

    suite_name = experiment_name
    print(f"Experiment name: {experiment_name}")
    print(f"LIBERO source: {args.libero_root.expanduser().resolve()}")
    print(f"Input dir: {input_dir}")
    print(f"Experiment root: {experiment_root}")
    print(f"Found {len(bddl_files)} BDDL files")

    for bddl_path in bddl_files:
        if not bddl_path.exists():
            raise FileNotFoundError(f"BDDL file does not exist: {bddl_path}")

        prompt_text = read_prompt_for_bddl(bddl_path)
        task_name = sanitize_task_name(bddl_path.stem)
        output_dir, summary_root = resolve_task_output_roots(
            experiment_root=experiment_root,
            task_name=task_name,
            output_root=args.output_root,
        )
        print(f"Output path: {output_dir}")
        print(f"Summary root: {summary_root}")

        if args.skip_existing and task_has_existing_results(output_dir=output_dir):
            print(f"Skipping task {task_name}: existing rollout outputs detected")
            continue

        successes, failures, trials_run = collect_task_trials(
            bddl_path=bddl_path,
            prompt_text=prompt_text,
            task_name=task_name,
            output_dir=output_dir,
            summary_root=summary_root,
            policy=policy,
            offscreen_env_cls=offscreen_env_cls,
            resolution=args.resolution,
            resize=args.resize,
            wait_steps=args.wait_steps,
            max_steps=args.max_steps,
            replan_steps=args.replan_steps,
            target_successes=args.target_successes,
            max_trials=args.max_trials,
            seed=args.seed,
            placement_retries=args.placement_retries,
            save_wrist=args.save_wrist,
            ood_type=suite_name,
        )
        print(
            f"Finished task {task_name}: success_saved={successes}/{args.target_successes}, "
            f"failure_saved={failures}, trials_run={trials_run}/{args.max_trials}"
        )


if __name__ == "__main__":
    main()
