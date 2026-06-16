"""
Purpose
-------
Summarize IGAR-style attention intervention rollout results for the selected
LIBERO OOD task.

The script summarizes rollout success rates for one task by reading the compact
`rollout_summary.json` produced by `ood_libero_rollouts.py`, with rollout video
filenames as a fallback.

Parameters
----------
--root
  OOD experiment root that contains the task result folder.
--task_name
  Task folder name to summarize.
--output
  Optional output JSON path. Defaults to
  `<root>/<task_name>/igar_comparison_summary.json`.

Usage
-----
python summarize_igar_results.py \
  --root /home/jinjaguo/AGTD/OOD_exp/dif_start_end_loc \
  --task_name put_the_cream_cheese_on_the_plate

Outputs
-------
The summary JSON is saved to:
`OOD_exp/dif_start_end_loc/put_the_cream_cheese_on_the_plate/igar_comparison_summary.json`
unless `--output` is provided.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any


DEFAULT_TASK_NAME = "put_the_cream_cheese_on_the_plate"
DEFAULT_ROOT = pathlib.Path(__file__).resolve().parent / "OOD_exp" / "dif_start_end_loc"


def summarize_mode(mode_dir: pathlib.Path, task_name: str) -> dict[str, Any]:
    video_root = mode_dir / "videos"
    if not any(video_root.glob("trial*_*.mp4")):
        nested_video_root = video_root / task_name
        if nested_video_root.exists():
            video_root = nested_video_root
    summary_path = mode_dir / "rollout_summary.json"
    success_videos = sorted(video_root.glob("trial*_success.mp4"))
    failure_videos = sorted(video_root.glob("trial*_failure.mp4"))
    successes = len(success_videos)
    failures = len(failure_videos)
    trials = successes + failures

    rollout_summary: dict[str, Any] = {}
    if summary_path.exists():
        rollout_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        successes = int(rollout_summary.get("success_saved", successes))
        failures = int(rollout_summary.get("failure_saved", failures))
        trials = int(rollout_summary.get("trials_run", trials))

    success_rate = successes / trials if trials else None
    return {
        "mode_dir": str(mode_dir),
        "video_root": str(video_root),
        "summary_path": str(summary_path) if summary_path.exists() else None,
        "num_trials": trials,
        "success_saved": successes,
        "failure_saved": failures,
        "success_rate": success_rate,
        "rollout_summary": rollout_summary,
        "success_videos": [str(path) for path in success_videos],
        "failure_videos": [str(path) for path in failure_videos],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=DEFAULT_ROOT)
    parser.add_argument("--task_name", type=str, default=DEFAULT_TASK_NAME)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    task_root = root / args.task_name
    summary = {
        "root": str(root),
        "task_name": args.task_name,
        "result": summarize_mode(task_root, args.task_name),
    }

    output = args.output.expanduser().resolve() if args.output else task_root / "igar_comparison_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved summary to: {output}")


if __name__ == "__main__":
    main()
