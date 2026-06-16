"""
Purpose
-------
Start an OpenPI websocket policy server for IGAR-style attention intervention
experiments on LIBERO OOD tasks.

This server wraps a PyTorch pi0/pi0.5 OpenPI policy with `AttentionTracingPolicy`.
It implements an IGAR-style baseline: in `recalibrated` mode, the wrapper
recomputes attention probabilities, applies softmax, detects high-sink text
tokens from value-state spike features, reduces their attention mass by a
retention factor `p`, redistributes the released mass to non-sink text tokens,
and only then performs `attention @ value_states`.

This is intentionally a first reproduction baseline for AGTD. It preserves the
paper's train-free, post-softmax attention-mass redistribution mechanism, but
does not yet implement the full paper head-query selection constraints c1/c2.

Parameters
----------
--env
  Default OpenPI environment/policy preset.
--default_prompt
  Fallback prompt if the request does not provide one.
--port
  Websocket port.
--trace_root
  Root directory for attention traces. Defaults to the AGTD OOD task
  folder for `put_the_cream_cheese_on_the_plate`.
--attention_mode
  `baseline`, `recalibrated`, `random_text`, or `visual_uniform`.
--save_full_attention
  Whether to save compressed full attention tensors.
--layers_to_save
  Optional layer ids for full attention dumps.
--topk
  Number of top attended keys saved per selected action query. Use 0 to skip
  attention_topk.jsonl files.
--sink_strategy
  `value_projection` detects sink keys from value-state norms; `none` disables
  sink detection.
--p
  Sink-token attention retention factor used by `recalibrated` mode.
--record_policy_io
  Whether to also enable OpenPI `PolicyRecorder`.
policy:checkpoint --policy.config ... --policy.dir ...
  Optional explicit checkpoint override.

Usage
-----
cd ~/openpi
source .venv/bin/activate
python /home/jinjaguo/AGTD/start_igar_attention_server.py \
  --attention-mode recalibrated \
  --p 0.6 \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir /home/jinjaguo/.cache/openpi/pytorch_checkpoints/pi05_libero

Outputs
-------
Attention traces are saved under:
`OOD_exp/dif_start_end_loc/put_the_cream_cheese_on_the_plate/attention_trace/trial_<trial_id>/chunk_<chunk_id>/`
"""

from __future__ import annotations

import dataclasses
import enum
import logging
import pathlib
import socket
import sys
from typing import Literal

import tyro

from attention_trace.openpi_policy import AttentionTracingPolicy


def _ensure_openpi_importable() -> None:
    try:
        import openpi  # noqa: F401

        return
    except ModuleNotFoundError:
        candidate = pathlib.Path(__file__).resolve().parent.parent / "openpi" / "src"
        if candidate.exists():
            sys.path.insert(0, str(candidate))


_ensure_openpi_importable()

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported OpenPI policy presets."""

    pi0_base = "pi0_base"
    pi0_libero = "pi0_libero"
    pi05_base = "pi05_base"
    pi05_libero = "pi05_libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    config: str
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.pi0_base: Checkpoint(
        config="pi0_aloha",
        dir="gs://openpi-assets/checkpoints/pi0_base",
    ),
    EnvMode.pi0_libero: Checkpoint(
        config="pi0_libero",
        dir="gs://openpi-assets/checkpoints/pi0_libero",
    ),
    EnvMode.pi05_base: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.pi05_libero: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def _maybe_load_fallback_norm_stats(
    train_config: _config.TrainConfig,
    checkpoint_dir: str | pathlib.Path,
    *,
    config_name: str,
):
    checkpoint_dir = pathlib.Path(checkpoint_dir).expanduser()
    weight_path = checkpoint_dir / "model.safetensors"
    assets_dir = checkpoint_dir / "assets"

    if not weight_path.exists():
        return None

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.asset_id is None:
        return None

    local_norm_stats_path = assets_dir / data_config.asset_id / "norm_stats.json"
    if local_norm_stats_path.exists():
        return None

    fallback_assets_dir = pathlib.Path.home() / ".cache" / "openpi" / "openpi-assets" / "checkpoints" / config_name / "assets"
    fallback_norm_stats_path = fallback_assets_dir / data_config.asset_id / "norm_stats.json"
    if not fallback_norm_stats_path.exists():
        return None

    logging.info(
        "Checkpoint assets are missing at %s. Falling back to norm stats from %s",
        local_norm_stats_path,
        fallback_norm_stats_path,
    )
    return _checkpoints.load_norm_stats(fallback_assets_dir, data_config.asset_id)


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    checkpoint = DEFAULT_CHECKPOINT.get(env)
    if checkpoint is None:
        raise ValueError(f"Unsupported environment mode: {env}")

    train_config = _config.get_config(checkpoint.config)
    return _policy_config.create_trained_policy(
        train_config,
        checkpoint.dir,
        default_prompt=default_prompt,
        norm_stats=_maybe_load_fallback_norm_stats(
            train_config,
            checkpoint.dir,
            config_name=checkpoint.config,
        ),
    )


def create_policy(args: "Args") -> _policy.Policy:
    if isinstance(args.policy, Checkpoint):
        train_config = _config.get_config(args.policy.config)
        return _policy_config.create_trained_policy(
            train_config,
            args.policy.dir,
            default_prompt=args.default_prompt,
            norm_stats=_maybe_load_fallback_norm_stats(
                train_config,
                args.policy.dir,
                config_name=args.policy.config,
            ),
        )
    if isinstance(args.policy, Default):
        return create_default_policy(args.env, default_prompt=args.default_prompt)
    raise TypeError(f"Unsupported policy argument: {type(args.policy)!r}")


REPO_ROOT = pathlib.Path(__file__).resolve().parent
DEFAULT_TASK_NAME = "put_the_cream_cheese_on_the_plate"
DEFAULT_TRACE_ROOT = (
    REPO_ROOT
    / "OOD_exp"
    / "dif_start_end_loc"
    / DEFAULT_TASK_NAME
    / "attention_trace"
)


@dataclasses.dataclass
class Args:
    """Arguments for the IGAR attention-intervention websocket server."""

    env: EnvMode = EnvMode.pi05_libero
    default_prompt: str | None = None
    port: int = 8000
    trace_root: pathlib.Path = DEFAULT_TRACE_ROOT
    attention_mode: Literal["baseline", "recalibrated", "random_text", "visual_uniform"] = "recalibrated"
    save_full_attention: bool = False
    layers_to_save: list[int] | None = None
    topk: int = 6
    sink_strategy: Literal["value_projection", "none"] = "value_projection"
    p: float = 0.6
    record_policy_io: bool = False
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    if not getattr(policy, "_is_pytorch_model", False):
        raise RuntimeError(
            "IGAR attention intervention requires a PyTorch OpenPI checkpoint containing model.safetensors. "
            "Use policy:checkpoint --policy.config pi05_libero --policy.dir "
            "/home/jinjaguo/.cache/openpi/pytorch_checkpoints/pi05_libero."
        )

    trace_root = args.trace_root.expanduser().resolve()
    policy = AttentionTracingPolicy(
        policy=policy,
        record_root=trace_root,
        mode=args.attention_mode,
        save_full_attention=args.save_full_attention,
        layers_to_save=args.layers_to_save,
        topk=args.topk,
        sink_strategy=args.sink_strategy,
        p=args.p,
        default_task_name=DEFAULT_TASK_NAME,
    )

    if args.record_policy_io:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating IGAR attention server (host: %s, ip: %s)", hostname, local_ip)
    logging.info("Mode: %s; sink_strategy: %s; p: %.3f", args.attention_mode, args.sink_strategy, args.p)
    logging.info("Saving attention traces under: %s", trace_root)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
