This analysis is based on the training objective in pi 0.5. During pre-training stage, the cross entropy between original text tokens and final text token logit is part of the loss. We can assume that the cross entropy may be different between success and failure cases for a same task.

## Files

### `capture_libero_observation.py`

This script captures one real observation from the LIBERO simulation. It should
be run in the `libero` conda environment because it needs LIBERO and robosuite.

It saves:

```text
observation.npz
front.png
wrist.png
capture_metadata.json
```

The NPZ stores the same policy-ready image/state/prompt payload used by the
second probing script.

### `text_token_probe.py`

This script tests whether the pi0.5 PaliGemma language branch can recover text
tokens from the same prefix used during LIBERO action inference.

It should be run in the OpenPI venv because it needs the PyTorch OpenPI policy
and checkpoint. The script does **not** hand-write a new prompt format. Instead,
it:

1. Reads `observation.npz` captured from LIBERO.
2. Runs the original OpenPI policy input transform.
3. Keeps the transformed inference prefix tokens unchanged.
4. Inserts a query, for example ` Target object:`, after the valid prefix tokens.
5. Runs the PaliGemma language branch without KV cache.
6. Converts the current generated-token hidden state to vocabulary logits with `lm_head`.
7. Autoregressively generates the model's own answer after the query.

This is intended as a prompt-understanding probe first. It can tell whether the
model's language branch assigns high probability to the target object implied by
the prompt. Image-language grounding should be tested with additional image
swap, masking, or prompt/image mismatch controls.

## First Test: `dif_start_end_loc`

The first suggested test uses one BDDL file from:

```text
/home/jinjaguo/BH_MOE/custom_bddl/libero_goal/dif_start_end_loc
```

Step 1: capture a simulation observation.

```bash
conda activate libero

python /home/jinjaguo/AGTD/text_token_per_chunk_analysis/capture_libero_observation.py \
  --bddl_path /home/jinjaguo/AGTD/custom_bddl/libero_goal/dif_start_end_loc/put_the_bowl_on_the_rack.bddl \
  --libero_root /home/jinjaguo/LIBERO \
  --output_npz /home/jinjaguo/AGTD/text_token_per_chunk_analysis/outputs/put_the_bowl_on_the_rack/observation.npz \
  --wait_steps 10
```

Step 2: run the text-token probe on that captured observation.

```bash
/home/jinjaguo/openpi/.venv/bin/python \
  /home/jinjaguo/AGTD/text_token_per_chunk_analysis/text_token_probe.py \
  --observation_npz /home/jinjaguo/AGTD/text_token_per_chunk_analysis/outputs/put_the_bowl_on_the_rack/observation.npz \
  --checkpoint_dir /home/jinjaguo/.cache/openpi/pytorch_checkpoints/pi05_libero \
  --policy_config pi05_libero \
  --query " Target object:" \
  --output_dir /home/jinjaguo/AGTD/text_token_per_chunk_analysis/outputs/put_the_bowl_on_the_rack \
  --top_k 8 \
  --max_new_tokens 8
```

Step 2 optional: run the text-token probe with given candidates on that captured observation.

```bash
/home/jinjaguo/openpi/.venv/bin/python \
  /home/jinjaguo/AGTD/text_token_per_chunk_analysis/text_token_probe_candidate.py \
  --observation_npz /home/jinjaguo/AGTD/text_token_per_chunk_analysis/outputs/put_the_bowl_on_the_rack/observation.npz \
  --checkpoint_dir /home/jinjaguo/.cache/openpi/pytorch_checkpoints/pi05_libero \
  --policy_config pi05_libero \
  --output_dir /home/jinjaguo/AGTD/text_token_per_chunk_analysis/outputs/put_the_bowl_on_the_rack/candidate_probe
```

Arguments:

```text
--bddl_path
  Capture script only. The LIBERO BDDL file used to create the simulation
  environment.

--libero_root
  Capture script only. Path to the LIBERO checkout.

--output_npz
  Capture script only. Path receiving the captured observation NPZ.

--observation_npz
  Probe script only. Captured observation NPZ produced by
  capture_libero_observation.py.

--checkpoint_dir
  Probe script only. PyTorch pi0.5 checkpoint directory containing
  model.safetensors.

--policy_config
  Probe script only. OpenPI policy config name. Use pi05_libero for the current
  LIBERO checkpoint.

--query
  Probe script only. Text inserted after the valid inference prefix tokens. The default is
  " Target object:".

--output_dir
  Probe script only. Directory receiving probe results.

--prompt
  Optional manual prompt override. If omitted, the prompt is read from the BDDL
  language field or sibling prompt files for capture, or from observation.npz
  for probing.

--wait_steps
  Capture script only. Number of dummy environment steps before capturing the
  observation.

--top_k
  Probe script only. Number of first-step next-token vocabulary probabilities to save.

--max_new_tokens
  Probe script only. Maximum number of tokens the model generates after the query.

--temperature
  Probe script only. Sampling temperature. The default 0 uses greedy decoding.

--stop_on_newline / --no-stop_on_newline
  Probe script only. Stop generation when a generated token contains a newline. Enabled by default.
```

Outputs:

```text
<output_dir>/
  observation.npz
  front.png
  wrist.png
  capture_metadata.json
  top_next_tokens.json
  generated_answer.json
  probe_metadata.json
```

`top_next_tokens.json` records the highest-probability first next-token options
after the query. `generated_answer.json` records the model's greedy or sampled
autoregressive answer.
