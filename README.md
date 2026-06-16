# AGTD
Attractor-Gated Token Dynamics:  
OOD failure in π0.5 is already encoded before action decoding; therefore, correcting the pre-decoder latent attractor is more direct than modifying the flow-matching head.

# Pre-Decoder Latent Failure in π0.5

## Problem Definition

This project studies **pre-decoder latent failure** in Vision-Language-Action models, with a focus on **π0.5** under LIBERO-style OOD task settings.

Our working hypothesis is that π0.5 does not primarily fail because of insufficient action generation capability. Instead, the latent representation read by the action decoder may already have fallen into the **attraction basin of a training template** before action decoding begins. Under this view, the flow-matching or diffusion action head is not the main source of failure. It simply expands an already-biased latent representation into an action trajectory.

In other words, if the latent is correct, the action decoder is expected to produce a reasonable action. If the latent has already entered a failure-template basin, the decoder faithfully unfolds that latent into a failed trajectory.

## Motivation

Several recent VLA failure studies suggest that high in-distribution success rates can hide shortcut behavior. In particular, the paper *Restoring Linguistic Grounding in VLA Models via Train-Free Attention Recalibration* proposes that, during action-token inference, the model can be excessively attracted by certain **sink tokens**. These sink tokens may cause visual priors or training templates to dominate action generation, weakening the influence of the language instruction.

The IGAR method intervenes at inference time by recalibrating attention weights. It reduces erroneous execution under contradictory instructions by modifying the information-routing pattern between action-related tokens and prefix tokens.

This project starts from a related but more feature-centric hypothesis: **attention sinks are a symptom of a deeper feature-space attractor problem**. Instead of only modifying attention weights, we aim to test whether directly intervening on the pre-decoder latent representation can move π0.5 out of a failure-template basin.

## Experimental Setting

The experiments use the same task style as the BH-MoE setting, based on constructed OOD `bddl` files for LIBERO.

π0.5 is used as the baseline VLA policy. The environment is already configured to support π0.5 inference, and feature extraction is available during inference.

The initial goal is to build a minimal experimental pipeline that can test whether the pre-decoder template direction is causally involved in OOD failure.

Use the IGAR attention intervention entry point below for current experiments.

## Working Log
Daily development notes are maintained in `working_log/`.
Each text file is named by date, for example `working_log/2026-06-11.txt`, and
summarizes what changed that day.

## IGAR Attention Intervention

For the IGAR-style attention baseline, use the single-task BDDL list:
`custom_bddl/libero_goal/dif_start_end_loc/cream_cheese_plate_tasks_info.txt`.
This first AGTD reproduction keeps the paper's train-free post-softmax
attention-mass redistribution idea, using spike-based text sink detection and
`p=0.6`, but does not yet implement the full paper head-query selection
constraints c1/c2.

Terminal 1 starts the attention-intervention server:

```bash
cd ~/openpi
source .venv/bin/activate
python /home/jinjaguo/AGTD/start_igar_attention_server.py \
  --attention-mode recalibrated \
  --p 0.6 \
  --topk 6 \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir /home/jinjaguo/.cache/openpi/pytorch_checkpoints/pi05_libero
```

Terminal 2 runs only `put_the_cream_cheese_on_the_plate.bddl`:

```bash
conda activate libero
cd /home/jinjaguo/AGTD
python ood_libero_rollouts.py \
  --input_dir /home/jinjaguo/AGTD/custom_bddl/libero_goal \
  --tasks_info /home/jinjaguo/AGTD/custom_bddl/libero_goal/dif_start_end_loc/cream_cheese_plate_tasks_info.txt \
  --libero_root /home/jinjaguo/LIBERO \
  --host localhost \
  --port 8000 \
  --target_successes 5 \
  --max_trials 20 \
  --experiment_name dif_start_end_loc \
  --output_root /home/jinjaguo/AGTD/OOD_exp/dif_start_end_loc/put_the_cream_cheese_on_the_plate/videos
```

The IGAR attention traces are saved under:
`OOD_exp/dif_start_end_loc/put_the_cream_cheese_on_the_plate/attention_trace/`.
The compact rollout summary is saved as:
`OOD_exp/dif_start_end_loc/put_the_cream_cheese_on_the_plate/rollout_summary.json`.

If top-k files are still too large, run the server with `--topk 0` to skip
`attention_topk.jsonl` entirely.

To summarize rollout videos, run:

```bash
cd /home/jinjaguo/AGTD
python summarize_igar_results.py \
  --root /home/jinjaguo/AGTD/OOD_exp/dif_start_end_loc \
  --task_name put_the_cream_cheese_on_the_plate
```

## First Edition: Feature-Space IGAR Baseline

The first version of the method is designed as a feature-space analogue of IGAR.

During π0.5 inference, we first locate the position corresponding to the IGAR intervention point. Specifically, we identify where **action-related tokens attend to prefix tokens**. The prefix tokens may include image tokens, text instruction tokens, state or proprioception tokens, and special/template tokens.

A precise token map is required before any intervention. Token types should not be guessed. They must be derived from the current π0.5 processor, tokenizer, input constructor, attention mask, and feature hooks. This token map is the foundation of all following experiments.

## Baselines

The first experimental stage contains three baselines.

The first baseline is the original π0.5 inference without any intervention. For both ID and OOD tasks, we save success rates, action chunks, hidden states from each layer, and attention maps.

The second baseline reproduces the core idea of IGAR. After attention softmax, high-sink tokens are reweighted to test whether OOD behavior changes when attention routing is modified.

The third baseline is the feature-space version. Instead of changing attention weights, we intervene on the action-relevant hidden state at the attention output or residual stream. After the action token reads information from sink tokens, we remove part of its component along the template direction. The modified hidden state is then rescaled to preserve the original hidden-state norm.

## Minimum Template Direction

To test the core mechanism in the simplest possible way, we define a minimum failure-template direction from pre-decoder features.

At the initial timestep, before the action decoder, we extract hidden states from one or more selected layers. We then separate rollout samples into successful and failed groups and compute the mean hidden state of each group.

The minimum template direction is defined as:

```text
u_template = mean(h_failure) - mean(h_success)
```

This direction is used as a first-order estimate of the failure-template direction in latent space.

The key question is whether removing the component of an action-relevant hidden state along this failure-template direction improves OOD success rate. If OOD success rate increases, then the failure-template direction is not merely a predictable label. It has causal influence on the final behavior.

## Projection Removal

The first feature-space intervention uses projection removal:

```text
h' = h - λ · Proj_U(h)
```

Here, `h` is the original action-relevant hidden state, `h'` is the intervened hidden state, `λ` controls intervention strength, and `U` is the template direction or template subspace.

In the first version, `U` can be a single direction defined by `u_template`. Later, `U` can be expanded into a subspace composed of multiple principal components from failure-template features.

After projection removal, RMS norm preservation is applied. The token norm of `h'` is rescaled back to the original norm of `h`. This avoids turning the method into a hidden-state scale perturbation and keeps the intervention closer to the original model manifold.

## Stage-One Goal

The goal of the first stage is not to build the final adaptive token-dynamics method. The goal is to prove one causal statement:

**The pre-decoder template direction is causally effective.**

If removing the pre-decoder failure-template component improves OOD performance while preserving ID performance, then the main hypothesis becomes much stronger. It would support the claim that π0.5 OOD failure is already encoded before action decoding and that feature-space correction is a meaningful intervention point.

## Expected Evidence

The first stage should produce both behavioral and mechanistic evidence.

Behaviorally, OOD success rate should improve after feature-space projection removal. ID success rate should be preserved as much as possible.

Mechanistically, the projection of action-relevant hidden states onto the failure-template direction should decrease after intervention. This decrease should happen before the action decoder. Ideally, samples that are recovered from failure should show reduced failure-template projection and a shift toward the successful latent region.

## Interpretation

If the feature-space intervention works, the result supports the following interpretation:

π0.5 does not fail mainly because the flow-matching or diffusion head cannot generate the correct action. Instead, the action decoder receives a biased latent representation that has already entered a training-template attraction basin. The decoder then expands this latent into the corresponding failed trajectory.

This would move the explanation from an action-decoder failure to a **pre-decoder latent attractor failure**.

## Next Step

After the minimum feature-space intervention is validated, the method can be extended into an adaptive token-dynamics framework.

Instead of using a fixed projection-removal strength, the intervention strength can become dynamic. If a token enters a legitimate grounding sink, the impulse should decay to zero and allow the token to settle naturally. If a token enters an illegitimate template sink, the impulse should increase, push the token out of the sink, and prevent it from returning to the same failure-template basin.

This later stage can introduce adaptive gating, impulse-response analysis, and potentially Gaussian-process-based uncertainty estimation.
