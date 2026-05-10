# R2R Project Overview

This document explains the R2R repository from the perspective of the workflow in `yxq_trial/trial.sh`, starting with data labeling and ending with router training, serving, and evaluation.

## What R2R does

R2R (Roads to Rome) is a routing system for paired models:

- a small model (`quick`) handles most tokens cheaply
- a larger model (`reference`) is used only when the router predicts that the small model is likely to diverge on the next token

In practice, the project has three major layers:

1. Data construction: build token-level labels that say where the small model and large model meaningfully diverge.
2. Router training: train a classifier on SLM hidden states, logits, and token information.
3. Inference and evaluation: use the trained router inside an R2R serving/inference stack and benchmark the result.

## Repo Structure

The most important top-level directories are:

- `script/data_labeling/`: offline pipeline that creates router training data
- `script/train/`: entrypoints for router training
- `script/inference/`: local interactive inference and API server launchers
- `script/evaluate/`: benchmark scripts for datasets such as AIME, GPQA, MMLU, and TruthfulQA
- `r2r/`: core library code used by all scripts
- `config/`: YAML configs for model pairs and routing behavior
- `resource/`: default router checkpoints and training configs
- `output/`: generated datasets, intermediate artifacts, checkpoints, and eval outputs
- `yxq_trial/`: local experiment scripts, including the workflow this overview follows

## Workflow In `yxq_trial/trial.sh`

`yxq_trial/trial.sh` combines several use cases in one script:

- launch an R2R server
- run interactive chat tests
- generate data labels for a chosen SLM/LLM pair
- train a router
- evaluate the trained setup against baselines

The data-labeling section is the core offline preparation flow.

## Stage 1: Initialize Query Dataset

Entry point:

- `script/data_labeling/init_dataset_conversion.py`

Purpose:

- normalize source datasets into a common schema for downstream processing

Key inputs:

- `--dataset_config` names from `script/data_labeling/support_dataset_config.json`
- optionally local datasets via `--is_local`

Key output:

- a HuggingFace dataset saved under `output/<model_prefix>/`
- a JSONL snapshot next to that directory, such as `output/<model_prefix>.jsonl`

Unified fields produced here include:

- `id`
- `question`
- `original_data`
- `source`
- `type`

Why this exists:

- different benchmarks store prompts, answers, and splits differently, so the rest of the pipeline expects one consistent format

## Stage 2: Generate Reference LLM Responses

Entry points:

- `script/data_labeling/step_0_llm_response.py`
- `script/data_labeling/step_0_llm_response_thinking.py`

Purpose:

- run the larger reference model on each query and save its full response

In `trial.sh`, the active command is:

- `step_0_llm_response_thinking.py`

This is used because the target model pair may need thinking-mode handling and model-specific chat-template options.

Key output location:

- `output/<model_prefix>/LLM_response/`

Important handoff artifact:

- `dataset_finished/`, which is used by the next step

Other outputs:

- `LLM_response_results.csv`
- `dataset/`
- `run_args.json`

Fields in the finished dataset commonly include:

- `id`
- `input_text` or `question`
- `model_reasoning`
- `model_response`
- `is_finished`

Conceptually, Step 0 produces the teacher trajectory that later steps probe token-by-token.

## Stage 3: Probe The SLM Against The LLM Trace

Entry point:

- `script/data_labeling/step_1_slm_prefill_local.py`

Purpose:

- teacher-force the full Step 0 response through the small model
- compare the SLM next-token prediction against the LLM token at every position
- save token-level signals for router training

This is not free-running generation. It is next-token probing over the fixed reference sequence.

Main outputs under:

- `output/<model_prefix>/LLM_response/SLM_prefill/`

Important artifacts:

- `prediction_comparison.csv`: token-level comparison table
- `SLM_top_logits.pt`
- `SLM_top_logits_indices.pt`
- `SLM_hidden_states.pt`
- `results_test_<model>.pth`
- `results_test_<model>.jsonl`
- `results_test_<model>_sequence.jsonl`
- JSONL exports for debugging and inspection

Important columns in `prediction_comparison.csv`:

- `row_id`
- `real_token`
- `token_id`
- `data_id`
- `token_type`
- `SLM_predictions`
- `SLM_prediction_samples`

The important distinction is that `real_token` is the reference LLM token on the teacher trajectory, while `SLM_predictions` is what the small model predicts at that position. `token_type` is later used to build the training `mask`.

These files are the raw router features. They encode where the SLM matches or mismatches the LLM, plus the local hidden-state/logit evidence around each token.

## Stage 4: Continue From Mismatch Points With The LLM

Entry point:

- `script/data_labeling/step_2_llm_continuation.py`

Purpose:

- take mismatch points from `prediction_comparison.csv`
- splice in the SLM token at each mismatch
- ask the LLM to continue from that altered prefix
- compare the alternative continuation with the original reference continuation

This step turns token mismatches into semantic divergence candidates.

Important dependency:

- `r2r/utils/model_configs.json`

`trial.sh` explicitly copies a YAML config into `r2r/utils/model_configs.json` before this step. That file defines the active `quick`, `reference`, `continuation_*`, `verify`, and `router` model settings used by the continuation and verification logic.

Main output folder:

- `output/<model_prefix>/LLM_response/SLM_prefill/LLM_continuation_verify/`

Important artifact:

- `generation_results_data_all_real_full.csv`

Important columns in that CSV:

- `data_id`
- `token_id`
- `small_diverge_text`
- `reference_diverge_text`
- `common_context`

Each row corresponds to a mismatch candidate. The `small_diverge_text` branch uses the SLM-predicted token at the mismatch point, while the `reference_diverge_text` branch follows the original reference token path.

## Stage 5: Verify Whether The Divergence Is Real

Entry point:

- `script/data_labeling/step_3_verify.py`

Purpose:

- judge whether the SLM-induced continuation and the original reference continuation are genuinely divergent

Inputs required in the CSV:

- `small_diverge_text`
- `reference_diverge_text`
- `common_context`

Outputs:

- a new CSV that appends
  - `divergent`
  - `verify_response`

In `trial.sh`, this becomes:

- `generation_results_data_all_real_full_verify.csv`

This is the step that converts raw mismatches into supervision usable for routing.

The verification label should be read as:

- `divergent = 1`: the SLM-induced branch is meaningfully different, so the router should learn to escalate at this token.
- `divergent = 0`: the token mismatch did not produce a meaningful divergence.

## Stage 6: Build The Final Training Dataset

Entry point:

- `script/data_labeling/step_4_construct_label_dataset.py`

Purpose:

- align verification labels with the Step 1 token index and tensor artifacts
- create a HuggingFace dataset for training the router

This step merges:

- verification CSV from Step 3
- `prediction_comparison.csv`
- `SLM_top_logits.pt`
- `SLM_top_logits_indices.pt`
- `SLM_hidden_states.pt`

Output location in `trial.sh`:

- `output/<model_prefix>/LLM_response/SLM_prefill/LLM_continuation_verify/divergent_label_dataset/`

Important side output:

- `scalar.csv` for easy inspection of non-tensor columns

Final dataset columns:

- `token_id`
- `data_id`
- `divergent`
- `small_token`
- `real_token`
- `small_logits`
- `small_indices`
- `small_last_hidden_states`
- `mismatch`
- `mask`

This stage aligns Step 3 verification labels back onto the full Step 1 token index. One implementation detail is that the verification CSV token id is adjusted by subtracting one before merging with the Step 1 index. Tokens that never appeared in the verification CSV are filled as `divergent = 0`, and `mismatch` marks the positions that came from mismatch candidates. The `mask` is derived from `token_type`; instruction tokens receive `mask = 0`, while reasoning/response tokens receive `mask = 1`.

At this point, the project has token-level labeled examples saying which positions should trigger escalation to the large model.

## Stage 7: Train The Router

Entry point:

- `script/train/train_router.py`

Purpose:

- load the constructed HuggingFace dataset
- build a token classifier
- optimize for the routing objective
- save a trained router checkpoint and training outputs

In `trial.sh`, the config is:

- `resource/default_training_config_qwen2.5-7B.json`

That config specifies:

- model architecture: `HiddenStatesTokenLMHeadLogitsClassifier`
- inputs: `hidden_states`, `token`, `logits`
- train/test dataset paths
- optimization hyperparameters
- threshold optimization target such as minimum recall
- output checkpoint destinations

Typical outputs:

- router checkpoint in `resource/`
- training checkpoints in `output/checkpoint_*`

## Safe Router Training With Qwen3Guard-Stream

`script/train/train_router_safe.py` extends the normal router training path with an optional Qwen3Guard-Stream safety constraint. It is designed for the `Qwen3-32B+Qwen3-4B-SafeRL` pair.

Triggering logic:

- If `safety_router.enabled` is explicitly set in the training config, that value is used.
- Otherwise, the script scans config string values and auto-enables this path when it finds `Qwen3-32B+Qwen3-4B-SafeRL`.

Default safety config values:

- `guard_model_path`: `/mnt/shared-storage-user/yangxuqing/models/Qwen3Guard-Stream-8B`
- `safe_probability_threshold`: `0.80`
- `score_column`: `guard_safe_prob`
- `force_column`: `guard_force_saferl`
- `route_label_column`: `divergent`
- `response_token_column`: `real_token`
- `source_dataset_relative_path`: `../../../dataset_finished`

### What Qwen3Guard-Stream Scores

Qwen3Guard-Stream receives token ids, not hidden states. The model internally embeds those token ids, computes transformer hidden states, and applies its risk-level head to produce `risk_level_logits`.

In `train_router_safe.py`, the safety score is:

- run Qwen3Guard-Stream on an input token sequence
- read `outputs.risk_level_logits`
- apply softmax over the risk classes
- take class index `0`, which corresponds to `Safe`

So `guard_safe_prob` means:

- `P(risk_level = Safe | current token and its left context)`

The score is token-level but not context-free. A token's risk logits come from that token position's causal hidden state, so the score is conditioned on preceding tokens in the same sequence.

### Where The Guard Input Tokens Come From

The guard annotation code uses two data sources.

From Step 4 `divergent_label_dataset`:

- `data_id`
- `token_id`
- `real_token`

The script sorts rows by `(data_id, token_id)` and groups each sample's `real_token` values into a sequence called `group_tokens`. These are reference LLM tokens from the teacher trajectory, not SLM-predicted tokens.

From Step 0 `dataset_finished`:

- `question` or `input_text`
- `model_reasoning`
- `model_response`
- optionally `assistant_content`

The script reconstructs assistant content in this priority order:

- if both `model_reasoning` and `model_response` exist, use `<think>\n{model_reasoning}\n</think>\n\n{model_response}`
- otherwise use `model_response`
- otherwise use existing `assistant_content`
- otherwise use `model_reasoning`

The preferred guard input is a full chat conversation:

```text
[
  {"role": "user", "content": question},
  {"role": "assistant", "content": assistant_content}
]
```

This conversation is tokenized with the Qwen3Guard-Stream tokenizer using `apply_chat_template(..., tokenize=True, add_generation_prompt=False, enable_thinking=False)`. Conceptually this becomes:

```text
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
{assistant_content}
<|im_end|>
```

Those `conversation_ids` are the preferred token ids passed to Qwen3Guard-Stream.

The Step 4 `real_token` sequence is then used as an alignment anchor. The script looks for `group_tokens` as a subsequence inside `conversation_ids`. If it finds a match, it slices the guard safe probabilities for exactly that token span and writes them back onto the corresponding router training rows.

Fallback behavior:

- If the full conversation was scored but the subsequence cannot be found, the script takes the last `len(group_tokens)` guard scores as `conversation_suffix_fallback`.
- If `dataset_finished` is unavailable or assistant content cannot be reconstructed, the script directly scores `group_tokens` as `token_only_fallback`.

The preferred path therefore gives Qwen3Guard both the user prompt and assistant response context. The fallback path only gives the reference response token sequence.

### How Safety Forces Routing To SafeRL

After the per-token `guard_safe_prob` values are computed, the script creates:

- `guard_force_saferl = 1` when `mask == 1` and `guard_safe_prob < safe_probability_threshold`
- `guard_force_saferl = 0` otherwise

Then it updates the training label:

```text
new_label = original_divergent OR guard_force_saferl
```

By default this writes back into `divergent`, so unsafe or low-confidence-safe tokens are trained as positive routing examples. In the `Qwen3-32B+Qwen3-4B-SafeRL` setting, that means the router is trained to route those tokens to the SafeRL model even if the original divergence verifier did not mark them as divergent.

## How The Core Library Maps To The Pipeline

The `r2r/` package holds the reusable internals behind the scripts:

- `r2r/data/`: mismatch extraction, continuation control, and verification helpers
- `r2r/models/`: router model definitions and save/load helpers
- `r2r/train/`: losses, optimizer flow, logging, and evaluation during training
- `r2r/evaluate/`: benchmark utilities and prompt formatting helpers
- `r2r/utils/`: shared config, dataset conversion, sampling, and model metadata

The scripts in `script/` are thin orchestration layers over this package.

## Inference And Serving

After training, the router is used during inference.

Main entrypoints:

- `script/inference/launch_r2r_server.py`: OpenAI-compatible serving
- `script/inference/interactive_chat.py`: text interactive test
- `script/inference/interactive_chat_vl.py`: vision-language interactive test

Configs from `config/*.yaml` define the active model pair and router path. In deployment, the router decides when to stay on the small model and when to route to the large model.

## Evaluation

Main entrypoints:

- `script/evaluate/hf_dataset_sglang.py`
- `script/evaluate/hf_dataset_sglang_local.py`
- `script/evaluate/hf_dataset_sglang_server.py`

These scripts evaluate:

- R2R hybrids
- SLM baselines
- LLM baselines

The `trial.sh` examples show how the same benchmark can be run with different configs to compare:

- routed hybrid
- pure small model
- pure large/reference model

## Important Configuration Files

- `config/*.yaml`: runtime model-pair configs used by inference/evaluation
- `r2r/utils/model_configs.json`: active JSON config used by continuation and verification scripts
- `resource/default_training_config*.json`: router training configs
- `script/data_labeling/support_dataset_config.json`: dataset conversion rules

One practical detail from `trial.sh` is that `model_configs.json` must match the model pair being labeled. If it points to the wrong `quick` or `reference` model, Step 2 and Step 3 can use inconsistent metadata.

## Typical Artifact Flow

The end-to-end artifact chain in `trial.sh` is:

1. `init_dataset_conversion.py`
   - `output/<model_prefix>/`
2. `step_0_llm_response*_py`
   - `output/<model_prefix>/LLM_response/dataset_finished/`
3. `step_1_slm_prefill_local.py`
   - `output/<model_prefix>/LLM_response/SLM_prefill/`
4. `step_2_llm_continuation.py`
   - `.../LLM_continuation_verify/generation_results_data_all_real_full.csv`
5. `step_3_verify.py`
   - `.../generation_results_data_all_real_full_verify.csv`
6. `step_4_construct_label_dataset.py`
   - `.../divergent_label_dataset/`
7. `train_router.py`
   - router checkpoint + training checkpoints

## Mental Model For New Contributors

If you are new to the repo, the simplest way to think about it is:

- `script/data_labeling/` creates supervision for routing
- `script/train/` turns that supervision into a router checkpoint
- `script/inference/` uses the router online
- `script/evaluate/` measures the routed system against baselines

`yxq_trial/trial.sh` is therefore not just a demo script. It is a compact map of the full R2R lifecycle for a specific model pair.
