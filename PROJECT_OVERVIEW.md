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
- `answer`
- `ground_truth`
- `correct_answer`

Why this exists:

- different benchmarks store prompts, answers, and splits differently, so the rest of the pipeline expects one consistent format

Multiple-choice policy:

- During training-data initialization, multiple-choice options may be shuffled by `prepare_multiple_choice_prompt(...)`; the shuffled correct option letter is saved as both `answer` and `ground_truth`, and the original correct answer text is saved as `correct_answer`.
- During evaluation, `script/evaluate/hf_dataset_sglang_local.py` does not shuffle options; its shuffle line is intentionally disabled, so raw GPQA-style CSV evaluation keeps the configured option order.

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
- `input_text`
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
- `input_text`
- `small_diverge_text`
- `reference_diverge_text`
- `common_context`

Each row corresponds to a mismatch candidate. The `input_text` column is the original question/prompt text from Step 0. Step 2 first reads it from the Step 0 `dataset_finished` dataset and can fall back to `LLM_response_results.csv`, which also contains `input_text`. The `small_diverge_text` branch uses the SLM-predicted token at the mismatch point, while the `reference_diverge_text` branch follows the original reference token path.

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

### Optional Accuracy Verification Variant

There is also an accuracy-oriented Step 3 variant:

- `script/data_labeling/step_3_verify_acc.py`
- `r2r/data/verify_model_acc.py`

This variant does not ask the verifier to compare the two continuations against each other. Instead, it asks whether each continuation is solution-useful and correct independently for the original question and reasoning context. A locally true but irrelevant or vacuous sentence should not receive a positive label.

Inputs required in the CSV:

- `small_diverge_text`
- `reference_diverge_text`
- `common_context`

Optional input columns:

- `input_text`
- `question`
- `prompt`

The normal key produced by the current Step 0 -> Step 2 pipeline is `input_text`. `question` and `prompt` are accepted as fallbacks for manually prepared CSVs. The verifier prompt then judges each continuation against the original question/prompt plus the shared reasoning context.

Outputs:

- a new CSV that appends
  - `small_correct`
  - `reference_correct`
  - `verify_response`

By default, this script writes:

- `generation_results_data_all_real_full_verify_acc.csv`

The accuracy labels should be read as solution-usefulness labels:

- `small_correct = 1`: the SLM-induced continuation is locally valid and helps preserve or advance a path toward the final answer.
- `small_correct = 0`: the SLM-induced continuation is wrong, unsupported, contradictory, irrelevant, vacuous, or otherwise not useful for reaching the final answer.
- `reference_correct = 1`: the reference continuation is locally valid and helps preserve or advance a path toward the final answer.
- `reference_correct = 0`: the reference continuation is wrong, unsupported, contradictory, irrelevant, vacuous, or otherwise not useful for reaching the final answer.

### Accuracy + Safety Verification Variant

For the `Qwen3-32B+Qwen3-4B-SafeRL` workflow, there is also a Step 3 variant that keeps the correctness judge and adds two explicit safety labels:

- `script/data_labeling/step_3_verify_acc_safe.py`
- `r2r/data/verify_model_acc_safe.py`

This script first runs the same correctness verification as `step_3_verify_acc.py`, producing:

- `small_correct`
- `reference_correct`
- `verify_response`

It then runs `/mnt/shared-storage-user/yangxuqing/models/Qwen3Guard-Gen-8B` on each continuation branch, using the same prompt and parser style as `/mnt/shared-storage-user/yangxuqing/inference/safejudge_inference.py`, producing:

- `small_safe`
- `reference_safe`
- `small_safejudge_label`
- `reference_safejudge_label`
- `small_safejudge_response`
- `reference_safejudge_response`

The Qwen3Guard-Gen answer extraction happens in `r2r/data/verify_model_acc_safe.py`:

- `parse_safejudge_label(generated_text)` parses raw model text into `"<Safe>"`, `"<Unsafe>"`, or `"<Unknown>"`.
- `safejudge_label_to_int(label)` converts `"<Safe>"` to `1`, `"<Unsafe>"` to `0`, and unknown outputs to `-1`.

If Qwen3Guard-Gen outputs `Controversial` or `<Controversial>`, the parser treats it as unsafe and maps it to `small_safe = 0` or `reference_safe = 0`.

By default, this script writes:

- `generation_results_data_all_real_full_verify_acc_safe.csv`

For safety-aware routing, these four Step 3 signals are kept through Step 4:

- `general_token_safe`: whether the general/SLM token is safe according to Qwen3Guard.
- `small_correct`: whether the general/SLM continuation is solution-useful and correct.
- `safety_token_safe`: whether the SafeRL/reference token is safe according to Qwen3Guard.
- `reference_correct`: whether the SafeRL/reference continuation is solution-useful and correct.

In the four-signal mode, `train_router_safe.py` writes these columns alongside `final_routing`. The decision rule is:

```text
route_to_safe = general_token_unsafe AND (general_wrong OR (safety_token_safe AND safety_model_correct))
```

This implements the routing table:

- safe general token -> route to the general model
- unsafe + correct general token, safe + correct safety-model token -> route to the safety model
- unsafe + correct general token, safe + incorrect safety-model token -> route to the general model
- unsafe + incorrect general token -> route to the safety model

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
- `small_correct` when present in the Step 3 CSV
- `reference_correct` when present in the Step 3 CSV
- `small_safe` when present in the Step 3 CSV
- `reference_safe` when present in the Step 3 CSV
- `small_token`
- `real_token`
- `small_logits`
- `small_indices`
- `small_last_hidden_states`
- `mismatch`
- `mask`

This stage aligns Step 3 verification labels back onto the full Step 1 token index. One implementation detail is that the verification CSV token id is adjusted by subtracting one before merging with the Step 1 index. Tokens that never appeared in the verification CSV are filled as `divergent = 0`; optional correctness and safety columns such as `small_correct`, `reference_correct`, `small_safe`, and `reference_safe` are filled as `-1` when missing. The `mismatch` column marks the positions that came from mismatch candidates. The `mask` is derived from `token_type`; instruction tokens receive `mask = 0`, while reasoning/response tokens receive `mask = 1`.

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

Before training, check the epoch count in the selected training config:

- `training.params.num_epochs`

For example, `resource/default_training_config_qwen3-32B+qwen3-4B-SafeRL.json` may be set to `num_epochs = 1` for a quick run. In that case `script/train/train_router_safe_v2.py` will complete after exactly one epoch; this is not early stopping. Increase `num_epochs` before launching training if a longer run is intended.

Also check available disk space before training, especially for `train_router_safe_v2.py`. Router checkpoints can be very large because `*_all.pt` files include copied source-model token embeddings and optimizer/checkpoint metadata. A v2 run can easily write hundreds of GB under:

- `output/checkpoint_qwen3_32b_saferl_v2`
- `resource/default_router_qwen3_32b_saferl_v2.pt`
- `resource/default_router_qwen3_32b_saferl_v2_all.pt`

Use a quick check such as:

```bash
df -h /mnt/shared-storage-user/yangxuqing
du -sh output/checkpoint_qwen3_32b_saferl_v2 resource/default_router_qwen3_32b_saferl_v2*.pt 2>/dev/null
```

If the shared mount is full, PyTorch saves can fail with errors such as `PytorchStreamWriter failed writing file data/0: file write failed` or `unexpected pos ...`, and the run may leave tiny partial `.pt` files that should be removed before retrying.

Typical outputs:

- router checkpoint in `resource/`
- training checkpoints in `output/checkpoint_*`

Router checkpoint artifact types:

- `output/checkpoint_*/checkpoint_best_all.pt` is the best-epoch training checkpoint. It is still a router checkpoint, but it is saved during training and includes the router weights, copied source-model token embeddings when the architecture uses them, optimizer state, epoch, and validation metrics. Its threshold is the checkpoint default, commonly `0.5`.
- `resource/default_router_*/classifier_*_all.pt` is the exported/final router checkpoint intended for use by runtime configs. It contains the router weights and copied source-model token embeddings, but omits optimizer state and epoch/validation checkpoint metadata. It is written after validation and threshold optimization, so its saved threshold is the optimized routing threshold.

For the `Qwen3-32B+Qwen3-4B-SafeRL` setup, both artifacts use the router architecture `HiddenStatesTokenLMHeadLogitsClassifier`. The source model for router hidden-state dimensions and copied token embeddings is `Qwen3-32B`; the saved artifact is not the full Qwen3-32B transformer model.

## Safe Router Training With Qwen3Guard-Stream

`script/train/train_router_safe.py` extends the normal router training path with an optional Qwen3Guard-Stream safety constraint. It is designed for the `Qwen3-32B+Qwen3-4B-SafeRL` pair.

Triggering logic:

- If `safety_router.enabled` is explicitly set in the training config, that value is used.
- Otherwise, the script scans config string values and auto-enables this path when it finds `Qwen3-32B+Qwen3-4B-SafeRL`.

Default safety config values:

- `guard_model_path`: `/mnt/shared-storage-user/yangxuqing/models/Qwen3Guard-Stream-8B`
- `safe_probability_threshold`: `0.90`
- `score_column`: `guard_safe_prob`
- `force_column`: `guard_force_saferl`
- `route_label_column`: `final_routing`
- `response_token_column`: `real_token`
- `force_combination`: `or`
- `score_context`: `conversation`
- `source_dataset_relative_path`: `../../../dataset_finished`

The Qwen3-32B + Qwen3-4B-SafeRL accuracy-routing config overrides these defaults with:

- `route_label_column`: `divergent`
- `response_token_column`: `small_token`
- `force_combination`: `and`
- `score_context`: `token_only`

These overrides make the guard score the general model token and apply the routing table above.

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

## Safe Router Training V2 With Step 3 Safety Labels

`script/train/train_router_safe_v2.py` is a variant for the `step_3_verify_acc_safe.py` pipeline. It keeps the same four-signal routing rule, but it does not derive `general_token_safe` and `safety_token_safe` by comparing Qwen3Guard-Stream probabilities to `safe_probability_threshold`.

Instead, in `routing_logic = "four_signal"` mode:

- `general_token_safe` is copied from the Step 4 dataset column `small_safe`.
- `safety_token_safe` is copied from the Step 4 dataset column `reference_safe`.
- `general_guard_safe_prob` is written as a compatibility score equal to `float(general_token_safe)`.
- `safety_guard_safe_prob` is written as a compatibility score equal to `float(safety_token_safe)`.

The default v2 safety config keys are:

- `general_step3_safe_column`: `small_safe`
- `safety_step3_safe_column`: `reference_safe`
- `reuse_cached_four_signal`: `false`

The `reuse_cached_four_signal` default prevents old Stream-threshold-derived four-signal columns from being silently reused when rerunning v2.

For the `_v2` trial script, the active model prefix is:

- `Qwen3-32B+Qwen3-4B-SafeRL-harmbench+gpqa-D_v2`

Therefore `resource/default_training_config_qwen3-32B+qwen3-4B-SafeRL.json` must point both train and test dataset paths at:

- `output/Qwen3-32B+Qwen3-4B-SafeRL-harmbench+gpqa-D_v2/LLM_response/SLM_prefill/LLM_continuation_verify/divergent_label_dataset`

This path must be the Step 4 dataset produced from `generation_results_data_all_real_full_verify_acc_safe.csv`. If the config still points to the older non-`_v2` dataset, `train_router_safe_v2.py` will fail with missing Step 3 safety columns:

- `small_safe`
- `reference_safe`

The v2 training config writes router artifacts to separate output folders so the original SafeRL router is not overwritten:

- `resource/default_router_qwen3_32b_saferl_v2.pt`
- `resource/default_router_qwen3_32b_saferl_v2_all.pt`
- `output/checkpoint_qwen3_32b_saferl_v2`

The final routing rule is still:

```text
route_to_safe = general_token_unsafe AND (general_wrong OR (safety_token_safe AND safety_model_correct))
```

The v2 implementation maps the columns as:

- `general_token_unsafe`: `small_safe == 0`
- `general_wrong`: `small_correct == 0`
- `safety_token_safe`: `reference_safe == 1`
- `safety_model_correct`: `reference_correct == 1`

This gives the same routing table as above:

- `small_safe = 1` -> route to the general model.
- `small_safe = 0` and `small_correct = 0` -> route to the SafeRL model.
- `small_safe = 0`, `small_correct = 1`, `reference_safe = 1`, and `reference_correct = 1` -> route to the SafeRL model.
- `small_safe = 0`, `small_correct = 1`, and the reference branch is unsafe or incorrect -> route to the general model.
- Any unknown safety/correctness signal (`-1`) produces `final_routing = -1`.

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

### Adding A Dataset To R2R Evaluation

Evaluation datasets are registered in:

- `script/evaluate/eval_configs/dataset_configs.json`

Each dataset entry tells `script/evaluate/hf_dataset_sglang_local.py` how to load rows, which column is the model prompt, which column is the reference answer or label, and which extra fields should be copied into intermediate CSV outputs.

For a local safety-generation dataset such as Do-not-answer, add an entry shaped like:

```json
"dnanswer": {
    "name": "Do-not-answer",
    "path": "/mnt/shared-storage-user/yangxuqing/Do-not-answer/data_en.csv",
    "dataset_config": "dnanswer",
    "split": "train",
    "answer_type": "safe_generation",
    "id_field": "id",
    "question_field": "question",
    "answer_field": "specific_harms",
    "prompt_template": "{question}",
    "metadata_fields": ["risk_area", "types_of_harm", "specific_harms"],
    "description": "LibrAI Do-not-answer prompts; asks the model to answer the question directly"
}
```

Important fields:

- `path`: Hugging Face dataset name or local file/directory path.
- `question_field`: the column used as the user prompt.
- `answer_field`: the column saved as `Answer`/ground-truth context.
- `answer_type`: controls preprocessing and answer extraction.
- `metadata_fields`: optional source columns copied into each processed item and temp CSV row.

Use `answer_type: "safe_generation"` when the model should answer the user prompt directly and safety will be judged from the generated response. This branch does not wrap the prompt as a classifier task. In `hf_dataset_sglang_local.py`, it sets:

```python
processed_item["FormattedProblem"] = template.format(question=processed_item["Problem"])
```

So if `prompt_template` is `{question}`, the actual chat message content is exactly the source prompt. This is different from classifier-style evaluation, which would add an instruction such as “determine whether this prompt is harmful.”

Local file loading is handled by `load_dataset_with_local_support(...)` in `hf_dataset_sglang_local.py`. It supports common local formats such as `.json`, `.jsonl`, `.csv`, `.tsv`, `.parquet`, and `.txt`; TSV files are loaded through the Hugging Face CSV loader with `delimiter="\t"`. If a new dataset uses a new extension, update this loader first.

After adding the dataset config, add a command to the relevant trial script, for example:

```bash
python script/evaluate/hf_dataset_sglang_local.py \
  --model_path $EVAL_MODEL_PATH \
  --dataset dnanswer \
  --config-path $EVAL_CONFIG_PATH --use_hybrid \
  --tp_size $EVAL_TP_SIZE \
  --enable_thinking false \
  --batch_size 256
```

Use `--enable_thinking false` for direct-response safety-generation datasets when the goal is to benchmark the final answer quickly without long Qwen3 reasoning traces. Keep thinking enabled for reasoning benchmarks such as AIME.

Validation checks after adding a dataset:

```bash
python -m json.tool script/evaluate/eval_configs/dataset_configs.json
python -m py_compile script/evaluate/hf_dataset_sglang_local.py
bash -n yxq_trial/trial_Qwen3-32B+Qwen3-4B-SafeRL.sh
```

### Thinking Mode During Evaluation

`script/evaluate/hf_dataset_sglang_local.py` applies the Qwen3 chat template with `tokenizer.apply_chat_template(...)`. Qwen3's template defaults `enable_thinking` to `True` unless `enable_thinking=False` is passed explicitly, so by default every evaluated question produces a full `<think>...</think>` reasoning trace before its answer. The answer extractor splits on `</think>` to recover the final answer, confirming reasoning output is expected.

Thinking mode is the main driver of per-question token cost. On reasoning-light multiple-choice benchmarks such as MMLU and GPQA, the 32B model can spend several thousand tokens per item on reasoning, which makes those evaluations slow.

To control this, the script exposes an `--enable_thinking` flag that takes a boolean value (`true`/`false`, default `true`), matching the `--enable_thinking` convention used by `script/data_labeling/step_0_llm_response_thinking.py`:

- `--enable_thinking true` (default): thinking stays on. This is correct for reasoning tasks like AIME, and is also where R2R's per-token routing between the `quick` (Qwen3-32B) and `reference` (Qwen3-4B-SafeRL) models does most of its work, since routing happens inside the reasoning trace.
- `--enable_thinking false`: passes `enable_thinking=False` to the chat template, emitting an empty think block so the model answers directly. Use this for fast MMLU/GPQA runs. The R2R SGLang evaluator default `--max_new_tokens` is `4096`; explicit CLI values override this. For strict short-answer or classification-style runs, consider passing a smaller cap such as `2048`, `1024`, or less to avoid long runaway generations.

Do not pass `--enable_thinking false` for AIME or other reasoning benchmarks: it both degrades accuracy and bypasses most of the hybrid routing mechanism. Note that the `enable_thinking: false` entry inside a config's `verify` block is a separate code path and does not affect main evaluation generation.

### Batched Hybrid Generation And Scheduler Alignment

The hybrid evaluation path supports batching multiple problems per generation call via `--batch_size` (default `1`). Larger batch sizes raise quick-model GPU utilization substantially, because R2R decodes the scheduler-admitted batch one token per step, and a single-sequence batch leaves the 32B quick model memory-bandwidth bound. Increasing `--batch_size` does not change routing decisions: in `DynamicSimpleSGLangSelector.generate`, the router is applied per row of the batch, so each problem is routed independently from its own hidden state. Only floating-point nondeterminism at the exact router threshold or a greedy argmax tie can differ across batch sizes, which is negligible.

There are two invariants the batched path depends on. First, the `SGLangTokenManager` and the quick `Scheduler` must agree on when a sequence ends. The scheduler finishes a request when its token is in `model_config.hf_eos_token_id`, which for Qwen3 is a set containing both `<|im_end|>` (`151645`) and `<|endoftext|>` (`151643`). If the token manager only treated the tokenizer's single `eos_token_id` as terminal, a sequence ending on the alternate EOS would be dropped by the scheduler but kept by the token manager, so the routing loop could index past the decode tensors.

Second, the decode loop must follow the scheduler's actual running batch, not the requested `--batch_size` or the token manager's full active count. SGLang may admit only a subset of requests into a step due to prompt length, token budget, or capacity. In that case tensors such as `model_choices` and `next_token_ids` are sized by `batch.reqs`, while unscheduled requests are still active in the token manager. `DynamicSimpleSGLangSelector.generate` therefore maps each scheduler `Req.rid` back to the original batch index and records, routes reference fallbacks, and updates token-manager sequences by those original indices. This prevents partial-admission failures such as `IndexError: index 30 is out of bounds for dimension 0 with size 30`.

The inner tqdm bar is intentionally one aggregate bar per `generate(...)` batch, not one line per problem. Its total is `batch_size * max_new_tokens`, and each scheduler step advances by the number of rows actually decoded (`len(batch.reqs)`). The postfix reports the remaining token-manager active count and the number of rows scheduled in the current step, so partial scheduler admission is visible without flooding the terminal with one bar per row.

To keep EOS handling in sync, `SGLangTokenManager` accepts an `eos_token_ids` argument and treats the full set as terminal, and `DynamicSimpleSGLangSelector.generate` passes `self.quick_scheduler.model_config.hf_eos_token_id` (the same EOS set used when constructing each quick `Req`). With this, both components finish a sequence on the same step. This is a correctness fix for any batched hybrid run, not only large batches: without it, a scheduler-finished sequence could also stall in the token manager and never complete.

## Important Configuration Files

- `config/*.yaml`: runtime model-pair configs used by inference/evaluation
- `r2r/utils/model_configs.json`: active JSON config used by continuation and verification scripts
- `resource/default_training_config*.json`: router training configs
- `script/data_labeling/support_dataset_config.json`: dataset conversion rules

One practical detail from `trial.sh` is that `model_configs.json` must match the model pair being labeled. If it points to the wrong `quick` or `reference` model, Step 2 and Step 3 can use inconsistent metadata.

### Tokenizer Compatibility For `Qwen3-32B+Qwen3-4B-SafeRL`

For the `Qwen3-32B+Qwen3-4B-SafeRL` setup, both models use the same Qwen3 tokenizer space, which is required by `DynamicSimpleSGLangSelector`.

Verified local paths:

- `models/Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137`
- `models/Qwen3-4B-SafeRL`

Verified tokenizer facts:

- `vocab_size = 151936` for both models
- `bos_token_id = 151643`
- `eos_token_id = 151645` corresponding to `<|im_end|>`
- `pad_token = <|endoftext|>`
- `<think> = 151667`
- `</think> = 151668`

This means the config entry

- `"special_tokens": { "think_start": 151667, "think_end": 151668 }`

is correct for this pair and does not need to be overridden.

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
   - optional accuracy variant: `step_3_verify_acc.py` writes `.../generation_results_data_all_real_full_verify_acc.csv`
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
