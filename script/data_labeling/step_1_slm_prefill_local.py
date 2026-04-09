"""
Step1 use SLM to prefill the LLM responses, finding all non-identical SLM next-token predictions.

IMPORTANT CLARIFICATION:
- This step is teacher-forced prefill (next-token probing), not autoregressive generation.
- The script feeds the full reference sequence (from Step0) into the SLM once and reads per-position next-token logits.
- SLM_predictions are sampled independently from each position's logits; they are not fed back as future inputs.
- Therefore, token-level predictions are suitable for mismatch analysis, top-logit analysis, and hidden-state extraction.
- Sequence text reconstructed from token-level predictions is for debugging only and should not be interpreted as true SLM free-run output.

Inputs:
- A huggingface dataset with the model responses.
    - The dataset from Step0. It contains columns: "id", "input_text", "model_reasoning", "model_response", and "is_finished". Each row corresponds to a query.

Outputs:
- prediction_comparison.csv: A csv file comparing LLM and SLM next-token predictions 
    - contains columns: data_id, token_id, real_token (predited tokens from LLM),SLM_predictions,SLM_prediction_samples
    - each row corresponds to a token in the original LLM response
- SLM_hidden_states/top_logits/top_logits_indices.pt: The last-layer hidden states, top logits, and top logits indices of the SLM for each token prediction. All tensors have the same first dimension of #total_tokens.
"""

import json
import os
import argparse
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig
import torch
from datasets import load_from_disk
import pandas as pd
import glob
import numpy as np

from r2r.utils.config import TOKEN_TYPE, MODEL_DICT
from r2r.utils.sampling import sample_token


def str2bool(value):
    """Parse common CLI boolean spellings."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "t"}:
        return True
    if text in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_model_size(model_identifier: str, fallback_idx: int) -> float:
    """Parse model size from model identifier (supports model names and local paths)."""
    normalized = model_identifier.rstrip("/")
    match = re.search(r"(\d+(?:\.\d+)?)B", normalized, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))

    fallback_size = float(fallback_idx + 1)
    print(
        f"Warning: Cannot parse model size from '{model_identifier}'. "
        f"Using fallback order key: {fallback_size}."
    )
    return fallback_size


def get_model_tag(model_identifier: str) -> str:
    """Get stable result file tag from model name or local path."""
    normalized = model_identifier.rstrip("/")
    base = os.path.basename(normalized)
    return base if base else normalized.replace("/", "_")


def _get_model_device(model):
    """Get a usable device for sending input tensors."""
    return next(model.parameters()).device


def _get_output_embedding_layer(model):
    """Best-effort getter for output embeddings across text/VL model wrappers."""
    if hasattr(model, "get_output_embeddings"):
        layer = model.get_output_embeddings()
        if layer is not None:
            return layer

    # Common direct attributes used by LM wrappers
    for attr in ("lm_head", "embed_out", "output_projection"):
        layer = getattr(model, attr, None)
        if layer is not None:
            return layer

    language_model = getattr(model, "language_model", None)
    if language_model is not None:
        if hasattr(language_model, "get_output_embeddings"):
            layer = language_model.get_output_embeddings()
            if layer is not None:
                return layer
        for attr in ("lm_head", "embed_out", "output_projection"):
            layer = getattr(language_model, attr, None)
            if layer is not None:
                return layer

    base_model = getattr(model, "model", None)
    if base_model is not None:
        for attr in ("lm_head", "embed_out", "output_projection"):
            layer = getattr(base_model, attr, None)
            if layer is not None:
                return layer

    return None


def _get_input_embedding_layer(model):
    """Best-effort getter for input embeddings for tied-weight logits fallback."""
    if hasattr(model, "get_input_embeddings"):
        layer = model.get_input_embeddings()
        if layer is not None:
            return layer

    language_model = getattr(model, "language_model", None)
    if language_model is not None and hasattr(language_model, "get_input_embeddings"):
        layer = language_model.get_input_embeddings()
        if layer is not None:
            return layer

    base_model = getattr(model, "model", None)
    if base_model is not None and hasattr(base_model, "get_input_embeddings"):
        layer = base_model.get_input_embeddings()
        if layer is not None:
            return layer

    return None


def _extract_logits(model, outputs):
    """Extract logits from model outputs, reconstructing from hidden states when needed."""
    logits = getattr(outputs, "logits", None)
    if logits is not None:
        return logits

    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states:
        last_hidden = hidden_states[-1]
    else:
        last_hidden = getattr(outputs, "last_hidden_state", None)
        if last_hidden is None:
            raise ValueError("Model outputs do not contain logits, hidden_states, or last_hidden_state.")

    output_layer = _get_output_embedding_layer(model)
    if output_layer is not None:
        if hasattr(output_layer, "weight"):
            logits = torch.matmul(last_hidden, output_layer.weight.transpose(0, 1))
            if hasattr(output_layer, "bias") and output_layer.bias is not None:
                logits = logits + output_layer.bias
            return logits

        # Fallback: try calling output layer directly if it's a projection module
        try:
            return output_layer(last_hidden)
        except Exception:
            pass

    # Final fallback: use tied input embeddings as LM head approximation.
    input_layer = _get_input_embedding_layer(model)
    if input_layer is not None and hasattr(input_layer, "weight"):
        return torch.matmul(last_hidden, input_layer.weight.transpose(0, 1))

    raise ValueError(
        f"Model {model.__class__.__name__} does not provide usable output/input embeddings for logits reconstruction."
    )

def load_model(model_name):
    """Load a model with basic error handling"""
    try:
        model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        loader_backend = "AutoModelForCausalLM"
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config=model_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            ).eval()
        except Exception as causal_exc:
            print(
                f"AutoModelForCausalLM load failed for {model_name}: {causal_exc}. "
                "Trying AutoModel fallback for VL or custom configs."
            )
            loader_backend = "AutoModel (fallback)"
            model = AutoModel.from_pretrained(
                model_name,
                config=model_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            ).eval()
        print(f"Model {model_name} loaded successfully!")
        print(f"Model loader backend: {loader_backend}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def save_slm_raw_jsonl(output_path, model_tag, predictions, token_ids, data_ids, token_types, real_tokens, tokenizer):
    """Save original SLM token-level outputs to JSONL."""
    output_jsonl = os.path.join(output_path, f"results_test_{model_tag}.jsonl")

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for row_id, (pred, token_id, data_id, token_type, real_token) in enumerate(
            zip(predictions, token_ids, data_ids, token_types, real_tokens)
        ):
            record = {
                "row_id": int(row_id),
                "data_id": int(data_id),
                "token_id": int(token_id),
                "token_type": int(token_type),
                "real_token": int(real_token),
                "SLM_prediction": int(pred),
                "real_token_text": tokenizer.decode([int(real_token)], skip_special_tokens=False),
                "SLM_prediction_text": tokenizer.decode([int(pred)], skip_special_tokens=False),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Raw SLM outputs saved to {output_jsonl}")


def save_slm_sequence_jsonl(output_path, model_tag, predictions, token_ids, data_ids, token_types, real_tokens, tokenizer):
    """Save sequence-level SLM outputs (one record per data_id) to JSONL."""
    output_jsonl = os.path.join(output_path, f"results_test_{model_tag}_sequence.jsonl")

    grouped = {}
    for pred, token_id, data_id, token_type, real_token in zip(
        predictions, token_ids, data_ids, token_types, real_tokens
    ):
        did = int(data_id)
        grouped.setdefault(did, []).append(
            (int(token_id), int(pred), int(token_type), int(real_token))
        )

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for data_id in sorted(grouped.keys()):
            rows = sorted(grouped[data_id], key=lambda x: x[0])
            token_ids_sorted = [r[0] for r in rows]
            pred_tokens = [r[1] for r in rows]
            token_types_sorted = [r[2] for r in rows]
            real_tokens_sorted = [r[3] for r in rows]

            # Logits at position i predict token i+1, so align predictions with next token types.
            aligned_pred_tokens = pred_tokens[:-1]
            aligned_token_types = token_types_sorted[1:]

            pred_reasoning_tokens = [
                t for t, tp in zip(aligned_pred_tokens, aligned_token_types)
                if tp == TOKEN_TYPE.REASONING
            ]
            pred_response_tokens = [
                t for t, tp in zip(aligned_pred_tokens, aligned_token_types)
                if tp == TOKEN_TYPE.RESPONSE
            ]
            pred_assistant_tokens = [
                t for t, tp in zip(aligned_pred_tokens, aligned_token_types)
                if tp in (TOKEN_TYPE.REASONING, TOKEN_TYPE.RESPONSE)
            ]

            real_reasoning_tokens = [
                t for t, tp in zip(real_tokens_sorted, token_types_sorted)
                if tp == TOKEN_TYPE.REASONING
            ]
            real_response_tokens = [
                t for t, tp in zip(real_tokens_sorted, token_types_sorted)
                if tp == TOKEN_TYPE.RESPONSE
            ]
            real_assistant_tokens = [
                t for t, tp in zip(real_tokens_sorted, token_types_sorted)
                if tp in (TOKEN_TYPE.REASONING, TOKEN_TYPE.RESPONSE)
            ]

            record = {
                "data_id": data_id,
                "num_tokens": len(token_ids_sorted),
                "token_id_start": token_ids_sorted[0] if token_ids_sorted else None,
                "token_id_end": token_ids_sorted[-1] if token_ids_sorted else None,
                "slm_predicted_full_text": tokenizer.decode(aligned_pred_tokens, skip_special_tokens=False),
                "slm_predicted_assistant_text": tokenizer.decode(pred_assistant_tokens, skip_special_tokens=False),
                "slm_predicted_reasoning_text": tokenizer.decode(pred_reasoning_tokens, skip_special_tokens=False),
                "slm_predicted_response_text": tokenizer.decode(pred_response_tokens, skip_special_tokens=False),
                "real_assistant_text": tokenizer.decode(real_assistant_tokens, skip_special_tokens=False),
                "real_reasoning_text": tokenizer.decode(real_reasoning_tokens, skip_special_tokens=False),
                "real_response_text": tokenizer.decode(real_response_tokens, skip_special_tokens=False),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Sequence-level SLM outputs saved to {output_jsonl}")


def apply_qwen_r1_chat_template(messages, add_generation_prompt=False):
    """
    Apply the Qwen R1 chat template to the messages. We rewrite the function to use the same template as the original one, adding the thinking process in the context. 
    The thinking process is originally excluded for multi-turn conversations.
    """
    prompt = "<｜begin▁of▁sentence｜>"
    ns = {
        "is_first": False,
        "is_tool": False,
        "is_output_first": True,
        "system_prompt": "",
    }

    # extract system prompt
    for message in messages:
        if message["role"] == "system":
            ns["system_prompt"] = message["content"]

    prompt += ns["system_prompt"]

    for message in messages:
        if message["role"] == "user":
            ns["is_tool"] = False
            prompt += "<｜User｜>" + message["content"]

        elif message["role"] == "assistant" and message["content"] is not None:
            content = message["content"]
            prompt += "<｜Assistant｜>" + content + "<｜end▁of▁sentence｜>"

    if add_generation_prompt:
        prompt += "<｜Assistant｜><think>\n"

    return prompt

def get_formatted_prompt(sample, dataset_path, tokenizer, model_name, enable_thinking=True):
    """Format prompt based on dataset type"""
    input_text = sample["input_text"]
    model_reasoning = sample["model_reasoning"]
    model_response = sample["model_response"]

    if model_reasoning == None or model_response == None:
        print(f"model_reasoning or model_response is None, skip")
        return None
    # print(f"model_reasoning in data_finished is not None! Its type is {type(model_reasoning)} and its value is: {model_reasoning} \n")
    input_text = sample["input_text"]

    messages = [
        {"role": "user", "content": input_text},
        {
            "role": "assistant",
            "content": None,
        },
    ]
    
    if "r1" in model_name.lower():
        if enable_thinking:
            messages[1]["content"] = f"<think>\n{model_reasoning}\n</think>\n\n" + model_response
        else:
            messages[1]["content"] = model_response
        prompt = apply_qwen_r1_chat_template(messages, add_generation_prompt=False)
    else:
        if enable_thinking:
            # messages[1]["content"] = f"{model_reasoning}\n</think>\n\n" + model_response
            messages[1]["content"] = f"<think>\n{model_reasoning}\n</think>\n\n" + model_response
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=True)
        else:
            messages[1]["content"] = model_response
            try:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
            except TypeError:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {
        "prompt": prompt,
        "assistant_content": messages[1]["content"],
    }


def categorize_token_types(input_ids, tokenizer):
    """
    Categorize tokens into INPUT_INSTRUCTION (0), REASONING (1), or RESPONSE (2)
    
    Args:
        input_ids: torch tensor of token IDs
        tokenizer: tokenizer used for encoding
    
    Returns:
        List of token type categories
    """
    THINK_START_TOKEN = MODEL_DICT["special_tokens"]["think_start"]
    THINK_END_TOKEN = MODEL_DICT["special_tokens"]["think_end"]
    
    token_types = []
    current_type = TOKEN_TYPE.INPUT_INSTRUCTION
    
    for i, token_id in enumerate(input_ids[0]):
        token_id = token_id.item()
        
        if token_id == THINK_START_TOKEN:
            current_type = TOKEN_TYPE.REASONING
        elif token_id == THINK_END_TOKEN:
            current_type = TOKEN_TYPE.RESPONSE
            
        token_types.append(current_type)
    
    return token_types


def _find_last_subsequence_start(sequence, subsequence):
    """Return last start index of subsequence in sequence, or -1 if not found."""
    n = len(sequence)
    m = len(subsequence)
    if m == 0 or m > n:
        return -1

    for start in range(n - m, -1, -1):
        if sequence[start : start + m] == subsequence:
            return start
    return -1


def categorize_token_types_no_thinking(input_ids, assistant_token_ids):
    """Categorize tokens when thinking is disabled by matching assistant content token span."""
    token_seq = input_ids[0].tolist()
    assistant_seq = assistant_token_ids.tolist()

    token_types = [TOKEN_TYPE.INPUT_INSTRUCTION] * len(token_seq)
    assistant_start = _find_last_subsequence_start(token_seq, assistant_seq)
    if assistant_start == -1:
        print(
            "Warning: Could not locate assistant content token span in prompt. "
            "Falling back to INPUT_INSTRUCTION labels for this sample."
        )
        return token_types

    assistant_end = assistant_start + len(assistant_seq)
    for i in range(assistant_start, assistant_end):
        token_types[i] = TOKEN_TYPE.RESPONSE

    return token_types


def sample_token_batched_sharded(logits, temperature=1.0, top_p=1.0, top_k=-1, shard_size=10000):
    """
    Sample tokens from batched logits with sharding for large batch sizes.
    
    Args:
        logits: Tensor of shape [batch_size, vocab_size]
        temperature: Temperature for sampling
        top_p: Top-p probability threshold for nucleus sampling
        top_k: Top-k threshold for sampling
        shard_size: Maximum size of each shard (default: 3000)
        
    Returns:
        Tensor of sampled token IDs for the entire batch
    """
    batch_size = logits.shape[0]
    
    # If batch size is smaller than shard size, process directly
    if batch_size <= shard_size:
        return sample_token(logits, temperature=temperature, top_p=top_p, top_k=top_k)
    
    # Split into shards and process each one
    results = []
    for i in range(0, batch_size, shard_size):
        end_idx = min(i + shard_size, batch_size)
        shard_logits = logits[i:end_idx]
        
        # Sample from this shard
        shard_predictions = sample_token(shard_logits, temperature=temperature, top_p=top_p, top_k=top_k)
        results.append(shard_predictions)
    
    # Concatenate all results
    return torch.cat(results, dim=0)


def process_dataset(args):
    """Process the dataset with all models and directly create the final prediction_comparison.csv"""
    # Create output directory
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Get model sizes for organizing output
    all_model_sizes = [
        parse_model_size(model, idx) for idx, model in enumerate(args.test_model_list)
    ]
    all_model_sizes.sort()
    print(f"Model sizes: {all_model_sizes}")

    # Load dataset
    print(f"Loading local dataset from {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)

    # Handle dataset splits
    if hasattr(dataset, "keys"):
        if "train" in dataset.keys():
            dataset = dataset["train"]
        elif "test" in dataset.keys():
            dataset = dataset["test"]

    # Limit dataset size if specified
    if args.index_range is not None:
        start_idx, end_idx = args.index_range
        dataset = dataset.select(range(start_idx, end_idx))

    print(f"Dataset length: {len(dataset)}")

    # Dictionary to store all predictions per model
    all_predictions = {}
    # Initialize lists to store common data
    all_real_tokens = []
    all_token_ids = []
    all_data_ids = []
    all_token_types = []

    # Process each model
    for model_idx, model_name in enumerate(args.test_model_list):
        model_size = parse_model_size(model_name, model_idx)
        model_path = get_model_tag(model_name)

        # Skip if results already exist
        if os.path.exists(
            os.path.join(args.output_path, f"results_test_{model_path}.pth")
        ):
            print(f"Results for {model_name} already exist, loading from file.")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            results_dict = torch.load(
                os.path.join(args.output_path, f"results_test_{model_path}.pth"),
                weights_only=False,
            )
            all_predictions[model_size] = results_dict["predictions"]

            # Export raw SLM outputs to JSONL from cached results
            save_slm_raw_jsonl(
                output_path=args.output_path,
                model_tag=model_path,
                predictions=results_dict["predictions"].cpu().tolist(),
                token_ids=results_dict["token_id"].cpu().tolist(),
                data_ids=results_dict["data_id"].cpu().tolist(),
                token_types=results_dict["token_type"].cpu().tolist(),
                real_tokens=results_dict["real_token"].cpu().tolist(),
                tokenizer=tokenizer,
            )
            save_slm_sequence_jsonl(
                output_path=args.output_path,
                model_tag=model_path,
                predictions=results_dict["predictions"].cpu().tolist(),
                token_ids=results_dict["token_id"].cpu().tolist(),
                data_ids=results_dict["data_id"].cpu().tolist(),
                token_types=results_dict["token_type"].cpu().tolist(),
                real_tokens=results_dict["real_token"].cpu().tolist(),
                tokenizer=tokenizer,
            )

            # Also load common data if we haven't processed any models yet
            if not all_real_tokens:
                all_real_tokens.append(results_dict["real_token"])
                all_token_ids.append(results_dict["token_id"]) 
                all_data_ids.append(results_dict["data_id"])
                all_token_types.append(results_dict["token_type"])
            continue

        # Load tokenizer for this model
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Load model
        model = load_model(model_name)
        if model is None:
            continue

        # Store results for this model
        predictions_list = []
        real_tokens_list = []
        token_ids_list = []
        data_ids_list = []
        token_types_list = []
        all_hidden_states = []
        all_top_logits = []
        all_top_logits_indices = []

        # Process each sample
        pbar = tqdm(total=len(dataset), desc=f"Processing {model_path}")
        with torch.no_grad():
            for data_id, sample in enumerate(dataset):
                # Get formatted prompt
                formatted_prompt = get_formatted_prompt(
                    sample,
                    args.dataset_path,
                    tokenizer,
                    model_name,
                    enable_thinking=args.enable_thinking,
                )
                if formatted_prompt is None:
                    pbar.update(1)
                    continue
                prompt = formatted_prompt["prompt"]

                # Tokenize
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
                    _get_model_device(model)
                )

                # Skip if too long
                if len(input_ids[0]) > args.max_input_length:
                    print(
                        f"Input length {len(input_ids[0])} exceeds max length {args.max_input_length}, skipping"
                    )
                    pbar.update(1)
                    continue

                # Run inference with output_hidden_states=True
                outputs = model(input_ids, output_hidden_states=True)
                # logits = outputs.logits # qwen2.5vl does not return logits, need to reconstruct from hidden states
                logits = _extract_logits(model, outputs)
                
                # Use batched sharded sampling instead of single sequence sampling
                pred = sample_token_batched_sharded(logits[0], temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, shard_size=3000)

                pred = pred.cpu()

                # Extract token IDs and data IDs
                token_id = torch.arange(0, input_ids.shape[-1], 1).cpu()
                data_id_tensor = torch.full_like(token_id, data_id).cpu()

                # Extract real tokens
                real_token = input_ids[0].cpu()

                # Categorize token types
                print(f"Type of args.enable_thinking: {type(args.enable_thinking)}, value: {args.enable_thinking}\n")
                if args.enable_thinking:
                    token_types = categorize_token_types(input_ids, tokenizer)
                else:
                    print("Thinking is disabled, categorizing token types by matching assistant content span.\n")
                    print(f"assistant_content: {formatted_prompt['assistant_content']}\n")
                    assistant_token_ids = tokenizer(
                        formatted_prompt["assistant_content"],
                        return_tensors="pt",
                        add_special_tokens=False,
                    ).input_ids[0]
                    token_types = categorize_token_types_no_thinking(
                        input_ids.cpu(), assistant_token_ids
                    )
                token_types_tensor = torch.tensor(token_types, dtype=torch.int32).cpu()

                # Extract top logits (top 100 to match small_ref)
                top_logits, top_logits_indices = torch.topk(logits[0], 100, dim=-1)
                # Convert to float32 to match small_ref format
                top_logits = top_logits.float().cpu()
                top_logits_indices = top_logits_indices.cpu()

                # Extract hidden states
                hidden_states = outputs.hidden_states[-1][0].cpu()

                # Append to lists
                predictions_list.append(pred)
                real_tokens_list.append(real_token)
                token_ids_list.append(token_id)
                data_ids_list.append(data_id_tensor)
                token_types_list.append(token_types_tensor)
                all_hidden_states.append(hidden_states)
                all_top_logits.append(top_logits)
                all_top_logits_indices.append(top_logits_indices)

                # pbar.write(f"Model: {model_name} | Input length: {len(input_ids[0])}")
                pbar.update(1)

        pbar.close()

        # Concatenate all results
        predictions = torch.cat(predictions_list, dim=0)
        real_tokens = torch.cat(real_tokens_list, dim=0)
        token_ids = torch.cat(token_ids_list, dim=0)
        data_ids = torch.cat(data_ids_list, dim=0)
        token_types = torch.cat(token_types_list, dim=0)

        # Store predictions in the dictionary
        all_predictions[model_size] = predictions
        all_real_tokens.append(real_tokens)
        all_token_ids.append(token_ids)
        all_data_ids.append(data_ids)
        all_token_types.append(token_types)

        # Save top logits and hidden states
        all_top_logits_tensor = torch.cat(all_top_logits, dim=0)
        all_top_logits_indices_tensor = torch.cat(all_top_logits_indices, dim=0)
        all_hidden_states_tensor = torch.cat(all_hidden_states, dim=0)

        # Save only top logits, indices and hidden states with proper naming
        torch.save(
            all_top_logits_tensor,
            os.path.join(args.output_path, f"SLM_top_logits.pt"),
        )
        torch.save(
            all_top_logits_indices_tensor,
            os.path.join(args.output_path, f"SLM_top_logits_indices.pt"),
        )
        torch.save(
            all_hidden_states_tensor,
            os.path.join(args.output_path, f"SLM_hidden_states.pt"),
        )

        # Also save all in one file to match small_ref format
        results_dict = {
            "predictions": predictions,
            "token_id": token_ids,
            "data_id": data_ids,
            "token_type": token_types,
            "top_logits": all_top_logits_tensor,
            "top_logits_index": all_top_logits_indices_tensor,
            "real_token": real_tokens,
        }
        torch.save(
            results_dict,
            os.path.join(args.output_path, f"results_test_{model_path}.pth"),
        )

        # Export raw SLM outputs to JSONL for easier downstream consumption
        save_slm_raw_jsonl(
            output_path=args.output_path,
            model_tag=model_path,
            predictions=predictions.cpu().tolist(),
            token_ids=token_ids.cpu().tolist(),
            data_ids=data_ids.cpu().tolist(),
            token_types=token_types.cpu().tolist(),
            real_tokens=real_tokens.cpu().tolist(),
            tokenizer=tokenizer,
        )
        save_slm_sequence_jsonl(
            output_path=args.output_path,
            model_tag=model_path,
            predictions=predictions.cpu().tolist(),
            token_ids=token_ids.cpu().tolist(),
            data_ids=data_ids.cpu().tolist(),
            token_types=token_types.cpu().tolist(),
            real_tokens=real_tokens.cpu().tolist(),
            tokenizer=tokenizer,
        )

    # If we have processed at least one model, combine the common data
    if all_real_tokens:
        real_tokens = torch.cat(all_real_tokens, dim=0)
        token_ids = torch.cat(all_token_ids, dim=0)
        data_ids = torch.cat(all_data_ids, dim=0)
        token_types = torch.cat(all_token_types, dim=0)
    else:
        # Get data from existing results files
        for model_name in args.test_model_list:
            model_path = get_model_tag(model_name)
            results_file = os.path.join(
                args.output_path, f"results_test_{model_path}.pth"
            )
            if os.path.exists(results_file):
                print(f"Loading data from existing results file: {results_file}")
                results_dict = torch.load(results_file, weights_only=False)
                real_tokens = results_dict["real_token"]
                token_ids = results_dict["token_id"]
                data_ids = results_dict["data_id"]
                token_types = results_dict["token_type"]
                break
        else:
            print("No results files found and no models were processed.")
            return

    # Create data analysis CSV directly
    create_data_analysis(
        output_path=args.output_path,
        model_sizes=all_model_sizes,
        real_tokens=real_tokens,
        token_ids=token_ids,
        data_ids=data_ids,
        token_types=token_types,
        all_predictions=all_predictions,
        top_k=args.top_k,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print("All processing completed!")


def create_data_analysis(
    output_path,
    model_sizes,
    real_tokens,
    token_ids,
    data_ids,
    token_types,
    all_predictions,
    top_k=-1,
    temperature=0.6,
    top_p=1.0,
):
    """Create prediction_comparison.csv directly from collected data"""
    # Create DataFrame with common data
    df = pd.DataFrame(
        {
            "row_id": range(len(real_tokens)),
            "real_token": real_tokens.numpy(),
            "token_id": token_ids.numpy(),
            "data_id": data_ids.numpy(),
            "token_type": token_types.numpy(),
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Add predictions from each model
    for model_size in tqdm(model_sizes, desc="Processing model sizes"):
        if model_size in all_predictions:
            df["SLM_predictions"] = all_predictions[model_size].cpu().numpy()

            # Load top logits and indices for this model
            top_logits = torch.load(
                os.path.join(output_path, f"SLM_top_logits.pt"),
                weights_only=False,
                map_location=device,
            )
            top_indices = torch.load(
                os.path.join(output_path, f"SLM_top_logits_indices.pt"),
                weights_only=False,
                map_location=device,
            )

            # Apply temperature and convert to probabilities
            probs = torch.nn.functional.softmax(top_logits / temperature, dim=-1)

            # Vectorized top-p calculation
            # Sort probabilities and get corresponding indices within the top_k dimension
            sorted_probs, indices_in_sorted = torch.sort(probs, dim=-1, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

            # Create mask for top-p
            mask = cumsum_probs <= top_p
            mask[:, 0] = True  # Ensure the top token is always included

            # Create list to store variable-length prediction samples
            all_samples = []

            # Iterate through each row to apply the mask and get final token indices
            for i in tqdm(
                range(probs.shape[0]), desc=f"Processing {model_size} predictions"
            ):
                row_mask = mask[i]
                row_indices_in_sorted = indices_in_sorted[i]
                row_top_indices = top_indices[
                    i
                ]  # Original token indices from the loaded data

                # Get the indices within the sorted list that satisfy the top-p condition
                filtered_indices_in_sorted = row_indices_in_sorted[row_mask]

                # Limit by top_k
                if top_k != -1:
                    k = min(top_k, len(filtered_indices_in_sorted))
                    final_indices_in_sorted = filtered_indices_in_sorted[:k]
                else:
                    final_indices_in_sorted = filtered_indices_in_sorted

                # Map these indices back to the original token IDs using the loaded top_indices
                final_token_ids = row_top_indices[final_indices_in_sorted]

                # Add to list
                all_samples.append(final_token_ids.cpu().tolist())

            # Add variable-length predictions to dataframe
            df["SLM_prediction_samples"] = all_samples

    # Save to CSV
    output_file = os.path.join(output_path, "prediction_comparison.csv")
    df.to_csv(output_file, index=False)
    print(f"Data analysis saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run model inference on datasets and save predictions directly"
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to the local dataset"
    )
    parser.add_argument(
        "--test_model_list",
        nargs="+",
        type=str,
        help="List of test models to run (supports HF model names or local model paths)",
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Directory to save output files"
    )
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=32768,
        help="Maximum length of input tokens",
    )
    parser.add_argument(
        "--index_range",
        nargs=2,
        type=int,
        default=None,
        help="Range of dataset samples to process [start_idx, end_idx]",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=-1,
        help="Number of top predictions to include in the output. If -1, no top-k filtering is applied.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature to apply to logits when calculating probabilities",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p probability threshold for nucleus sampling (0 < top_p ≤ 1)",
    )
    parser.add_argument(
        "--enable_thinking",
        type=str2bool,
        default=True,
        help="Whether to keep <think> content in assistant message when constructing prompts.",
    )
    args = parser.parse_args()

    if args.test_model_list is None:
        args.test_model_list=[f"{MODEL_DICT['quick']['model_path']}"]
    
    process_dataset(args)

    # save args as json
    with open(os.path.join(args.output_path, "args.json"), "w") as f:
        json.dump(args.__dict__, f)


if __name__ == "__main__":
    main()
