"""
Step 3 variant that keeps correctness verification and adds safety signals.

This script first uses the existing correctness judge from
`r2r.data.verify_model_acc` to produce:

- small_correct
- reference_correct
- verify_response

It then uses Qwen3Guard-Gen-8B, with the same prompt/label parsing style as
`/mnt/shared-storage-user/yangxuqing/inference/safejudge_inference.py`, to add:

- small_safe
- reference_safe
- small_safejudge_label
- reference_safejudge_label
- small_safejudge_response
- reference_safejudge_response
"""

import argparse
import math
import os
from typing import Iterable, List

import pandas as pd
from tqdm import tqdm

from r2r.data.generation_controller_original import DivergePoint
from r2r.data.verify_model_acc import VerifyModel
from r2r.data.verify_model_acc_safe import DEFAULT_SAFEJUDGE_MODEL, SafeJudgeModel
from r2r.utils.config import MODEL_DICT


CORRECTNESS_COLUMNS = ["small_correct", "reference_correct", "verify_response"]
SAFETY_COLUMNS = [
    "small_safe",
    "reference_safe",
    "small_safejudge_label",
    "reference_safejudge_label",
    "small_safejudge_response",
    "reference_safejudge_response",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify continuation correctness and add Qwen3Guard-Gen safety labels"
    )
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output_csv", type=str, default=None, help="Path to output CSV file")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for both judges")

    parser.add_argument(
        "--verify_model",
        type=str,
        default=None,
        help="Correctness judge model. Defaults to MODEL_DICT['verify']['model_path']",
    )
    parser.add_argument(
        "--verify_mode",
        type=str,
        default="common_context",
        choices=["common_context"],
        help="Correctness judgment mode",
    )
    parser.add_argument("--tp_size", type=int, default=4, help="Tensor parallel size for correctness judge")
    parser.add_argument("--mem_fraction", type=float, default=0.9, help="GPU memory fraction for correctness judge")

    parser.add_argument(
        "--safe_model",
        type=str,
        default=DEFAULT_SAFEJUDGE_MODEL,
        help="Safety judge model path",
    )
    parser.add_argument(
        "--safe_gpu_ids",
        type=str,
        default=None,
        help="Optional comma-separated CUDA_VISIBLE_DEVICES value for Qwen3Guard-Gen",
    )
    parser.add_argument("--safe_tp_size", type=int, default=1, help="Tensor parallel size for safety judge")
    parser.add_argument("--safe_mem_fraction", type=float, default=0.8, help="GPU memory fraction for safety judge")
    parser.add_argument("--safe_max_tokens", type=int, default=32, help="Max generated tokens for safety judge")
    parser.add_argument("--safe_max_model_len", type=int, default=30000, help="Max model length for safety judge")
    parser.add_argument("--safe_max_num_seqs", type=int, default=256, help="Max vLLM sequences for safety judge")

    parser.add_argument(
        "--save_interval",
        type=int,
        default=10,
        help="Save the full output CSV every N batches. 0 saves only at each phase end.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from an existing output CSV")
    parser.add_argument(
        "--skip_correctness",
        action="store_true",
        help="Only run Qwen3Guard-Gen safety labels; requires existing correctness columns if needed downstream.",
    )
    parser.add_argument(
        "--skip_safety",
        action="store_true",
        help="Only run correctness labels.",
    )
    return parser.parse_args()


def convert_row_to_diverge_point(row) -> DivergePoint:
    diverge_point = DivergePoint(
        data_id=row.get("data_id", 0),
        token_id=row.get("token_id", 0),
        small_diverge_text=row["small_diverge_text"],
        reference_diverge_text=row["reference_diverge_text"],
        common_context=row["common_context"],
        pred_small_token=row.get("pred_small_token", []),
        pred_small_text=row.get("pred_small_text", ""),
    )
    for question_column in ("input_text", "question", "prompt"):
        if question_column in row and pd.notna(row[question_column]):
            diverge_point.question = row[question_column]
            break
    return diverge_point


def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column not in df.columns:
            df[column] = None


def save_results(df: pd.DataFrame, output_csv: str) -> None:
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to {output_csv}")
    df.to_csv(output_csv, index=False)


def missing_rows(df: pd.DataFrame, columns: List[str]) -> List[int]:
    mask = pd.Series(False, index=df.index)
    for column in columns:
        mask = mask | df[column].isna()
    return df.index[mask].tolist()


def validate_input(df: pd.DataFrame) -> None:
    required_columns = ["small_diverge_text", "reference_diverge_text", "common_context"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {missing_columns}")


def batched(indices: List[int], batch_size: int):
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def run_correctness_phase(args, df: pd.DataFrame) -> pd.DataFrame:
    ensure_columns(df, CORRECTNESS_COLUMNS)
    target_indices = missing_rows(df, CORRECTNESS_COLUMNS)
    if not target_indices:
        print("Correctness columns are already complete.")
        return df

    verify_model_path = args.verify_model or MODEL_DICT["verify"]["model_path"]
    max_new_tokens = MODEL_DICT["verify"]["max_new_tokens"]
    apply_chat_template_kwargs = MODEL_DICT["verify"].get("apply_chat_template_kwargs", None)

    verify_model = VerifyModel(
        model_name=verify_model_path,
        verify_mode=args.verify_mode,
        max_new_tokens=max_new_tokens,
        mem_fraction_static=args.mem_fraction,
        tp_size=args.tp_size,
        apply_chat_template_kwargs=apply_chat_template_kwargs,
    )

    print(f"Using correctness judge model: {verify_model_path}")
    print(f"Correctness rows to process: {len(target_indices)}")
    num_batches = math.ceil(len(target_indices) / args.batch_size)

    try:
        for batch_idx, batch_indices in enumerate(tqdm(list(batched(target_indices, args.batch_size)), desc="Correctness batches")):
            batch_df = df.loc[batch_indices]
            batch_diverge_points = [
                convert_row_to_diverge_point(row)
                for _, row in batch_df.iterrows()
            ]
            batch_comparison_points = verify_model.batch_compare_diverge_points(batch_diverge_points)

            for row_idx, point in zip(batch_indices, batch_comparison_points):
                df.at[row_idx, "small_correct"] = point.small_correct
                df.at[row_idx, "reference_correct"] = point.reference_correct
                df.at[row_idx, "verify_response"] = point.verify_response

            if args.save_interval > 0 and (batch_idx + 1) % args.save_interval == 0:
                save_results(df, args.output_csv)

        save_results(df, args.output_csv)
    finally:
        verify_model.shutdown()

    print(f"Finished correctness phase ({num_batches} batches).")
    return df


def run_safety_phase(args, df: pd.DataFrame) -> pd.DataFrame:
    ensure_columns(df, SAFETY_COLUMNS)
    target_indices = missing_rows(df, SAFETY_COLUMNS)
    if not target_indices:
        print("Safety columns are already complete.")
        return df

    safe_model = SafeJudgeModel(
        model_name=args.safe_model,
        gpu_ids=args.safe_gpu_ids,
        max_new_tokens=args.safe_max_tokens,
        mem_fraction_static=args.safe_mem_fraction,
        tp_size=args.safe_tp_size,
        max_model_len=args.safe_max_model_len,
        max_num_seqs=args.safe_max_num_seqs,
    )

    print(f"Using safety judge model: {args.safe_model}")
    print(f"Safety rows to process: {len(target_indices)}")
    num_batches = math.ceil(len(target_indices) / args.batch_size)

    try:
        for batch_idx, batch_indices in enumerate(tqdm(list(batched(target_indices, args.batch_size)), desc="Safety batches")):
            batch_df = df.loc[batch_indices]
            batch_diverge_points = [
                convert_row_to_diverge_point(row)
                for _, row in batch_df.iterrows()
            ]
            batch_safety_points = safe_model.batch_compare_diverge_points(batch_diverge_points)

            for row_idx, point in zip(batch_indices, batch_safety_points):
                df.at[row_idx, "small_safe"] = point.small_safe
                df.at[row_idx, "reference_safe"] = point.reference_safe
                df.at[row_idx, "small_safejudge_label"] = point.small_safejudge_label
                df.at[row_idx, "reference_safejudge_label"] = point.reference_safejudge_label
                df.at[row_idx, "small_safejudge_response"] = point.small_safejudge_response
                df.at[row_idx, "reference_safejudge_response"] = point.reference_safejudge_response

            if args.save_interval > 0 and (batch_idx + 1) % args.save_interval == 0:
                save_results(df, args.output_csv)

        save_results(df, args.output_csv)
    finally:
        safe_model.shutdown()

    print(f"Finished safety phase ({num_batches} batches).")
    return df


def main():
    args = parse_args()

    if args.output_csv is None:
        args.output_csv = args.input_csv.replace(".csv", "_verify_acc_safe.csv")

    input_path = args.output_csv if args.resume and os.path.exists(args.output_csv) else args.input_csv
    print(f"Loading CSV from {input_path}")
    df = pd.read_csv(input_path)
    validate_input(df)

    ensure_columns(df, CORRECTNESS_COLUMNS + SAFETY_COLUMNS)

    if not args.skip_correctness:
        df = run_correctness_phase(args, df)
    else:
        print("Skipping correctness phase by request.")

    if not args.skip_safety:
        df = run_safety_phase(args, df)
    else:
        print("Skipping safety phase by request.")

    save_results(df, args.output_csv)
    print("Done!")


if __name__ == "__main__":
    main()
