#!/usr/bin/env python3
"""Convert per-item temp CSV evaluation outputs into inference-style JSONL.

Example:
    python R2R/script/evaluate/extract_temp_csv_jsonl.py \
        /path/to/eval_run/temp_csv

By default, the output is written next to the temp_csv directory as an
indented JSON array with a .jsonl suffix, matching own-inference outputs:
    /path/to/eval_run/<eval_run_name>.jsonl
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


CSV_NAME_RE = re.compile(r"^(?P<item_idx>.+)_run_(?P<run_idx>\d+)\.csv$")
CHAT_USER_RE = re.compile(
    r"<\|im_start\|>user\n(?P<content>.*?)(?:<\|im_end\|>|\Z)",
    re.DOTALL,
)
OPTION_RE = re.compile(
    r"(?ms)^([A-D])\)\s*(.*?)(?=^[A-D]\)\s*|\Z)",
)


def parse_int_if_possible(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit() or (stripped.startswith("-") and stripped[1:].isdigit()):
            return int(stripped)
    return value


def parse_csv_name(path: Path) -> Tuple[Any, Optional[int]]:
    match = CSV_NAME_RE.match(path.name)
    if not match:
        return path.stem, None
    return (
        parse_int_if_possible(match.group("item_idx")),
        int(match.group("run_idx")),
    )


def natural_csv_sort_key(path: Path) -> Tuple[int, Any, int, str]:
    item_idx, run_idx = parse_csv_name(path)
    if isinstance(item_idx, int):
        item_sort = (0, item_idx)
    else:
        item_sort = (1, str(item_idx))
    return item_sort[0], item_sort[1], run_idx or 0, path.name


def iter_csv_rows(temp_csv_dir: Path) -> Iterable[Tuple[Path, Dict[str, str]]]:
    for csv_path in sorted(temp_csv_dir.glob("*.csv"), key=natural_csv_sort_key):
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield csv_path, row


def choose_first_nonempty(row: Dict[str, str], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return ""


def extract_user_content(text_prompt: str) -> str:
    match = CHAT_USER_RE.search(text_prompt or "")
    if match:
        return match.group("content").strip()
    return (text_prompt or "").strip()


def parse_mmlu_question_and_choices(user_content: str, row: Dict[str, str]) -> Tuple[str, str]:
    choices = [
        row.get("option_A") or "",
        row.get("option_B") or "",
        row.get("option_C") or "",
        row.get("option_D") or "",
    ]

    parsed_options = OPTION_RE.findall(user_content)
    if parsed_options:
        option_map = {letter: text.strip() for letter, text in parsed_options}
        choices = [
            option_map.get("A", choices[0]),
            option_map.get("B", choices[1]),
            option_map.get("C", choices[2]),
            option_map.get("D", choices[3]),
        ]

    question = user_content
    first_option = re.search(r"(?m)^A\)\s*", user_content)
    if first_option:
        question = user_content[: first_option.start()].strip()

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", question) if part.strip()]
    if len(paragraphs) > 1:
        question = paragraphs[-1]

    # mmlu_inference_sglang.py mutates data[item_idx]["choices"] to this string
    # before unpacking **data[item_idx] into the result.
    return question, "Check text prompt!"


def build_inference_result(csv_path: Path, row: Dict[str, str]) -> Dict[str, Any]:
    filename_item_idx, filename_run_idx = parse_csv_name(csv_path)
    batch_item_index = parse_int_if_possible(row.get("problem_id", filename_item_idx))
    generated_idx = filename_run_idx or parse_int_if_possible(row.get("generated_idx", 1))
    if not isinstance(generated_idx, int):
        generated_idx = 1

    text_prompt = extract_user_content(row.get("text_prompt") or "")
    question, choices = parse_mmlu_question_and_choices(text_prompt, row)
    answer = choose_first_nonempty(row, "answer", "correct_answer", "ground_truth", "Answer")

    return {
        "batch_item_index": batch_item_index,
        "generated_idx": generated_idx,
        "text_prompt": text_prompt,
        "model_output": choose_first_nonempty(row, "model_output", "generated_text", "full_output"),
        "ground_truth": choose_first_nonempty(row, "ground_truth", "correct_answer", "Answer", "answer"),
        "question": question,
        "choices": choices,
        "answer": answer,
    }


def build_legacy_result(csv_path: Path, row: Dict[str, str]) -> Dict[str, Any]:
    result = build_inference_result(csv_path, row)
    result.update(row)
    return result


def default_output_path(temp_csv_dir: Path) -> Path:
    parent = temp_csv_dir.parent
    return parent / f"{parent.name}.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract temp_csv evaluation outputs into an inference-style .jsonl file."
    )
    parser.add_argument(
        "temp_csv_dir",
        type=Path,
        help="Path to a temp_csv directory containing files like 0_run_1.csv.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path. Defaults to <temp_csv_dir_parent>/<run_dir_name>.jsonl.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--line-jsonl",
        dest="line_jsonl",
        action="store_true",
        help="Write separate JSON objects instead of the default indented JSON array.",
    )
    parser.add_argument(
        "--jsonl",
        dest="line_jsonl",
        action="store_true",
        help="Alias for --line-jsonl.",
    )
    parser.add_argument(
        "--pretty-jsonl",
        dest="pretty_jsonl",
        action="store_true",
        default=True,
        help="Indent each object when --line-jsonl is used. This is the default.",
    )
    parser.add_argument(
        "--compact-jsonl",
        dest="pretty_jsonl",
        action="store_false",
        help="Write each JSONL object on a single line.",
    )
    parser.add_argument(
        "--include-csv-metadata",
        action="store_true",
        help="Also include all original CSV columns after the inference-style fields.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    temp_csv_dir = args.temp_csv_dir.expanduser().resolve()
    if not temp_csv_dir.is_dir():
        raise FileNotFoundError(f"temp_csv directory not found: {temp_csv_dir}")

    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else default_output_path(temp_csv_dir)
    )
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path} (use --overwrite)")

    csv_files = sorted(temp_csv_dir.glob("*.csv"), key=natural_csv_sort_key)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {temp_csv_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_result = build_legacy_result if args.include_csv_metadata else build_inference_result
    results: List[Dict[str, Any]] = []
    count = 0

    if args.line_jsonl:
        with output_path.open("w", encoding="utf-8") as f:
            for csv_path, row in iter_csv_rows(temp_csv_dir):
                json.dump(
                    build_result(csv_path, row),
                    f,
                    ensure_ascii=False,
                    indent=4 if args.pretty_jsonl else None,
                )
                f.write("\n")
                count += 1
    else:
        for csv_path, row in iter_csv_rows(temp_csv_dir):
            results.append(build_result(csv_path, row))
            count += 1
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
            f.write("\n")

    print(f"Wrote {count} rows to {output_path}")


if __name__ == "__main__":
    main()
