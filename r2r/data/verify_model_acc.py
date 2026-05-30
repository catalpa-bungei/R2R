import re
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
import sglang as sgl
from sglang.srt.hf_transformers_utils import get_tokenizer
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict, Any
import pandas as pd
import os

from r2r.data.data_process import MismatchPoint
from r2r.data.generation_controller import DivergePoint, ModelController
from r2r.utils.config import MODEL_DICT
import r2r.models.sglang_patch.sgl_engine_patcher

ACCURACY_SYSTEM_PROMPT = "You are a precise correctness evaluator."

@dataclass
class ComparisonPoint:
    """Represents a comparison point with judgment results"""
    # Identifiers
    data_id: int
    token_id: int

    # Prediction data
    pred_small_token: List[int]
    pred_small_text: str

    # Judgment data
    small_diverge_text: str
    reference_diverge_text: str
    common_context: str
    question: str = ""
    small_correct: int = None  # 1 if the SLM continuation is correct, else 0
    reference_correct: int = None  # 1 if the LLM continuation is correct, else 0
    verify_response: str = None  # Response from the model

    def print(self):
        """Print comparison information in a formatted way"""
        print(f"Question: {self.question}")
        print(f"Common context: {self.common_context}")
        print(f"Refer diverge text: {self.reference_diverge_text}")
        print(f"Small diverge text: {self.small_diverge_text}")
        print(f"Verify response: {self.verify_response}")
        print(f"Small correct: {self.small_correct}")
        print(f"Reference correct: {self.reference_correct}")

class VerifyModel:
    """Model for judging correctness of two text continuations independently"""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        verify_mode: str = "common_context",
        max_new_tokens: int = 128,
        mem_fraction_static: float = 0.5,
        tp_size: int = 2,
        base_gpu_id: int = 0,
        apply_chat_template_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.device = device
        self.model_name = model_name
        self.verify_mode = verify_mode
        self.apply_chat_template_kwargs = apply_chat_template_kwargs or {}

        print(f"Loading verify model {self.model_name}...")
        # Using HuggingFace tokenizer directly for token-based processing
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

        # Initialize Engine with skip_tokenizer_init=True for token-based processing
        self.model = sgl.Engine(
            model_path=self.model_name,
            dtype="bfloat16",
            mem_fraction_static=mem_fraction_static,
            skip_tokenizer_init=True,
            tp_size=tp_size,
            base_gpu_id=base_gpu_id
        )

        self.max_new_tokens = max_new_tokens

        # Set system prompt based on verify_mode
        if verify_mode == "common_context":
            self.system_prompt = ACCURACY_SYSTEM_PROMPT
        else:
            raise ValueError(f"Invalid verify mode: {verify_mode}")

        print(f"Using {verify_mode} mode for verifying")

        # Create the system prompt message
        self.system_message = [{"role": "system", "content": self.system_prompt}]

    @staticmethod
    def get_accuracy_user_message_common_context(question: str, text1: str, text2: str, text3: str) -> str:
        """Create a prompt that asks for independent solution-usefulness judgments."""
        accuracy_user_prompt = f"""**Task:**
Judge whether each continuation is solution-useful and correct independently for the given question and reasoning context.

A continuation is correct only if it helps preserve or advance a valid path toward the final answer. A sentence that is true in isolation but irrelevant, vacuous, or unhelpful for solving the question must be marked incorrect.

**Instructions:**
- The marker `<< >>` indicates the token continuation point. It is **not** part of the original text.
- Do not compare Sentence 1 against Sentence 2 to decide which is better.
- Judge each sentence against the Question, the Common Context, and the goal of reaching the final answer.
- Output `1` only when the sentence is both locally valid and useful for the subsequent reasoning or final answer.
- Mark a sentence `0` if it introduces a factual error, invalid reasoning, contradiction, impossible step, wrong conclusion, or unsupported claim.
- Mark a sentence `0` if it is true but irrelevant to the question, generic filler, a vacuous restatement, or a detour that does not help reach the final answer.
- A sentence does not need to contain the final answer to be `1`; a partial step is correct if it is relevant, logically justified, and helps the solution progress.

**Output format:**
Small Correct: <0 or 1>
Reference Correct: <0 or 1>
Reasoning: <brief explanation>

---

### Example:
Question:
\"\"\"
What is 12 + 7?
\"\"\"
Common Context:
\"\"\"
We need compute 12 + 7. Since 12 + 7 =
\"\"\"
Sentence 1:
\"\"\"
<<19>>, the answer is 19.
\"\"\"
Sentence 2:
\"\"\"
<<20>>, the answer is 20.
\"\"\"
Small Correct: 1
Reference Correct: 0
Reasoning: Sentence 1 completes the arithmetic correctly, while Sentence 2 gives an incorrect sum.

### Example 2:
Question:
\"\"\"
What is 12 + 7?
\"\"\"
Common Context:
\"\"\"
We need compute 12 + 7.
\"\"\"
Sentence 1:
\"\"\"
<<The sky is blue.>>
\"\"\"
Sentence 2:
\"\"\"
<<12 + 7 = 19.>>
\"\"\"
Small Correct: 0
Reference Correct: 1
Reasoning: Sentence 1 may be true in isolation but is irrelevant and does not help solve the question. Sentence 2 is a useful correct step toward the final answer.

---

### Now complete the task:

Question:
\"\"\"
{question}
\"\"\"

Common Context:
\"\"\"
{text1}
\"\"\"

Sentence 1:
\"\"\"
{text2}
\"\"\"

Sentence 2:
\"\"\"
{text3}
\"\"\"

**Answer:**"""
        return accuracy_user_prompt

    def verify(self, text1: str, text2: str) -> Tuple[int, int, str]:
        """Accuracy verification requires common context; use verify_common_context."""
        raise NotImplementedError("Use verify_common_context(question, common_context, small_text, reference_text)")

    def verify_common_context(self, question: str, text1: str, text2: str, text3: str) -> Tuple[int, int, str]:
        """Verify correctness of two text continuations using token-based processing.

        Args:
            question: Original question or prompt.
            text1: Common context.
            text2: Sentence 1.
            text3: Sentence 2.

        Returns:
            Tuple of (small_correct, reference_correct, response)
            - small_correct: 1 if Sentence 1 is correct, 0 if incorrect, -1 on parse/error
            - reference_correct: 1 if Sentence 2 is correct, 0 if incorrect, -1 on parse/error
            - response: Response from the model
        """
        user_prompt = self.get_accuracy_user_message_common_context(question, text1, text2, text3)
        user_message = [{"role": "user", "content": user_prompt}]
        messages = self.system_message + user_message
        chat_template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        chat_template_kwargs.update(self.apply_chat_template_kwargs)
        full_prompt = self.tokenizer.apply_chat_template(messages, **chat_template_kwargs)
        input_token_ids = self.tokenizer.encode(full_prompt)

        sampling_params = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": 0.0
        }

        try:
            # Generate the response using token IDs directly
            outputs = self.model.generate(input_ids=[input_token_ids], sampling_params=sampling_params)

            # Get generated token IDs and decode to text
            output_token_ids = outputs[0]['output_ids']
            response = self.tokenizer.decode(output_token_ids, skip_special_tokens=True)
            small_correct, reference_correct = self._response_to_correctness(response)
            return small_correct, reference_correct, response
        except Exception as e:
            print(f"Error in token-based generation: {e}")
            return -1, -1, str(e)

    def compare_diverge_point(self, diverge_point: DivergePoint) -> ComparisonPoint:
        """Compare the two continuations in a DivergePoint and return a ComparisonPoint

        Args:
            diverge_point: DivergePoint containing the two continuations to compare

        Returns:
            ComparisonPoint with verify results
        """
        small_correct, reference_correct, verify_response = self.verify_common_context(
            getattr(diverge_point, "question", ""),
            diverge_point.common_context,
            diverge_point.small_diverge_text,
            diverge_point.reference_diverge_text
        )

        return ComparisonPoint(
            data_id=diverge_point.data_id,
            token_id=diverge_point.token_id,
            pred_small_token=diverge_point.pred_small_token,
            pred_small_text=diverge_point.pred_small_text,
            small_diverge_text=diverge_point.small_diverge_text,
            reference_diverge_text=diverge_point.reference_diverge_text,
            common_context=diverge_point.common_context,
            question=getattr(diverge_point, "question", ""),
            small_correct=small_correct,
            reference_correct=reference_correct,
            verify_response=verify_response
        )

    def batch_compare_diverge_points(self, diverge_points: List[DivergePoint]) -> List[ComparisonPoint]:
        """Compare multiple diverge points in a batch using token-based processing

        Args:
            diverge_points: List of DivergePoints to compare

        Returns:
            List of ComparisonPoints with verify results
        """
        comparison_points = []

        # Prepare all prompts and tokenize them for batch processing
        input_ids_list = []
        for diverge_point in diverge_points:
            if self.verify_mode == "common_context":
                user_prompt = self.get_accuracy_user_message_common_context(
                    getattr(diverge_point, "question", ""),
                    diverge_point.common_context,
                    diverge_point.small_diverge_text,
                    diverge_point.reference_diverge_text
                )

            else:
                raise ValueError(f"Invalid verify mode: {self.verify_mode}")

            # Prepare full prompt and tokenize
            user_message = [{"role": "user", "content": user_prompt}]
            messages = self.system_message + user_message

            # Merge default kwargs with user-provided kwargs
            chat_template_kwargs = {"tokenize": False, "add_generation_prompt": True}
            chat_template_kwargs.update(self.apply_chat_template_kwargs)

            full_prompt = self.tokenizer.apply_chat_template(messages, **chat_template_kwargs)
            input_token_ids = self.tokenizer.encode(full_prompt)
            input_ids_list.append(input_token_ids)

        # Prepare sampling parameters
        sampling_params = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": 0.0
        }

        # Execute batch generation using token IDs directly
        try:
            outputs = self.model.generate(input_ids=input_ids_list, sampling_params=sampling_params)

            # Process results
            for i, (output, diverge_point) in enumerate(zip(outputs, diverge_points)):
                # Get generated token IDs and decode to text
                output_token_ids = output['output_ids']
                response = self.tokenizer.decode(output_token_ids, skip_special_tokens=True)

                # Extract separate correctness labels from response
                small_correct, reference_correct = self._response_to_correctness(response)

                # Create comparison point
                comparison_point = ComparisonPoint(
                    data_id=diverge_point.data_id,
                    token_id=diverge_point.token_id,
                    pred_small_token=diverge_point.pred_small_token,
                    pred_small_text=diverge_point.pred_small_text,
                    small_diverge_text=diverge_point.small_diverge_text,
                    reference_diverge_text=diverge_point.reference_diverge_text,
                    question=getattr(diverge_point, "question", ""),
                    small_correct=small_correct,
                    reference_correct=reference_correct,
                    verify_response=response,
                    common_context=diverge_point.common_context
                )
                if hasattr(diverge_point, "pred_ref_token") and hasattr(diverge_point, "pred_ref_text"):
                    comparison_point.pred_ref_token = diverge_point.pred_ref_token
                    comparison_point.pred_ref_text = diverge_point.pred_ref_text

                comparison_points.append(comparison_point)

        except Exception as e:
            print(f"Error in batch token-based processing: {e}")
            # Create fallback comparison points with error for all diverge points
            for diverge_point in diverge_points:
                comparison_points.append(ComparisonPoint(
                    data_id=diverge_point.data_id,
                    token_id=diverge_point.token_id,
                    pred_small_token=diverge_point.pred_small_token,
                    pred_small_text=diverge_point.pred_small_text,
                    small_diverge_text=diverge_point.small_diverge_text,
                    reference_diverge_text=diverge_point.reference_diverge_text,
                    question=getattr(diverge_point, "question", ""),
                    small_correct=-1,
                    reference_correct=-1,
                    verify_response=f"Error: {str(e)}",
                    common_context=diverge_point.common_context
                ))

        return comparison_points

    def _response_to_correctness(self, response: str) -> Tuple[int, int]:
        """Convert the response to separate correctness labels."""
        small_match = re.search(r'(?i)\bsmall\s+correct\b[^0-9]*([01])', response)
        reference_match = re.search(r'(?i)\breference\s+correct\b[^0-9]*([01])', response)
        if small_match and reference_match:
            return int(small_match.group(1)), int(reference_match.group(1))

        digits = re.findall(r'\b[01]\b', response)
        if len(digits) >= 2:
            return int(digits[0]), int(digits[1])

        print(f"Warning: Could not parse separate correctness labels from response: {response}")
        return -1, -1

    def shutdown(self):
        """Shut down the Engine instance to free resources"""
        try:
            self.model.shutdown()
            print(f"VerifyModel engine shut down successfully")
        except Exception as e:
            print(f"Error shutting down engine: {str(e)}")


def data_points_to_df(
    comparison_points: List[ComparisonPoint],
    mismatch_points: Dict[Tuple[int, int], MismatchPoint],
    comparison_model: str,
    is_verify: bool = True,
) -> pd.DataFrame:
    """Convert a list of ComparisonPoints to a pandas DataFrame, combining with MismatchPoint data

    Args:
        comparison_points: List of ComparisonPoints to convert
        mismatch_points: Dictionary of MismatchPoints by (data_id, token_id)
        comparison_model: Model to use for comparison ('reference' or 'real')
        is_verify: Whether verify model is being used. If False, don't include similarity score andverify response

    Returns:
        DataFrame containing comparison and mismatch data
    """
    # Convert each ComparisonPoint to a dict
    data = []
    for comparison_point in tqdm(comparison_points, desc="Converting comparison points to DataFrame", leave=False):
        # Get the corresponding mismatch point using data_id and token_id as keys
        mismatch_point = mismatch_points.get((comparison_point.data_id, comparison_point.token_id))

        if mismatch_point is None:
            # Skip if no matching mismatch point (this shouldn't happen with proper implementation)
            continue

        point_dict = {
            # Basic identifiers
            "data_id": comparison_point.data_id,
            "token_id": comparison_point.token_id,
            # Original tokens and predictions from MismatchPoint
            "real_token": mismatch_point.real_token,
            "real_text": mismatch_point.real_text,
            "pred_small_token": comparison_point.pred_small_token,
            "pred_small_text": comparison_point.pred_small_text,
            # Divergent continuations from ComparisonPoint
            "small_diverge_text": comparison_point.small_diverge_text,
            "reference_diverge_text": comparison_point.reference_diverge_text,
            "common_context": comparison_point.common_context,
            "question": comparison_point.question,
        }

        # Only add verify results if is_verify is True
        if is_verify:
            point_dict["small_correct"] = comparison_point.small_correct
            point_dict["reference_correct"] = comparison_point.reference_correct
            point_dict["verify_response"] = comparison_point.verify_response

        data.append(point_dict)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Define basic columns that are always included

    column_order = [
        # Identifiers
        "data_id",
        "token_id",
        # Original and predictions
        "real_token",
        "real_text",
        "pred_small_token",
        "pred_small_text",
         # Divergent continuations
        "small_diverge_text",
        "reference_diverge_text",
        "common_context",
        "question",
    ]

    # Add verify columns if is_verify is True
    if is_verify:
        column_order.extend(["small_correct", "reference_correct", "verify_response"])

    # Only include columns that exist in the DataFrame
    available_columns = [col for col in column_order if col in df.columns]
    return df[available_columns]
