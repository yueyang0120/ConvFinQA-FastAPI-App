#!/usr/bin/env python3
"""
Evaluation script for Financial QA System.

This script evaluates the performance of the Financial QA workflow on real examples
from the ConvFinQA dataset, using both exact match and numeric tolerance metrics.
"""

import os
import re
import json
import argparse
import pandas as pd
import numpy as np
import sys
from typing import Dict, List, Any, Tuple, Optional
from tqdm import tqdm

# Add the project root to the Python path to resolve imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the workflow
from src.core.workflow import process_financial_question

# Constants
ABSOLUTE_EPSILON = 0.05  # For small values (absolute difference)
RELATIVE_EPSILON = 0.01  # For larger values (1% relative difference)


def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """
    Load dataset from a JSON file.
    
    Args:
        file_path: Path to the dataset file
        
    Returns:
        List of examples
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def extract_qa_pairs(example: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Extract question-answer pairs from an example.
    
    Args:
        example: The example containing QA data in various formats
        
    Returns:
        List of dictionaries, each containing 'question' and 'answer' keys
    """
    qa_pairs = []
    
    # Check for a single 'qa' field
    if 'qa' in example and isinstance(example['qa'], dict):
        qa_data = example['qa']
        if 'question' in qa_data and 'answer' in qa_data:
            qa_pairs.append({
                'question': qa_data['question'],
                'answer': qa_data['answer']
            })
    
    # Also look for qa_0, qa_1, etc. fields
    for key in example:
        if key.startswith('qa_') and key[3:].isdigit():
            qa_data = example[key]
            if isinstance(qa_data, dict) and 'question' in qa_data and 'answer' in qa_data:
                qa_pairs.append({
                    'question': qa_data['question'],
                    'answer': qa_data['answer']
                })
    
    return qa_pairs


def normalize_answer(answer: str) -> str:
    """
    Normalize the answer for comparison.
    
    Args:
        answer: The answer string
        
    Returns:
        Normalized answer string
    """
    # Handle None values
    if answer is None:
        return ""
        
    # Convert to lowercase
    answer = answer.lower()
    
    # Remove punctuation except decimal points in numbers
    answer = re.sub(r'[^\w\s\.\-\$%]', '', answer)
    
    # Normalize spaces
    answer = re.sub(r'\s+', ' ', answer).strip()
    
    # Normalize percentage format (e.g., "10%" -> "10 %")
    answer = re.sub(r'(\d+)%', r'\1 %', answer)
    
    # Normalize dollar format (e.g., "$10" -> "$ 10")
    answer = re.sub(r'\$(\d+)', r'$ \1', answer)
    
    # Normalize million/billion format
    answer = answer.replace('million', 'm').replace('m ', 'm')
    answer = answer.replace('billion', 'b').replace('b ', 'b')
    
    return answer


def extract_numeric_value(answer: str) -> Optional[float]:
    """
    Extract numeric value from an answer string.
    
    Args:
        answer: The answer string
        
    Returns:
        Extracted numeric value or None if no numeric value is found
    """
    # Check if answer is None
    if answer is None:
        return None
        
    # Extract numeric part
    numeric_pattern = r'(-?\d+(?:\.\d+)?)'
    match = re.search(numeric_pattern, answer)
    
    if not match:
        return None
    
    value = float(match.group(1))
    
    # Apply scale factor if the answer contains million/billion
    if 'million' in answer.lower() or ' m' in answer.lower() or answer.lower().endswith('m'):
        value *= 1_000_000
    elif 'billion' in answer.lower() or ' b' in answer.lower() or answer.lower().endswith('b'):
        value *= 1_000_000_000
    
    # Check for percentage
    if '%' in answer:
        value /= 100  # Convert percentage to decimal
    
    return value


def get_decimal_precision(num: float) -> int:
    """
    Get the number of decimal places in a float value.
    
    Args:
        num: The number to check
        
    Returns:
        Number of decimal places
    """
    str_num = str(num)
    if '.' not in str_num:
        return 0
    return len(str_num) - str_num.index('.') - 1


def minimum_precision_match(pred_num: Optional[float], exp_num: Optional[float]) -> bool:
    """
    Match numeric values using the lower precision level of the two numbers.
    
    This approach rounds both numbers to the precision of the less detailed number
    before comparison, preventing false mismatches due to trailing digits.
    
    Args:
        pred_num: Predicted numeric value
        exp_num: Expected numeric value
        
    Returns:
        True if values match after rounding to common precision, False otherwise
    """
    # Handle None values
    if pred_num is None or exp_num is None:
        return False
    
    # Get decimal precision of each number
    pred_precision = get_decimal_precision(pred_num)
    exp_precision = get_decimal_precision(exp_num)
    
    # Use the lower precision for rounding
    target_precision = min(pred_precision, exp_precision)
    
    # Round both numbers to the target precision
    rounded_pred = round(pred_num, target_precision)
    rounded_exp = round(exp_num, target_precision)
    
    return rounded_pred == rounded_exp


def scale_aware_match(pred_num: Optional[float], exp_num: Optional[float]) -> bool:
    """
    Match numeric values with tolerance levels appropriate to their scale.
    
    Uses different comparison strategies based on the magnitude of values:
    - Small values: Fixed absolute tolerance
    - Large values: Percentage-based tolerance
    - Medium values: Balanced approach that scales with the value
    
    Args:
        pred_num: Predicted numeric value
        exp_num: Expected numeric value
        
    Returns:
        True if values match within appropriate tolerance, False otherwise
    """
    # Handle None values
    if pred_num is None or exp_num is None:
        return False
    
    # Calculate absolute difference
    abs_diff = abs(pred_num - exp_num)
    
    # For very small values, use pure absolute tolerance
    if abs(exp_num) < 1.0:
        return abs_diff < ABSOLUTE_EPSILON
    
    # For large values, use pure relative tolerance
    if abs(exp_num) >= 100.0:
        relative_diff = abs_diff / max(abs(exp_num), abs(pred_num))
        return relative_diff < RELATIVE_EPSILON
    
    # For medium values, use a sliding scale (blend of absolute and relative)
    # As values get larger, we rely more on relative difference
    weight = (abs(exp_num) - 1.0) / 99.0  # 0.0 at value=1, 1.0 at value=100
    
    # Calculate weighted tolerance threshold
    threshold = ABSOLUTE_EPSILON * (1.0 - weight) + (RELATIVE_EPSILON * abs(exp_num)) * weight
    
    return abs_diff < threshold


def calculate_metrics(predicted: str, expected: str) -> Dict[str, bool]:
    """
    Calculate evaluation metrics.
    
    Args:
        predicted: Predicted answer
        expected: Expected answer
        
    Returns:
        Dictionary of metrics
    """
    # Normalize answers
    norm_pred = normalize_answer(predicted)
    norm_exp = normalize_answer(expected)
    
    # Exact match after normalization
    exact_match = norm_pred == norm_exp
    
    # Extract numeric values for tolerance matching
    pred_num = extract_numeric_value(predicted)
    exp_num = extract_numeric_value(expected)
    
    # Apply different matching strategies
    precision_match = minimum_precision_match(pred_num, exp_num)
    tolerance_match = scale_aware_match(pred_num, exp_num)
    
    return {
        "exact_match": exact_match,
        "precision_match": precision_match,
        "tolerance_match": tolerance_match
    }


def process_example(example: Dict[str, Any], qa_pair: Dict[str, str]) -> Dict[str, Any]:
    """
    Process a single example through the workflow.
    
    Args:
        example: The example containing context, table, etc.
        qa_pair: The question-answer pair to evaluate
        
    Returns:
        Results dictionary
    """
    # Extract data from example
    question = qa_pair.get('question', '')
    expected_answer = qa_pair.get('answer', '')
    
    # Handle pre_text and post_text as arrays
    pre_text_array = example.get('pre_text', [])
    post_text_array = example.get('post_text', [])
    
    # Join arrays into strings
    pre_context = '\n'.join(pre_text_array) if pre_text_array else ""
    post_context = '\n'.join(post_text_array) if post_text_array else ""
    
    # Get table content - keep as array of arrays
    table_array = example.get('table', [])
    
    # Process the question
    try:
        result = process_financial_question(
            question=question,
            table_data=table_array,  # Pass the table directly as array of arrays
            pre_context=pre_context,
            post_context=post_context
        )
        
        # Calculate metrics
        predicted_answer = result.get('answer', '')
        metrics = calculate_metrics(predicted_answer, expected_answer)
        
        # Return results
        return {
            "example_id": example.get('id', ''),
            "question": question,
            "expected_answer": expected_answer,
            "predicted_answer": predicted_answer,
            "exact_match": metrics["exact_match"],
            "precision_match": metrics["precision_match"],
            "tolerance_match": metrics["tolerance_match"],
            "steps": result.get('steps', []),
            "variables": result.get('variables', {}),
            "error": None
        }
    except Exception as e:
        return {
            "example_id": example.get('id', ''),
            "question": question,
            "expected_answer": expected_answer,
            "predicted_answer": "",
            "exact_match": False,
            "precision_match": False,
            "tolerance_match": False,
            "steps": [],
            "variables": {},
            "error": str(e)
        }


def evaluate_dataset(dataset: List[Dict[str, Any]], num_examples: int = None) -> Dict[str, Any]:
    """
    Evaluate the workflow on a dataset.
    
    Args:
        dataset: The dataset to evaluate
        num_examples: Number of examples to evaluate, or None for all
        
    Returns:
        Evaluation results
    """
    # Create a list of all (example, qa_pair) combinations
    evaluation_pairs = []
    for example in dataset:
        qa_pairs = extract_qa_pairs(example)
        if qa_pairs:
            for qa_pair in qa_pairs:
                evaluation_pairs.append((example, qa_pair))
    
    # Limit the number of examples if specified
    if num_examples is not None and num_examples > 0:
        evaluation_pairs = evaluation_pairs[:num_examples]
    
    results = []
    exact_matches = 0
    precision_matches = 0
    tolerance_matches = 0
    errors = 0
    
    # Process each example and QA pair
    print(f"Evaluating {len(evaluation_pairs)} question-answer pairs...")
    for example, qa_pair in tqdm(evaluation_pairs):
        result = process_example(example, qa_pair)
        results.append(result)
        
        if result['exact_match']:
            exact_matches += 1
        if result['precision_match']:
            precision_matches += 1
        if result['tolerance_match']:
            tolerance_matches += 1
        if result['error']:
            errors += 1
    
    # Calculate metrics
    total = len(results)
    exact_match_accuracy = exact_matches / total if total > 0 else 0
    precision_match_accuracy = precision_matches / total if total > 0 else 0
    tolerance_match_accuracy = tolerance_matches / total if total > 0 else 0
    error_rate = errors / total if total > 0 else 0
    
    return {
        "results": results,
        "metrics": {
            "total_examples": total,
            "exact_match": exact_matches,
            "precision_match": precision_matches,
            "tolerance_match": tolerance_matches,
            "errors": errors,
            "exact_match_accuracy": exact_match_accuracy,
            "precision_match_accuracy": precision_match_accuracy,
            "tolerance_match_accuracy": tolerance_match_accuracy,
            "error_rate": error_rate
        }
    }


def print_evaluation_summary(eval_results: Dict[str, Any]) -> None:
    """
    Print a summary of evaluation results.
    
    Args:
        eval_results: Evaluation results
    """
    metrics = eval_results["metrics"]
    
    print("\n=== Evaluation Summary ===")
    print(f"Total examples: {metrics['total_examples']}")
    print(f"Exact matches: {metrics['exact_match']} ({metrics['exact_match_accuracy']:.2%})")
    print(f"Precision-based matches: {metrics['precision_match']} ({metrics['precision_match_accuracy']:.2%})")
    print(f"Tolerance-based matches: {metrics['tolerance_match']} ({metrics['tolerance_match_accuracy']:.2%})")
    print(f"Errors: {metrics['errors']} ({metrics['error_rate']:.2%})")
    print("==========================\n")


def save_results(eval_results: Dict[str, Any], output_file: str) -> None:
    """
    Save evaluation results to a file.
    
    Args:
        eval_results: Evaluation results
        output_file: Output file path
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"Results saved to {output_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Evaluate Financial QA on the ConvFinQA dataset')
    parser.add_argument('--dataset', default='data/ConvFinQA/data/train.json',
                      help='Path to the dataset file')
    parser.add_argument('--num-examples', type=int, default=10,
                      help='Number of examples to evaluate (0 for all)')
    parser.add_argument('--output', default='evaluation_results.json',
                      help='Output file for evaluation results')
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    dataset = load_dataset(args.dataset)
    
    # Count examples with QA pairs
    examples_with_qa = 0
    total_qa_pairs = 0
    for example in dataset:
        qa_pairs = extract_qa_pairs(example)
        if qa_pairs:
            examples_with_qa += 1
            total_qa_pairs += len(qa_pairs)
    
    print(f"Loaded {len(dataset)} examples.")
    print(f"Found {examples_with_qa} examples with QA pairs, containing {total_qa_pairs} total questions.")
    
    # Evaluate
    num_examples = None if args.num_examples == 0 else args.num_examples
    eval_results = evaluate_dataset(dataset, num_examples)
    
    # Print summary
    print_evaluation_summary(eval_results)
    
    # Save results
    save_results(eval_results, args.output)


if __name__ == '__main__':
    main() 