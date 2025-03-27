"""
Script for running comparative analysis between DeepSeek-R1-Distill-Qwen-1.5B
and the reference model for privacy risk detection.
"""
import sys
import os

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
sys.path.insert(0, project_root)


import os
import argparse
import torch
from transformers import AutoTokenizer, set_seed

from src.implementation.data_utils import load_self_disclosure_dataset
from src.implementation.model import PrivacyRiskClassifier, FewShotPrivacyRiskClassifier
from evaluator import PrivacyRiskEvaluator, load_reference_model

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Privacy Risk Detection Comparative Analysis")
    
    # Model arguments
    parser.add_argument(
        "--deepseek_model",
        type=str,
        default="deepseek-ai/deepseek-r1-distill-qwen-1.5b",
        help="Name or path of the DeepSeek model"
    )
    parser.add_argument(
        "--reference_model",
        type=str,
        default="douy/roberta-large-self-disclosure-sentence-classification",
        help="Name or path of the reference model for comparison"
    )
    
    # Task arguments
    parser.add_argument(
        "--task",
        type=str,
        choices=["classification", "span"],
        default="classification",
        help="Task type: sentence-level classification or span-level detection"
    )
    
    # Approach arguments
    parser.add_argument(
        "--approach",
        type=str,
        choices=["fine-tuning", "few-shot"],
        default="few-shot",
        help="Approach to use for DeepSeek model: fine-tuning or few-shot learning"
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Whether to use LoRA for parameter-efficient fine-tuning"
    )
    
    # Data arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="comparison_results",
        help="Directory to save comparison results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()

def main():
    """Main function for running comparative analysis."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize tokenizer for DeepSeek model
    print(f"Loading tokenizer for {args.deepseek_model}...")
    deepseek_tokenizer = AutoTokenizer.from_pretrained(args.deepseek_model)
    
    # Ensure the tokenizer has padding token
    if deepseek_tokenizer.pad_token is None:
        deepseek_tokenizer.pad_token = deepseek_tokenizer.eos_token
    
    # Load dataset
    print("Loading dataset...")
    _, _, test_dataloader = load_self_disclosure_dataset(
        deepseek_tokenizer, batch_size=args.batch_size, task=args.task
    )
    
    # Initialize models
    print("Initializing models...")
    
    # Initialize DeepSeek model
    if args.approach == "fine-tuning":
        deepseek_model = PrivacyRiskClassifier(
            model_name=args.deepseek_model,
            task=args.task,
            use_lora=args.use_lora
        )
    else:  # few-shot learning
        deepseek_model = FewShotPrivacyRiskClassifier(
            model_name=args.deepseek_model,
            task=args.task
        )
    
    # Initialize reference model
    reference_model = load_reference_model(args.reference_model)
    
    # Initialize evaluator
    evaluator = PrivacyRiskEvaluator(output_dir=args.output_dir)
    
    # Define model configurations for comparison
    model_configs = [
        {
            "name": f"DeepSeek-R1-Distill-Qwen-1.5B ({args.approach})",
            "model": deepseek_model
        },
        {
            "name": "RoBERTa-large-self-disclosure (reference)",
            "model": reference_model
        }
    ]
    
    # Run comparison
    comparison = evaluator.compare_models(
        model_configs, test_dataloader, deepseek_tokenizer, task=args.task
    )
    
    # Generate report
    report = evaluator.generate_report(comparison, task=args.task)
    
    print(f"Comparison completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
