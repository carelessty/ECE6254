"""
Script to run the full comparative analysis between DeepSeek-R1-Distill-Qwen-1.5B
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
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, set_seed

from src.implementation.data_utils import load_self_disclosure_dataset
from src.implementation.model import PrivacyRiskClassifier, FewShotPrivacyRiskClassifier
from src.evaluation.evaluator import PrivacyRiskEvaluator, load_reference_model
from src.evaluation.config import EXPERIMENTS

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Privacy Risk Detection Comparative Analysis")
    
    parser.add_argument(
        "--experiment",
        type=str,
        default="sentence_classification_few_shot",
        choices=[exp["name"] for exp in EXPERIMENTS],
        help="Name of experiment to run"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save results"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--use_mock_data",
        action="store_true",
        help="Whether to use mock data instead of actual dataset"
    )
    
    return parser.parse_args()

def run_comparative_analysis(args):
    """
    Run the comparative analysis between DeepSeek-R1-Distill-Qwen-1.5B
    and the reference model for privacy risk detection.
    """
    print(f"Running comparative analysis for experiment: {args.experiment}")
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Get experiment configuration
    experiment = next(exp for exp in EXPERIMENTS if exp["name"] == args.experiment)
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, experiment["name"])
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize tokenizer for DeepSeek model
    print(f"Loading tokenizer for {experiment['deepseek_model']}...")
    tokenizer = AutoTokenizer.from_pretrained(experiment['deepseek_model'])
    
    # Ensure the tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    print("Loading dataset...")
    if args.use_mock_data:
        from src.implementation.data_utils import create_mock_dataloaders
        _, _, test_dataloader = create_mock_dataloaders(
            tokenizer, batch_size=experiment['batch_size'], task=experiment['task']
        )
    else:
        _, _, test_dataloader = load_self_disclosure_dataset(
            tokenizer, batch_size=experiment['batch_size'], task=experiment['task']
        )
    
    # Initialize models
    print("Initializing models...")
    
    # Initialize DeepSeek model
    if experiment['approach'] == "fine-tuning":
        print(f"Initializing {experiment['deepseek_model']} for fine-tuning...")
        deepseek_model = PrivacyRiskClassifier(
            model_name=experiment['deepseek_model'],
            task=experiment['task'],
            use_lora=experiment['use_lora']
        )
    else:  # few-shot learning
        print(f"Initializing {experiment['deepseek_model']} for few-shot learning...")
        deepseek_model = FewShotPrivacyRiskClassifier(
            model_name=experiment['deepseek_model'],
            task=experiment['task']
        )
    
    # Initialize reference model
    print(f"Loading reference model: {experiment['reference_model']}...")
    reference_model = load_reference_model(experiment['reference_model'])
    
    # Initialize evaluator
    evaluator = PrivacyRiskEvaluator(output_dir=output_dir)
    
    # Define model configurations for comparison
    model_configs = [
        {
            "name": f"DeepSeek-R1-Distill-Qwen-1.5B ({experiment['approach']})",
            "model": deepseek_model
        },
        {
            "name": "RoBERTa-large-self-disclosure (reference)",
            "model": reference_model
        }
    ]
    
    # Run comparison
    print("Running comparison...")
    comparison = evaluator.compare_models(
        model_configs, test_dataloader, tokenizer, task=experiment['task']
    )
    
    # Generate report
    print("Generating report...")
    report = evaluator.generate_report(comparison, task=experiment['task'])
    
    # Create visualization
    print("Creating visualization...")
    create_visualization(comparison, output_dir)
    
    print(f"Comparative analysis completed. Results saved to {output_dir}")
    
    return comparison

def create_visualization(comparison, output_dir):
    """Create visualization of comparison results."""
    # Create visualizations directory
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Define metrics to plot
    metrics = ["accuracy", "precision", "recall", "f1"]
    
    # Create a figure
    plt.figure(figsize=(10, 6))
    
    # Set up bar positions
    x = np.arange(len(metrics))
    width = 0.35
    
    # Extract values for each model
    model_names = list(comparison.keys())
    model1_values = [comparison[model_names[0]][metric] for metric in metrics]
    model2_values = [comparison[model_names[1]][metric] for metric in metrics]
    
    # Create bars
    plt.bar(x - width/2, model1_values, width, label=model_names[0])
    plt.bar(x + width/2, model2_values, width, label=model_names[1])
    
    # Add labels and title
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.title("Privacy Risk Detection Performance Comparison")
    plt.xticks(x, [m.capitalize() for m in metrics])
    plt.legend()
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for i, v in enumerate(model1_values):
        plt.text(i - width/2, v + 0.02, f"{v:.2f}", ha="center")
    
    for i, v in enumerate(model2_values):
        plt.text(i + width/2, v + 0.02, f"{v:.2f}", ha="center")
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "performance_comparison.png"))
    plt.close()

if __name__ == "__main__":
    args = parse_args()
    run_comparative_analysis(args)
