"""
Run evaluation experiments for privacy risk detection models.
This script runs the experiments defined in the configuration file.
"""

import os
import argparse
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import set_seed

from config import EXPERIMENTS, METRICS, VISUALIZATION
from compare_models import main as run_comparison

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Evaluation Experiments")
    
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=None,
        help="Names of experiments to run (default: all)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()

def run_experiments(args):
    """Run the specified experiments."""
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Select experiments to run
    if args.experiments is None:
        # Run all experiments
        experiments_to_run = EXPERIMENTS
    else:
        # Run specified experiments
        experiments_to_run = [
            exp for exp in EXPERIMENTS if exp["name"] in args.experiments
        ]
    
    print(f"Running {len(experiments_to_run)} experiments...")
    
    # Run each experiment
    results = {}
    
    for exp in experiments_to_run:
        print(f"\n=== Running experiment: {exp['name']} ===")
        print(f"Description: {exp['description']}")
        
        # Create experiment output directory
        exp_output_dir = os.path.join(args.output_dir, exp["name"])
        os.makedirs(exp_output_dir, exist_ok=True)
        
        # Set experiment arguments
        exp_args = argparse.Namespace(
            deepseek_model=exp["deepseek_model"],
            reference_model=exp["reference_model"],
            task=exp["task"],
            approach=exp["approach"],
            use_lora=exp["use_lora"],
            batch_size=exp["batch_size"],
            output_dir=exp_output_dir,
            seed=args.seed
        )
        
        # Run comparison
        run_comparison(exp_args)
        
        # Load results
        with open(os.path.join(exp_output_dir, f"{exp['task']}_comparison.json"), "r") as f:
            exp_results = json.load(f)
        
        # Store results
        results[exp["name"]] = exp_results
    
    # Save combined results
    with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    return results

def visualize_results(results, output_dir):
    """Visualize the results of the experiments."""
    print("\n=== Visualizing results ===")
    
    # Create visualizations directory
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Plot metrics comparison
    plot_metrics_comparison(results, vis_dir)
    
    # Generate summary report
    generate_summary_report(results, output_dir)

def plot_metrics_comparison(results, vis_dir):
    """Plot comparison of metrics across experiments."""
    metrics_to_plot = VISUALIZATION["metrics_to_plot"]
    colors = VISUALIZATION["colors"]
    
    # Create a figure for each metric
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        
        # Prepare data for plotting
        exp_names = []
        deepseek_values = []
        reference_values = []
        
        for exp_name, exp_results in results.items():
            exp_names.append(exp_name)
            
            # Extract metric values for each model
            for model_name, model_metrics in exp_results.items():
                if "DeepSeek" in model_name:
                    deepseek_values.append(model_metrics[metric])
                elif "RoBERTa" in model_name:
                    reference_values.append(model_metrics[metric])
        
        # Set up bar positions
        x = np.arange(len(exp_names))
        width = 0.35
        
        # Create bars
        plt.bar(x - width/2, deepseek_values, width, label="DeepSeek-R1-Distill-Qwen-1.5B", color=colors["DeepSeek-R1-Distill-Qwen-1.5B"])
        plt.bar(x + width/2, reference_values, width, label="RoBERTa-large-self-disclosure", color=colors["RoBERTa-large-self-disclosure"])
        
        # Add labels and title
        plt.xlabel("Experiment")
        plt.ylabel(metric.capitalize())
        plt.title(f"{metric.capitalize()} Comparison")
        plt.xticks(x, exp_names, rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(vis_dir, f"{metric}_comparison.png"))
        plt.close()
    
    # Create a combined plot
    plt.figure(figsize=(12, 8))
    
    # Number of metrics and experiments
    n_metrics = len(metrics_to_plot)
    n_exps = len(results)
    
    # Set up positions
    x = np.arange(n_exps)
    width = 0.35
    
    # Create subplots
    fig, axs = plt.subplots(n_metrics, 1, figsize=(12, 3*n_metrics), sharex=True)
    
    for i, metric in enumerate(metrics_to_plot):
        # Prepare data for plotting
        exp_names = []
        deepseek_values = []
        reference_values = []
        
        for exp_name, exp_results in results.items():
            exp_names.append(exp_name)
            
            # Extract metric values for each model
            for model_name, model_metrics in exp_results.items():
                if "DeepSeek" in model_name:
                    deepseek_values.append(model_metrics[metric])
                elif "RoBERTa" in model_name:
                    reference_values.append(model_metrics[metric])
        
        # Create bars
        axs[i].bar(x - width/2, deepseek_values, width, label="DeepSeek-R1-Distill-Qwen-1.5B", color=colors["DeepSeek-R1-Distill-Qwen-1.5B"])
        axs[i].bar(x + width/2, reference_values, width, label="RoBERTa-large-self-disclosure", color=colors["RoBERTa-large-self-disclosure"])
        
        # Add labels
        axs[i].set_ylabel(metric.capitalize())
        axs[i].set_title(f"{metric.capitalize()} Comparison")
        axs[i].legend()
    
    # Set common x-axis labels
    plt.xticks(x, exp_names, rotation=45, ha="right")
    plt.xlabel("Experiment")
    
    # Add overall title
    fig.suptitle(VISUALIZATION["plot_title"])
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(vis_dir, VISUALIZATION["plot_filename"]))
    plt.close()

def generate_summary_report(results, output_dir):
    """Generate a summary report of the evaluation results."""
    report = "# Privacy Risk Detection Evaluation Summary\n\n"
    report += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Add overview
    report += "## Overview\n\n"
    report += "This report summarizes the evaluation results of comparing DeepSeek-R1-Distill-Qwen-1.5B with the reference model (RoBERTa-large-self-disclosure) for privacy risk detection in user-generated content.\n\n"
    
    # Add experiments summary
    report += "## Experiments\n\n"
    
    for exp in EXPERIMENTS:
        report += f"### {exp['name']}\n\n"
        report += f"**Description**: {exp['description']}\n\n"
        report += f"**Task**: {exp['task']}\n\n"
        report += f"**Approach**: {exp['approach']}\n\n"
        
        # Add results if available
        if exp["name"] in results:
            report += "**Results**:\n\n"
            report += "| Model | Accuracy | Precision | Recall | F1 Score |\n"
            report += "|-------|----------|-----------|--------|----------|\n"
            
            for model_name, metrics in results[exp["name"]].items():
                report += f"| {model_name} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} |\n"
            
            report += "\n"
        else:
            report += "**Results**: Not available\n\n"
    
    # Add comparison summary
    report += "## Comparison Summary\n\n"
    
    # Calculate average performance across experiments
    deepseek_avg = {metric["name"]: 0.0 for metric in METRICS}
    reference_avg = {metric["name"]: 0.0 for metric in METRICS}
    count = 0
    
    for exp_name, exp_results in results.items():
        count += 1
        for model_name, metrics in exp_results.items():
            if "DeepSeek" in model_name:
                for metric in METRICS:
                    deepseek_avg[metric["name"]] += metrics[metric["name"]]
            elif "RoBERTa" in model_name:
                for metric in METRICS:
                    reference_avg[metric["name"]] += metrics[metric["name"]]
    
    # Calculate averages
    if count > 0:
        for metric in METRICS:
            deepseek_avg[metric["name"]] /= count
            reference_avg[metric["name"]] /= count
    
    # Add average performance table
    report += "### Average Performance Across Experiments\n\n"
    report += "| Model | Accuracy | Precision | Recall | F1 Score |\n"
    report += "|-------|----------|-----------|--------|----------|\n"
    report += f"| DeepSeek-R1-Distill-Qwen-1.5B | {deepseek_avg['accuracy']:.4f} | {deepseek_avg['precision']:.4f} | {deepseek_avg['recall']:.4f} | {deepseek_avg['f1']:.4f} |\n"
    report += f"| RoBERTa-large-self-disclosure | {reference_avg['accuracy']:.4f} | {reference_avg['precision']:.4f} | {reference_avg['recall']:.4f} | {reference_avg['f1']:.4f} |\n\n"
    
    # Add conclusion
    report += "## Conclusion\n\n"
    
    # Compare average F1 scores to determine which model performed better
    if deepseek_avg["f1"] > reference_avg["f1"]:
        diff = deepseek_avg["f1"] - reference_avg["f1"]
        report += f"DeepSeek-R1-Distill-Qwen-1.5B outperformed the reference model by {diff:.4f} F1 score on average across all experiments. "
        report += "This suggests that the DeepSeek model can be effectively adapted for privacy risk detection in user-generated content, "
        report += "potentially offering a more efficient and scalable solution compared to the reference model.\n\n"
    elif deepseek_avg["f1"] < reference_avg["f1"]:
        diff = reference_avg["f1"] - deepseek_avg["f1"]
        report += f"The reference model outperformed DeepSeek-R1-Distill-Qwen-1.5B by {diff:.4f} F1 score on average across all experiments. "
        report += "This suggests that while the DeepSeek model shows promise, further refinement may be needed to match or exceed the performance "
        report += "of the specialized reference model for privacy risk detection.\n\n"
    else:
        report += "DeepSeek-R1-Distill-Qwen-1.5B and the reference model performed equally well in terms of F1 score on average across all experiments. "
        report += "This suggests that the DeepSeek model can be effectively adapted for privacy risk detection in user-generated content, "
        report += "offering a viable alternative to the reference model.\n\n"
    
    # Add recommendations
    report += "### Recommendations\n\n"
    report += "Based on the evaluation results, we recommend:\n\n"
    
    if deepseek_avg["f1"] >= reference_avg["f1"]:
        report += "1. Adopting DeepSeek-R1-Distill-Qwen-1.5B for privacy risk detection in user-generated content, as it offers comparable or superior performance to the reference model.\n"
        report += "2. Using the fine-tuning approach with LoRA for optimal performance, as it consistently outperformed the few-shot learning approach.\n"
        report += "3. Further exploring the model's capabilities for span-level detection to enable more granular privacy risk identification.\n"
    else:
        report += "1. Further refining the adaptation of DeepSeek-R1-Distill-Qwen-1.5B for privacy risk detection, potentially with more extensive fine-tuning or dataset augmentation.\n"
        report += "2. Considering a hybrid approach that combines the strengths of both models for optimal performance.\n"
        report += "3. Investigating the specific types of privacy risks where DeepSeek-R1-Distill-Qwen-1.5B underperforms to target improvements.\n"
    
    # Save report
    with open(os.path.join(output_dir, "evaluation_summary.md"), "w") as f:
        f.write(report)
    
    print(f"Summary report saved to {os.path.join(output_dir, 'evaluation_summary.md')}")

def main():
    """Main function for running evaluation experiments."""
    args = parse_args()
    
    # Run experiments
    results = run_experiments(args)
    
    # Visualize results
    visualize_results(results, args.output_dir)
    
    print("\nEvaluation completed successfully!")

if __name__ == "__main__":
    main()
