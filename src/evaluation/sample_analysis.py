"""
Script to run a sample comparative analysis between DeepSeek-R1-Distill-Qwen-1.5B
and the reference model using mock data for demonstration purposes.
"""

import sys
import os

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
sys.path.insert(0, project_root)


import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, set_seed
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.implementation.data_utils import create_mock_dataloaders
from src.implementation.model import FewShotPrivacyRiskClassifier
from evaluator import load_reference_model, PrivacyRiskEvaluator

def run_sample_analysis():
    """
    Run a sample comparative analysis using mock data.
    This function demonstrates the comparative analysis process
    without requiring access to the actual models or dataset.
    """
    print("Running sample comparative analysis...")
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Create output directory
    output_dir = "./privacy_risk_detection/results/sample_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create mock tokenizer and data
    print("Creating mock data...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    _, _, test_dataloader = create_mock_dataloaders(tokenizer, batch_size=4, task="classification")
    
    # Create mock models
    print("Creating mock models...")
    
    # Mock DeepSeek model predictions (simulating 75% accuracy)
    class MockDeepSeekModel:
        def predict(self, texts):
            # Simulate predictions with 75% accuracy
            return [1 if "23-year-old" in text or "husband" in text or "16F" in text or "sexual" in text else 0 for text in texts]
    
    # Mock reference model predictions (simulating 85% accuracy)
    class MockReferenceModel:
        def predict(self, texts):
            # Simulate predictions with 85% accuracy
            return [1 if "23-year-old" in text or "husband" in text or "16F" in text or "sexual" in text or "Mexico" in text else 0 for text in texts]
    
    # Initialize evaluator
    evaluator = PrivacyRiskEvaluator(output_dir=output_dir)
    
    # Define model configurations for comparison
    model_configs = [
        {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B (few-shot)",
            "model": MockDeepSeekModel()
        },
        {
            "name": "RoBERTa-large-self-disclosure (reference)",
            "model": MockReferenceModel()
        }
    ]
    
    # Extract texts and labels from test dataloader
    texts = []
    labels = []
    
    for batch in test_dataloader:
        batch_texts = tokenizer.batch_decode(batch[0], skip_special_tokens=True)
        batch_labels = batch[2].tolist()
        texts.extend(batch_texts)
        labels.extend(batch_labels)
    
    # Run comparison manually for demonstration
    comparison = {}
    
    for config in model_configs:
        model_name = config["name"]
        model = config["model"]
        
        print(f"Evaluating {model_name}...")
        predictions = model.predict(texts)
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(labels, predictions),
            "precision": precision_score(labels, predictions, average="binary", zero_division=0),
            "recall": recall_score(labels, predictions, average="binary", zero_division=0),
            "f1": f1_score(labels, predictions, average="binary", zero_division=0)
        }
        
        comparison[model_name] = metrics
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
    
    # Save comparison results
    with open(os.path.join(output_dir, "comparison_results.json"), "w") as f:
        json.dump(comparison, f, indent=2)
    
    # Create visualization
    create_visualization(comparison, output_dir)
    
    # Generate report
    generate_report(comparison, output_dir)
    
    print(f"Sample analysis completed. Results saved to {output_dir}")
    
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

def generate_report(comparison, output_dir):
    """Generate a report of the comparison results."""
    report = "# Privacy Risk Detection Model Comparison\n\n"
    report += "## Overview\n\n"
    report += "This report presents a comparison between DeepSeek-R1-Distill-Qwen-1.5B and the reference model (RoBERTa-large-self-disclosure) for privacy risk detection in user-generated content.\n\n"
    
    report += "## Results\n\n"
    report += "### Performance Metrics\n\n"
    report += "| Model | Accuracy | Precision | Recall | F1 Score |\n"
    report += "|-------|----------|-----------|--------|----------|\n"
    
    for model_name, metrics in comparison.items():
        report += f"| {model_name} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} |\n"
    
    report += "\n"
    
    # Add analysis
    report += "## Analysis\n\n"
    
    # Compare F1 scores
    model_names = list(comparison.keys())
    model1_f1 = comparison[model_names[0]]["f1"]
    model2_f1 = comparison[model_names[1]]["f1"]
    
    if model1_f1 > model2_f1:
        diff = model1_f1 - model2_f1
        report += f"DeepSeek-R1-Distill-Qwen-1.5B outperforms the reference model by {diff:.4f} F1 score. "
        report += "This suggests that the DeepSeek model can be effectively adapted for privacy risk detection in user-generated content, "
        report += "potentially offering a more efficient and scalable solution compared to the reference model.\n\n"
    elif model1_f1 < model2_f1:
        diff = model2_f1 - model1_f1
        report += f"The reference model outperforms DeepSeek-R1-Distill-Qwen-1.5B by {diff:.4f} F1 score. "
        report += "This suggests that while the DeepSeek model shows promise, further refinement may be needed to match or exceed the performance "
        report += "of the specialized reference model for privacy risk detection.\n\n"
    else:
        report += "DeepSeek-R1-Distill-Qwen-1.5B and the reference model perform equally well in terms of F1 score. "
        report += "This suggests that the DeepSeek model can be effectively adapted for privacy risk detection in user-generated content, "
        report += "offering a viable alternative to the reference model.\n\n"
    
    # Add conclusion
    report += "## Conclusion\n\n"
    report += "Based on this sample analysis, we can draw the following conclusions:\n\n"
    
    if model1_f1 >= model2_f1:
        report += "1. DeepSeek-R1-Distill-Qwen-1.5B shows strong potential for privacy risk detection in user-generated content.\n"
        report += "2. The model can be effectively adapted using few-shot learning or fine-tuning approaches.\n"
        report += "3. Further optimization could potentially improve performance even further.\n\n"
    else:
        report += "1. While DeepSeek-R1-Distill-Qwen-1.5B shows promise, it currently underperforms compared to the reference model.\n"
        report += "2. Further refinement through more extensive fine-tuning or dataset augmentation may be needed.\n"
        report += "3. Exploring alternative adaptation approaches could help bridge the performance gap.\n\n"
    
    report += "**Note**: This is a sample analysis using mock data for demonstration purposes. Actual performance may vary when using the real models and dataset.\n"
    
    # Save report
    with open(os.path.join(output_dir, "comparison_report.md"), "w") as f:
        f.write(report)

if __name__ == "__main__":
    run_sample_analysis()
