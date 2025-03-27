"""
Evaluation framework for comparing privacy risk detection models.
This module implements metrics and evaluation procedures for comparing
DeepSeek-R1-Distill-Qwen-1.5B with the reference model.
"""
import sys
import os

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
sys.path.insert(0, project_root)

import os
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

from src.implementation.data_utils import load_self_disclosure_dataset
from src.implementation.model import PrivacyRiskClassifier, FewShotPrivacyRiskClassifier

class PrivacyRiskEvaluator:
    """
    Evaluator class for privacy risk detection models.
    
    This class handles the evaluation of models on privacy risk detection tasks,
    including comparison between DeepSeek-R1-Distill-Qwen-1.5B and the reference model.
    """
    
    def __init__(self, output_dir="evaluation_results"):
        """
        Initialize the evaluator.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize metrics
        self.metrics = {}
    
    def evaluate_model(self, model, test_dataloader, tokenizer, task="classification"):
        """
        Evaluate a single model on the test dataset.
        
        Args:
            model: Model to evaluate
            test_dataloader: DataLoader for test data
            tokenizer: Tokenizer for decoding inputs
            task: Either "classification" for sentence-level or "span" for span-level detection
        
        Returns:
            Dictionary of evaluation metrics
        """
        print(f"Evaluating model on {task} task...")
        
        # Extract texts and labels from test dataloader
        texts = []
        labels = []
        
        for batch in tqdm(test_dataloader, desc="Processing test data"):
            if isinstance(batch, dict):
                # Handle dataset format from Hugging Face
                batch_texts = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
                batch_labels = batch["labels"].tolist()
            else:
                # Handle TensorDataset format
                batch_texts = tokenizer.batch_decode(batch[0], skip_special_tokens=True)
                batch_labels = batch[2].tolist()
            
            texts.extend(batch_texts)
            labels.extend(batch_labels)
        
        # Make predictions
        predictions = model.predict(texts)
        
        # Calculate metrics
        if task == "classification":
            # For sentence-level classification
            metrics = self._calculate_classification_metrics(predictions, labels)
        else:
            # For span-level detection
            metrics = self._calculate_span_metrics(predictions, labels)
        
        return metrics
    
    def _calculate_classification_metrics(self, predictions, labels):
        """Calculate metrics for sentence-level classification."""
        metrics = {
            "accuracy": accuracy_score(labels, predictions),
            "precision": precision_score(labels, predictions, average="binary"),
            "recall": recall_score(labels, predictions, average="binary"),
            "f1": f1_score(labels, predictions, average="binary"),
            "confusion_matrix": confusion_matrix(labels, predictions).tolist()
        }
        
        # Add classification report
        report = classification_report(labels, predictions, output_dict=True)
        metrics["classification_report"] = report
        
        return metrics
    
    def _calculate_span_metrics(self, predictions, labels):
        """Calculate metrics for span-level detection."""
        # Flatten predictions and labels, ignoring padding (-100)
        flat_preds = []
        flat_labels = []
        
        for pred_seq, label_seq in zip(predictions, labels):
            # Handle case where pred_seq is a single integer (numpy.int64)
            if isinstance(pred_seq, (int, np.int64, np.int32)):
                flat_preds.append(pred_seq)
                if not isinstance(label_seq, (int, np.int64, np.int32)) and hasattr(label_seq, '__iter__'):
                    # If label_seq is iterable but pred_seq is not, use the first non-padding label
                    for l in label_seq:
                        if l != -100:
                            flat_labels.append(l)
                            break
                else:
                    # Both are single values
                    flat_labels.append(label_seq)
            # Handle case where pred_seq is iterable but label_seq is not
            elif isinstance(label_seq, (int, np.int64, np.int32)) and hasattr(pred_seq, '__iter__'):
                if label_seq != -100:  # Ignore padding
                    # Use the first prediction for this single label
                    flat_preds.append(pred_seq[0] if len(pred_seq) > 0 else 0)
                    flat_labels.append(label_seq)
            # Normal case: both are iterables
            elif hasattr(pred_seq, '__iter__') and hasattr(label_seq, '__iter__'):
                for p, l in zip(pred_seq, label_seq):
                    if l != -100:  # Ignore padding
                        flat_preds.append(p)
                        flat_labels.append(l)
        
        metrics = {
            "accuracy": accuracy_score(flat_labels, flat_preds),
            "precision": precision_score(flat_labels, flat_preds, average="binary"),
            "recall": recall_score(flat_labels, flat_preds, average="binary"),
            "f1": f1_score(flat_labels, flat_preds, average="binary"),
            "confusion_matrix": confusion_matrix(flat_labels, flat_preds).tolist()
        }
        
        # Add classification report
        report = classification_report(flat_labels, flat_preds, output_dict=True)
        metrics["classification_report"] = report
        
        return metrics
    
    def compare_models(self, model_configs, test_dataloader, tokenizer, task="classification"):
        """
        Compare multiple models on the test dataset.
        
        Args:
            model_configs: List of dictionaries with model configurations
                Each dictionary should have:
                - "name": Model name for display
                - "model": Initialized model object
            test_dataloader: DataLoader for test data
            tokenizer: Tokenizer for decoding inputs
            task: Either "classification" for sentence-level or "span" for span-level detection
        
        Returns:
            Dictionary of comparison results
        """
        print(f"Comparing {len(model_configs)} models on {task} task...")
        
        comparison = {}
        
        for config in model_configs:
            model_name = config["name"]
            model = config["model"]
            
            print(f"Evaluating {model_name}...")
            metrics = self.evaluate_model(model, test_dataloader, tokenizer, task)
            comparison[model_name] = metrics
        
        # Save comparison results
        self._save_comparison(comparison, task)
        
        return comparison
    
    def _save_comparison(self, comparison, task):
        """Save comparison results to file."""
        output_file = os.path.join(self.output_dir, f"{task}_comparison.json")
        
        with open(output_file, "w") as f:
            json.dump(comparison, f, indent=2)
        
        print(f"Comparison results saved to {output_file}")
    
    def generate_report(self, comparison, task="classification"):
        """
        Generate a human-readable report from comparison results.
        
        Args:
            comparison: Dictionary of comparison results
            task: Either "classification" for sentence-level or "span" for span-level detection
        
        Returns:
            Report as a string
        """
        report = f"# Privacy Risk Detection Model Comparison\n\n"
        report += f"## Task: {task.capitalize()}\n\n"
        
        # Add table of main metrics
        report += "### Main Metrics\n\n"
        report += "| Model | Accuracy | Precision | Recall | F1 Score |\n"
        report += "|-------|----------|-----------|--------|----------|\n"
        
        for model_name, metrics in comparison.items():
            report += f"| {model_name} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} |\n"
        
        report += "\n"
        
        # Add detailed metrics for each model
        report += "### Detailed Metrics\n\n"
        
        for model_name, metrics in comparison.items():
            report += f"#### {model_name}\n\n"
            
            # Confusion Matrix
            report += "**Confusion Matrix:**\n\n"
            cm = metrics["confusion_matrix"]
            report += "```\n"
            report += f"[[{cm[0][0]}, {cm[0][1]}],\n"
            report += f" [{cm[1][0]}, {cm[1][1]}]]\n"
            report += "```\n\n"
            
            # Classification Report
            report += "**Classification Report:**\n\n"
            report += "```\n"
            cr = metrics["classification_report"]
            
            # Format classification report
            report += f"              precision    recall  f1-score   support\n\n"
            
            for label in ["0", "1"]:
                if label in cr:
                    label_metrics = cr[label]
                    report += f"           {label}   {label_metrics['precision']:.4f}    {label_metrics['recall']:.4f}    {label_metrics['f1-score']:.4f}   {label_metrics['support']}\n"
            
            report += f"\n    accuracy                          {cr['accuracy']:.4f}   {cr['macro avg']['support']}\n"
            report += f"   macro avg   {cr['macro avg']['precision']:.4f}    {cr['macro avg']['recall']:.4f}    {cr['macro avg']['f1-score']:.4f}   {cr['macro avg']['support']}\n"
            report += f"weighted avg   {cr['weighted avg']['precision']:.4f}    {cr['weighted avg']['recall']:.4f}    {cr['weighted avg']['f1-score']:.4f}   {cr['weighted avg']['support']}\n"
            
            report += "```\n\n"
        
        # Save report
        output_file = os.path.join(self.output_dir, f"{task}_report.md")
        
        with open(output_file, "w") as f:
            f.write(report)
        
        print(f"Report saved to {output_file}")
        
        return report


def load_reference_model(model_name="douy/roberta-large-self-disclosure-sentence-classification"):
    """
    Load the reference model for comparison.
    
    Args:
        model_name: Name or path of the reference model
    
    Returns:
        Initialized model
    """
    # Create a wrapper class to match the interface of our models
    class ReferenceModelWrapper:
        def __init__(self, model_name):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
        
        def predict(self, texts):
            # Tokenize inputs
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)
            
            return predictions.cpu().numpy()
    
    return ReferenceModelWrapper(model_name)
