"""
Evaluation framework for comparing privacy risk detection models.
This module implements metrics and evaluation procedures for comparing
DeepSeek-R1-Distill-Qwen-1.5B with the reference model.
"""
import os
import sys
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

# Optional: Adjust sys.path if needed (e.g., when running as script)
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.implementation.data_utils import load_self_disclosure_dataset
from src.implementation.model import PrivacyRiskClassifier


class PrivacyRiskEvaluator:
    def __init__(self, output_dir="evaluation_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.metrics = {}

    def evaluate_model(self, model, test_dataloader, tokenizer, task="classification"):
        print(f"Evaluating model on {task} task...")
        texts = []
        labels = []

        for batch in tqdm(test_dataloader, desc="Processing test data"):
            if isinstance(batch, dict):
                input_ids = batch["input_ids"]
                batch_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                label_tensor = batch["labels"]
            else:
                batch_texts = tokenizer.batch_decode(batch[0], skip_special_tokens=True)
                label_tensor = batch[2]

            if torch.is_tensor(label_tensor):
                batch_labels = label_tensor.cpu().tolist()
            else:
                batch_labels = list(label_tensor)

            texts.extend(batch_texts)
            labels.extend(batch_labels)

        predictions = model.predict(texts)

        if task == "classification":
            metrics = self._calculate_classification_metrics(predictions, labels)
        else:
            metrics = self._calculate_span_metrics(predictions, labels)

        return metrics

    def _calculate_classification_metrics(self, predictions, labels):
        metrics = {
            "accuracy": accuracy_score(labels, predictions),
            "precision": precision_score(labels, predictions, average="binary", zero_division=0),
            "recall": recall_score(labels, predictions, average="binary", zero_division=0),
            "f1": f1_score(labels, predictions, average="binary", zero_division=0),
            "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist()
        }
        metrics["classification_report"] = classification_report(labels, predictions, output_dict=True, zero_division=0)
        return metrics

    def _calculate_span_metrics(self, predictions, labels):
        flat_preds = []
        flat_labels = []

        for pred_seq, label_seq in zip(predictions, labels):
            if isinstance(pred_seq, (int, np.integer)):
                flat_preds.append(pred_seq)
                if hasattr(label_seq, '__iter__') and not isinstance(label_seq, str):
                    for l in label_seq:
                        if l != -100:
                            flat_labels.append(l)
                            break
                else:
                    flat_labels.append(label_seq)
            elif isinstance(label_seq, (int, np.integer)) and hasattr(pred_seq, '__iter__'):
                if label_seq != -100:
                    flat_preds.append(pred_seq[0] if len(pred_seq) > 0 else 0)
                    flat_labels.append(label_seq)
            elif hasattr(pred_seq, '__iter__') and hasattr(label_seq, '__iter__'):
                for p, l in zip(pred_seq, label_seq):
                    if l != -100:
                        flat_preds.append(p)
                        flat_labels.append(l)

        metrics = {
            "accuracy": accuracy_score(flat_labels, flat_preds),
            "precision": precision_score(flat_labels, flat_preds, average="binary", zero_division=0),
            "recall": recall_score(flat_labels, flat_preds, average="binary", zero_division=0),
            "f1": f1_score(flat_labels, flat_preds, average="binary", zero_division=0),
            "confusion_matrix": confusion_matrix(flat_labels, flat_preds, labels=[0, 1]).tolist()
        }
        metrics["classification_report"] = classification_report(flat_labels, flat_preds, output_dict=True, zero_division=0)
        return metrics

    def compare_models(self, model_configs, test_dataloader, tokenizer, task="classification"):
        print(f"Comparing {len(model_configs)} models on {task} task...")
        comparison = {}
        for config in model_configs:
            model_name = config["name"]
            model = config["model"]
            print(f"Evaluating {model_name}...")
            metrics = self.evaluate_model(model, test_dataloader, tokenizer, task)
            comparison[model_name] = metrics
        self._save_comparison(comparison, task)
        return comparison

    def _save_comparison(self, comparison, task):
        output_file = os.path.join(self.output_dir, f"{task}_comparison.json")
        with open(output_file, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"Comparison results saved to {output_file}")

    def generate_report(self, comparison, task="classification"):
        report = f"# Privacy Risk Detection Model Comparison\n\n"
        report += f"## Task: {task.capitalize()}\n\n"
        report += "### Main Metrics\n\n"
        report += "| Model | Accuracy | Precision | Recall | F1 Score |\n"
        report += "|-------|----------|-----------|--------|----------|\n"

        for model_name, metrics in comparison.items():
            report += f"| {model_name} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} |\n"

        report += "\n### Detailed Metrics\n\n"

        for model_name, metrics in comparison.items():
            report += f"#### {model_name}\n\n"
            report += "**Confusion Matrix:**\n\n```\n"
            for row in metrics["confusion_matrix"]:
                report += f"{row}\n"
            report += "```\n\n"

            report += "**Classification Report:**\n\n```\n"
            cr = metrics["classification_report"]
            report += f"{'Label':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}\n\n"

            label_keys = [k for k in cr.keys() if k.isdigit()]
            for label in sorted(label_keys, key=int):
                m = cr[label]
                report += f"{label:>10} {m['precision']:10.4f} {m['recall']:10.4f} {m['f1-score']:10.4f} {m['support']:10}\n"

            if 'accuracy' in cr:
                report += f"\n{'Accuracy':>10} {cr['accuracy']:>10.4f}\n"
            for avg_type in ['macro avg', 'weighted avg']:
                if avg_type in cr:
                    m = cr[avg_type]
                    support = m.get('support', 0)
                    report += f"{avg_type:>10} {m['precision']:10.4f} {m['recall']:10.4f} {m['f1-score']:10.4f} {support:10}\n"

            report += "```\n\n"

        output_file = os.path.join(self.output_dir, f"{task}_report.md")
        with open(output_file, "w") as f:
            f.write(report)

        print(f"Report saved to {output_file}")
        return report


def load_reference_model(model_name="douy/roberta-large-self-disclosure-sentence-classification"):
    class ReferenceModelWrapper:
        def __init__(self, model_name):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)

        def predict(self, texts, batch_size=32):
            all_preds = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    all_preds.extend(predictions.cpu().numpy())
            return all_preds

    return ReferenceModelWrapper(model_name)
