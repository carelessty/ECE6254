"""
Configuration file for evaluation experiments.
This module defines the experiment configurations for comparing
DeepSeek-R1-Distill-Qwen-1.5B with the reference model.
"""

# Define experiment configurations
EXPERIMENTS = [
    # Experiment 1: Sentence-level classification with fine-tuning
    {
        "name": "sentence_classification_fine_tuning",
        "description": "Sentence-level classification using fine-tuning with LoRA",
        "deepseek_model": "deepseek-ai/deepseek-r1-distill-qwen-1.5b",
        "reference_model": "douy/roberta-large-self-disclosure-sentence-classification",
        "task": "classification",
        "approach": "fine-tuning",
        "use_lora": True,
        "batch_size": 8,
        "output_dir": "results/sentence_classification_fine_tuning"
    },
    
    # Experiment 2: Span-level detection with fine-tuning
    {
        "name": "span_detection_fine_tuning",
        "description": "Span-level detection using fine-tuning with LoRA",
        "deepseek_model": "deepseek-ai/deepseek-r1-distill-qwen-1.5b",
        "reference_model": "douy/roberta-large-self-disclosure-sentence-classification",
        "task": "span",
        "approach": "fine-tuning",
        "use_lora": True,
        "batch_size": 8,
        "output_dir": "results/span_detection_fine_tuning"
    }
]

# Define evaluation metrics
METRICS = [
    {
        "name": "accuracy",
        "description": "Proportion of correct predictions",
        "higher_is_better": True
    },
    {
        "name": "precision",
        "description": "Precision for the positive class (self-disclosure)",
        "higher_is_better": True
    },
    {
        "name": "recall",
        "description": "Recall for the positive class (self-disclosure)",
        "higher_is_better": True
    },
    {
        "name": "f1",
        "description": "F1 score for the positive class (self-disclosure)",
        "higher_is_better": True
    }
]

# Define visualization settings
VISUALIZATION = {
    "colors": {
        "DeepSeek-R1-Distill-Qwen-1.5B": "#1f77b4",  # Blue
        "RoBERTa-large-self-disclosure": "#ff7f0e"   # Orange
    },
    "metrics_to_plot": ["accuracy", "precision", "recall", "f1"],
    "plot_title": "Privacy Risk Detection Performance Comparison",
    "plot_filename": "performance_comparison.png"
}
