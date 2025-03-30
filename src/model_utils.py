"""
Model configuration utilities for the self-disclosure detection task.
"""

import logging
from typing import Dict, List, Optional, Union

import torch
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)

logger = logging.getLogger(__name__)

def get_model_and_tokenizer(
    model_name_or_path: str = "roberta-large",
    num_labels: int = 0,
    label_list: Optional[List[str]] = None,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a pre-trained model and tokenizer for token classification.
    
    Args:
        model_name_or_path: Name or path of the pre-trained model
        num_labels: Number of labels for classification
        label_list: List of labels
        cache_dir: Directory to cache the model
        local_files_only: Whether to use only local files
        
    Returns:
        Tuple of (model, tokenizer)
    """
    if label_list is not None:
        num_labels = len(label_list)
    
    # Load config
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        id2label={i: label for i, label in enumerate(label_list)} if label_list else None,
        label2id={label: i for i, label in enumerate(label_list)} if label_list else None,
        cache_dir=cache_dir,
        local_files_only=local_files_only
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        use_fast=True,
        local_files_only=local_files_only,
        add_prefix_space=True
    )
    
    # Load model
    model = AutoModelForTokenClassification.from_pretrained(
        model_name_or_path,
        config=config,
        cache_dir=cache_dir,
        local_files_only=local_files_only
    )
    
    return model, tokenizer

def get_training_args(
    output_dir: str,
    num_train_epochs: int = 5,
    per_device_train_batch_size: int = 16,
    per_device_eval_batch_size: int = 16,
    warmup_steps: int = 500,
    weight_decay: float = 0.01,
    learning_rate: float = 5e-5,
    save_strategy: str = "epoch",
    evaluation_strategy: str = "epoch",
    load_best_model_at_end: bool = True,
    metric_for_best_model: str = "f1",
    greater_is_better: bool = True,
    fp16: bool = False,
    seed: int = 42
) -> Dict:
    """
    Get training arguments for the Trainer.
    
    Args:
        output_dir: Directory to save the model
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size for training
        per_device_eval_batch_size: Batch size for evaluation
        warmup_steps: Number of warmup steps
        weight_decay: Weight decay
        learning_rate: Learning rate
        save_strategy: When to save the model
        evaluation_strategy: When to evaluate the model
        load_best_model_at_end: Whether to load the best model at the end
        metric_for_best_model: Metric to use for best model
        greater_is_better: Whether higher is better for the metric
        fp16: Whether to use mixed precision
        seed: Random seed
        
    Returns:
        Dictionary of training arguments
    """
    return {
        "output_dir": output_dir,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "warmup_steps": warmup_steps,
        "weight_decay": weight_decay,
        "learning_rate": learning_rate,
        "save_strategy": save_strategy,
        "evaluation_strategy": evaluation_strategy,
        "load_best_model_at_end": load_best_model_at_end,
        "metric_for_best_model": metric_for_best_model,
        "greater_is_better": greater_is_better,
        "fp16": fp16,
        "seed": seed,
        "logging_dir": f"{output_dir}/logs",
        "logging_strategy": "steps",
        "logging_steps": 100,
        "save_total_limit": 2,
        "overwrite_output_dir": True,
        "dataloader_num_workers": 4,
        "group_by_length": True,
        "report_to": "tensorboard"
    }
