"""
Model configuration utilities for the self-disclosure detection task.
"""

import logging
from typing import Dict, List, Optional, Union
from src.model import RobertaForSelfDisclosureDetection

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
    
    # Load config with memory optimizations
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        id2label={i: label for i, label in enumerate(label_list)} if label_list else None,
        label2id={label: i for i, label in enumerate(label_list)} if label_list else None,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        gradient_checkpointing=True,  # Enable gradient checkpointing by default
        use_cache=False,  # Disable KV cache during training
    )
    
    # Load tokenizer efficiently
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        use_fast=True,  # Use fast tokenizer
        local_files_only=local_files_only,
        add_prefix_space=True,
        model_max_length=512,  # Set a reasonable max length
    )
    
    # Load model with optimizations - no longer specifying dtype here
    # Let the Trainer handle precision based on fp16/bf16 arguments
    logger.info("Loading model with memory optimizations...")
    model = RobertaForSelfDisclosureDetection.from_pretrained(
        model_name_or_path,
        config=config,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        low_cpu_mem_usage=True,  # Optimize CPU memory usage during loading
    )
    
    # Log model size information
    model_size = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded with {model_size/1e6:.2f}M parameters")
    
    return model, tokenizer

def get_training_args(
    output_dir: str,
    num_train_epochs: int = 5,
    per_device_train_batch_size: int = 24,
    per_device_eval_batch_size: int = 24,
    warmup_steps: int = 100,
    weight_decay: float = 0.01,
    learning_rate: float = 5e-5,
    save_strategy: str = "steps",
    evaluation_strategy: str = "steps",
    load_best_model_at_end: bool = True,
    metric_for_best_model: str = "f1",
    greater_is_better: bool = True,
    fp16: bool = False,
    seed: int = 42,
    gradient_accumulation_steps: int = 4,
    gradient_checkpointing: bool = True,
    bf16: bool = False,
    logging_steps: int = 100,
    eval_steps: int = 500,
    save_steps: int = 500,
) -> Dict:
    """
    Get training arguments for the Trainer with optimized settings.
    
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
        gradient_accumulation_steps: Number of steps for gradient accumulation
        gradient_checkpointing: Whether to use gradient checkpointing
        bf16: Whether to use bfloat16 mixed precision
        logging_steps: How often to log
        eval_steps: How often to evaluate
        save_steps: How often to save
        
    Returns:
        Dictionary of training arguments
    """
    return {
        "output_dir": output_dir,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "warmup_steps": warmup_steps,
        "weight_decay": weight_decay,
        "learning_rate": learning_rate,
        "save_strategy": save_strategy,
        "evaluation_strategy": evaluation_strategy,
        "load_best_model_at_end": load_best_model_at_end,
        "metric_for_best_model": metric_for_best_model,
        "greater_is_better": greater_is_better,
        "fp16": fp16,
        "bf16": bf16,
        "seed": seed,
        "logging_dir": f"{output_dir}/logs",
        "logging_strategy": "steps",
        "logging_steps": logging_steps,
        "save_total_limit": 2,  # Only keep the 2 best checkpoints
        "overwrite_output_dir": True,
        "dataloader_num_workers": 4,  # Reduced from 16 to prevent memory issues
        "group_by_length": True,  # Group similar length sequences for efficiency
        "report_to": "tensorboard",
        "label_names": ["labels"],
        "save_steps": save_steps,
        "eval_steps": eval_steps,
        "gradient_checkpointing": gradient_checkpointing,
        "optim": "adamw_torch",  # Use PyTorch's implementation
        "ddp_find_unused_parameters": False,  # Improve distributed training
        "torch_compile": False,  # Disable compilation to avoid memory spikes
        "auto_find_batch_size": True,  # Automatically find optimal batch size
        "lr_scheduler_type": "cosine",  # Use cosine learning rate schedule
    }
