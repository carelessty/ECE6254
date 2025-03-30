"""
Training script for fine-tuning RoBERTa-large on the Reddit self-disclosure dataset.
"""

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional
from peft import get_peft_model, LoraConfig, TaskType

import numpy as np
import torch
from datasets import load_dataset, DatasetDict, load_from_disk
from transformers import (
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.data_utils import (
    load_reddit_self_disclosure_dataset,
    prepare_dataset_for_training,
    get_labels,
    compute_metrics,
    compute_partial_span_f1,
    split_dataset
)
from src.model_utils import get_model_and_tokenizer, get_training_args

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa-large for self-disclosure detection")
    
    # Dataset arguments
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face token for accessing the dataset",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory to cache the dataset and models",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing preprocessed dataset",
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help="Proportion of data to use for training (if dataset needs splitting)",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Proportion of data to use for validation (if dataset needs splitting)",
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="roberta-large",
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save the model",
    )
    
    # Training arguments
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size for training",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use mixed precision",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    args = parser.parse_args()
    return args

def main():
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Load dataset
    logger.info("Loading dataset...")
    
    # Try to load from preprocessed data directory first
    if args.data_dir and os.path.exists(os.path.join(args.data_dir, "processed_dataset")):
        logger.info(f"Loading preprocessed dataset from {args.data_dir}/processed_dataset")
        dataset = load_from_disk(os.path.join(args.data_dir, "processed_dataset"))
    else:
        # Otherwise load from Hugging Face or original dataset directory
        dataset = load_reddit_self_disclosure_dataset(
            token=args.hf_token, 
            cache_dir=args.cache_dir,
            data_dir=os.path.join(args.data_dir, "original_dataset") if args.data_dir else None
        )
        
        # Check if dataset has only a train split
        if list(dataset.keys()) == ["train"]:
            logger.info("Dataset has only a train split. Creating validation and test splits...")
            dataset = split_dataset(
                dataset["train"],
                train_split=args.train_split,
                val_split=args.val_split,
                seed=args.seed
            )
    print(dataset)
    # Get label list
    label_list = get_labels(dataset["train"])
    num_labels = len(label_list)
    logger.info(f"Number of labels: {num_labels}")
    
    # Load model and tokenizer
    logger.info(f"Loading model and tokenizer from {args.model_name_or_path}...")
    model, tokenizer = get_model_and_tokenizer(
        model_name_or_path=args.model_name_or_path,
        num_labels=num_labels,
        label_list=label_list,
        cache_dir=args.cache_dir,
    )
    lora_config = LoraConfig(
    task_type=TaskType.TOKEN_CLS,
    inference_mode=False,  # 训练模式下设置为 False
    r=8,                 # 低秩矩阵的秩，可以根据需要调整
    lora_alpha=32,       # 缩放因子
    lora_dropout=0.1,    # dropout 概率
    )

    # 应用 LoRA 到模型
    model = get_peft_model(model, lora_config)
    # Prepare dataset for training
    logger.info("Preparing dataset for training...")
    processed_dataset = prepare_dataset_for_training(
        dataset=dataset,
        tokenizer=tokenizer,
        max_length=512,
        label_all_tokens=False
    )
    print(processed_dataset)
    # Define compute_metrics function for the Trainer
    def compute_metrics_fn(eval_preds):
        preds, labels = eval_preds
        # First, compute standard metrics
        metrics = compute_metrics(eval_preds, label_list)
        
        # Then, compute partial span F1
        partial_metrics = compute_partial_span_f1(
            np.argmax(preds, axis=2),
            labels,
            label_list
        )
        
        # Combine metrics
        metrics.update(partial_metrics)
        return metrics
    
    # Get training arguments
    training_args = TrainingArguments(
        **get_training_args(
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            warmup_steps=args.warmup_steps,
            weight_decay=args.weight_decay,
            learning_rate=args.learning_rate,
            fp16=args.fp16,
            seed=args.seed,
        )
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_fn,
    )
    
    # Train the model
    logger.info("Training the model...")
    trainer.train()
    
    # Evaluate the model
    logger.info("Evaluating the model...")
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_results}")
    
    # Save evaluation results
    with open(os.path.join(args.output_dir, "eval_results.txt"), "w") as f:
        for key, value in eval_results.items():
            f.write(f"{key} = {value}\n")
    
    # Save the model
    logger.info(f"Saving the model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Test the model
    logger.info("Testing the model...")
    test_results = trainer.evaluate(processed_dataset["test"])
    logger.info(f"Test results: {test_results}")
    
    # Save test results
    with open(os.path.join(args.output_dir, "test_results.txt"), "w") as f:
        for key, value in test_results.items():
            f.write(f"{key} = {value}\n")
    
    logger.info("Done!")

if __name__ == "__main__":
    main()
