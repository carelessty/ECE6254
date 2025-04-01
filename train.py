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
    TrainerCallback,
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
        default=128,
        help="Batch size for training",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=128,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before backward pass",
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
        "--bf16",
        action="store_true",
        help="Whether to use bfloat16 mixed precision",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--gradient_checkpointing", 
        action="store_true",
        help="Enable gradient checkpointing to save memory"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Run evaluation every X steps",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X steps",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Log every X steps",
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
    
    # Prevent using both FP16 and BF16 at the same time, which can cause issues
    if args.fp16 and args.bf16:
        logger.warning("Both fp16 and bf16 were enabled. Disabling bf16.")
        args.bf16 = False
    
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
    
    # Enable gradient checkpointing if specified
    if args.gradient_checkpointing:
        logger.info("Enabling gradient checkpointing")
        model.gradient_checkpointing_enable()
    
    # Configure LoRA for parameter-efficient fine-tuning
    lora_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        inference_mode=False,
        r=16,                 # Increased rank for better expressivity
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "key", "value"],  # Target specific modules for efficiency
    )

    # Apply LoRA to model
    logger.info("Applying LoRA adaptation to model")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Log trainable parameters %
    
    # Prepare dataset for training efficiently
    logger.info("Preparing dataset for training...")
    processed_dataset = prepare_dataset_for_training(
        dataset=dataset,
        tokenizer=tokenizer,
        max_length=512,
        label_all_tokens=False
    )
    
    # Log dataset sizes
    logger.info(f"Train dataset size: {len(processed_dataset['train'])}")
    logger.info(f"Validation dataset size: {len(processed_dataset['validation'])}")
    logger.info(f"Test dataset size: {len(processed_dataset['test'])}")
    
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        save_strategy="steps",
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=f"{args.output_dir}/logs",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_total_limit=2,  # Only keep the 2 best checkpoints
        overwrite_output_dir=True,
        dataloader_num_workers=4,  # Reduced workers for less memory pressure
        group_by_length=True,  # Group similar length sequences for efficiency
        report_to="tensorboard",
        label_names=["labels"],
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        seed=args.seed,
        # Memory optimization
        optim="adamw_torch",  # Use torch implementation for better memory usage
        ddp_find_unused_parameters=False,  # Improve DDP efficiency
        torch_compile=False,  # Disable torch compile to avoid memory spikes
        gradient_checkpointing=args.gradient_checkpointing,
        # Add max_grad_norm to handle gradient explosions
        max_grad_norm=1.0,
        # Restrict evaluation size to prevent OOMs during validation
        eval_accumulation_steps=8,  # Accumulate gradients during eval to reduce memory
        # Set eval batch size reduction to use less memory during evaluation
    )
    
    # Function to free up memory after evaluation
    def cleanup_memory():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        import gc
        gc.collect()
    
    # Create a smaller validation dataset if the evaluation set is too large
    # This helps prevent OOM during validation
    eval_dataset = processed_dataset["validation"]
    if len(eval_dataset) > 1000:  # If validation set is large
        logger.info(f"Creating smaller validation dataset for frequent evaluation")
        # Use a stratified subset for validation to maintain distribution
        eval_indices = np.random.choice(
            len(eval_dataset), 
            size=min(1000, len(eval_dataset)), 
            replace=False
        )
        validation_subset = eval_dataset.select(eval_indices)
        logger.info(f"Using validation subset of size {len(validation_subset)} for training")
    else:
        validation_subset = eval_dataset
    
    # Modified compute_metrics to be more memory-efficient
    def compute_metrics_fn(eval_preds):
        preds, labels = eval_preds
        # Process predictions in smaller chunks to avoid OOM
        chunk_size = 32
        metrics_list = []
        
        # Process in chunks
        for i in range(0, len(preds), chunk_size):
            chunk_preds = preds[i:i+chunk_size]
            chunk_labels = labels[i:i+chunk_size]
            
            # Calculate metrics for this chunk
            chunk_metrics = compute_metrics((chunk_preds, chunk_labels), label_list)
            metrics_list.append(chunk_metrics)
        
        # Average metrics across chunks
        metrics = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0].keys()}
        
        # Compute partial span F1 more efficiently
        partial_metrics = compute_partial_span_f1(
            np.argmax(preds, axis=2),
            labels,
            label_list
        )
        
        # Combine metrics
        metrics.update(partial_metrics)
        
        # Force garbage collection to free memory
        cleanup_memory()
        
        return metrics
    
    # Create a custom callback to clean up memory after evaluation
    class MemoryCleanupCallback(TrainerCallback):
        """Callback to clean up memory after evaluation."""
        
        def on_evaluate(self, args, state, control, **kwargs):
            """Called after evaluation."""
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            import gc
            gc.collect()
            return control
        
        def on_prediction_step(self, args, state, control, **kwargs):
            """Called after each prediction step."""
            # Less aggressive cleanup during prediction to avoid too much overhead
            if torch.cuda.is_available() and state.global_step % 10 == 0:
                torch.cuda.empty_cache()
            return control
        
        def on_save(self, args, state, control, **kwargs):
            """Called after model saving."""
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            import gc
            gc.collect()
            return control
    
    # Initialize Trainer with memory-efficient validation
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset["train"],
        eval_dataset=validation_subset,  # Use smaller validation set
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_fn,
        callbacks=[
            # Add custom callback to clean up memory after evaluation
            MemoryCleanupCallback()
        ],
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
