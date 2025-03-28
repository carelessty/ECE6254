"""
Main script for training and evaluating privacy risk detection models.
This script implements the methodology from the paper for adapting
DeepSeek-R1-Distill-Qwen-1.5B for privacy risk detection.
"""

import os
import argparse
import torch
from transformers import AutoTokenizer, set_seed

from data_utils import load_self_disclosure_dataset
from model import PrivacyRiskClassifier

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Privacy Risk Detection")
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/deepseek-r1-distill-qwen-1.5b",
        help="Name or path of the pre-trained model"
    )
    parser.add_argument(
        "--reference_model",
        type=str,
        default="douy/roberta-large-self-disclosure-sentence-classification",
        help="Name or path of the reference model for comparison"
    )
    
    # Task arguments
    parser.add_argument(
        "--task",
        type=str,
        choices=["classification", "span"],
        default="classification",
        help="Task type: sentence-level classification or span-level detection"
    )
    
    # Training arguments
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Whether to use LoRA for parameter-efficient fine-tuning"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training and evaluation"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save results"
    )
    
    return parser.parse_args()

def main():
    """Main function for training and evaluating privacy risk detection models."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize tokenizer
    print(f"Loading tokenizer for {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Ensure the tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    print("Loading dataset...")
    train_dataloader, val_dataloader, test_dataloader = load_self_disclosure_dataset(
        tokenizer, batch_size=args.batch_size, task=args.task
    )
    
    # Initialize model
    print(f"Initializing {args.model_name} for fine-tuning...")
    model = PrivacyRiskClassifier(
        model_name=args.model_name,
        task=args.task,
        use_lora=args.use_lora
    )
    
    # Train model
    print("Training model...")
    model.train(
        train_dataloader,
        val_dataloader,
        output_dir=os.path.join(args.output_dir, "model"),
        num_epochs=args.num_epochs
    )
    
    # Save model
    model.save(os.path.join(args.output_dir, "model"))
    
    # Evaluate model
    print("Evaluating model...")
    # Extract texts and labels from test dataloader
    texts = []
    labels = []
    for batch in test_dataloader:
        batch_texts = tokenizer.batch_decode(batch[0], skip_special_tokens=True)
        batch_labels = batch[2].tolist()
        texts.extend(batch_texts)
        labels.extend(batch_labels)
    
    # Make predictions
    predictions = model.predict(texts)
    
    # Calculate metrics
    accuracy = sum(p == l for p, l in zip(predictions, labels)) / len(labels)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Save results
    with open(os.path.join(args.output_dir, "results.txt"), "w") as f:
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Task: {args.task}\n")
        f.write(f"Test accuracy: {accuracy:.4f}\n")
    
    print("Done!")

if __name__ == "__main__":
    main()
