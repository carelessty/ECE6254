"""
Dataset loading and preprocessing script for the Reddit self-disclosure dataset.
"""

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

from src.data_utils import (
    load_reddit_self_disclosure_dataset,
    read_conll_format,
    convert_to_iob2_format,
    get_labels,
    split_dataset
)

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Process Reddit self-disclosure dataset")
    
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
        help="Directory to cache the dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Directory to save processed data",
    )
    parser.add_argument(
        "--local_data_file",
        type=str,
        default=None,
        help="Path to local data file in CoNLL format (if not using HF dataset)",
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help="Proportion of data to use for training",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Proportion of data to use for validation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset splitting",
    )
    
    args = parser.parse_args()
    return args

def process_conll_data(file_path: str, output_dir: str, train_split: float = 0.8, val_split: float = 0.1, seed: int = 42):
    """
    Process data in CoNLL format and create train/val/test splits.
    
    Args:
        file_path: Path to the CoNLL format file
        output_dir: Directory to save processed data
        train_split: Proportion of data to use for training
        val_split: Proportion of data to use for validation
        seed: Random seed for reproducibility
    
    Returns:
        DatasetDict containing train, validation, and test splits
    """
    # Read data
    logger.info(f"Reading data from {file_path}...")
    sentences = read_conll_format(file_path)
    
    # Ensure IOB2 format
    logger.info("Converting to IOB2 format...")
    iob2_sentences = convert_to_iob2_format(sentences)
    
    # Create dataset
    tokens_list = []
    tags_list = []
    
    for sentence in iob2_sentences:
        tokens, tags = zip(*sentence)
        tokens_list.append(list(tokens))
        tags_list.append(list(tags))
    
    # Create DataFrame
    df = pd.DataFrame({"tokens": tokens_list, "tags": tags_list})
    
    # Split data
    logger.info(f"Splitting data into train/val/test with ratios {train_split}/{val_split}/{1-train_split-val_split}...")
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # Shuffle
    
    train_end = int(len(df) * train_split)
    val_end = int(len(df) * (train_split + val_split))
    
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]
    
    # Create datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # Create DatasetDict
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })
    
    # Save processed data
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    logger.info(f"Saving processed data to {output_dir}...")
    dataset_dict.save_to_disk(os.path.join(output_dir, "processed_dataset"))
    
    # Print statistics
    logger.info(f"Train set: {len(train_dataset)} examples")
    logger.info(f"Validation set: {len(val_dataset)} examples")
    logger.info(f"Test set: {len(test_dataset)} examples")
    
    return dataset_dict

def analyze_dataset(dataset: DatasetDict):
    """
    Analyze the dataset and print statistics.
    
    Args:
        dataset: Dataset to analyze
    """
    logger.info("Analyzing dataset...")
    
    # Get label list
    label_list = get_labels(dataset["train"])
    logger.info(f"Labels: {label_list}")
    logger.info(f"Number of labels: {len(label_list)}")
    
    # Count examples per split
    for split in dataset.keys():
        logger.info(f"{split.capitalize()} set: {len(dataset[split])} examples")
    
    # Count labels in train set
    label_counts = {}
    for example in dataset["train"]:
        for tag in example["tags"]:
            if tag not in label_counts:
                label_counts[tag] = 0
            label_counts[tag] += 1
    
    logger.info("Label distribution in train set:")
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {label}: {count}")
    
    # Calculate average sequence length
    seq_lengths = [len(example["tokens"]) for example in dataset["train"]]
    avg_seq_length = sum(seq_lengths) / len(seq_lengths)
    max_seq_length = max(seq_lengths)
    
    logger.info(f"Average sequence length: {avg_seq_length:.2f}")
    logger.info(f"Maximum sequence length: {max_seq_length}")

def main():
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Load dataset
    if args.local_data_file:
        logger.info(f"Processing local data file: {args.local_data_file}")
        dataset = process_conll_data(
            file_path=args.local_data_file,
            output_dir=args.output_dir,
            train_split=args.train_split,
            val_split=args.val_split,
            seed=args.seed
        )
    else:
        logger.info("Loading dataset from Hugging Face...")
        # Load the dataset. Splitting now happens inside this function.
        dataset = load_reddit_self_disclosure_dataset(
            token=args.hf_token,
            cache_dir=args.cache_dir,
            # Optionally pass a directory to save the *raw* downloaded data
            data_dir=os.path.join(args.output_dir, "raw_downloaded") if args.output_dir else None 
        )
        logger.info(f"Dataset loaded and split. Splits: {list(dataset.keys())}")
        
        # The split_dataset call is no longer needed here as it's done internally
        # # Check if dataset has only a train split
        # if list(dataset.keys()) == ["train"]:
        #     logger.info("Dataset has only a train split. Creating validation and test splits...")
        #     dataset = split_dataset(
        #         dataset["train"],
        #         train_split=args.train_split,
        #         val_split=args.val_split,
        #         seed=args.seed
        #     )
        
        # Save the processed and split dataset to disk
        # It seems this script might be intended to save the *preprocessed* (tokenized) data?
        # Currently, it saves the data *before* tokenization. 
        # If the goal is to save the tokenized data, we need to add the call to 
        # prepare_dataset_for_training here.
        # For now, just save the result of load_reddit_self_disclosure_dataset.
        processed_save_path = os.path.join(args.output_dir, "processed_splits")
        logger.info(f"Saving processed dataset splits to {processed_save_path}...")
        os.makedirs(processed_save_path, exist_ok=True)
        dataset.save_to_disk(processed_save_path)
    
    # Analyze dataset (this should analyze the loaded/split dataset)
    analyze_dataset(dataset)
    
    logger.info("Done!")

if __name__ == "__main__":
    main()
