"""
Data utilities for loading and preprocessing the Reddit self-disclosure dataset.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import PreTrainedTokenizer
from sklearn.metrics import classification_report
from seqeval.metrics import f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)

def parse_text_to_tokens_and_tags(example: Dict) -> Dict:
    """
    Parse a CoNLL-style text string into tokens and tags.

    Args:
        example: A dictionary with a 'text' field.

    Returns:
        A dictionary with 'tokens' and 'tags' lists.
    """
    tokens = []
    tags = []

    for line in example["text"].splitlines():
        line = line.strip()
        if not line or line == "[SEP]" or ".txt" in line:
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) == 2:
            token, tag = parts
        else:
            token, tag = parts[0], "O"
        tokens.append(token)
        tags.append(tag)

    return {"tokens": tokens, "tags": tags}

def load_reddit_self_disclosure_dataset(
    token: Optional[str] = None,
    cache_dir: Optional[str] = None,
    data_dir: Optional[str] = None
) -> DatasetDict:
    """
    Load the Reddit self-disclosure dataset from Hugging Face or local directory.
    
    Args:
        token: Hugging Face token for accessing the dataset
        cache_dir: Directory to cache the dataset
        data_dir: Directory containing locally saved dataset
        
    Returns:
        DatasetDict containing dataset splits
    """
    try:
        # First try to load from local directory if specified
        if data_dir and os.path.exists(data_dir):
            logger.info(f"Loading dataset from local directory: {data_dir}")
            return DatasetDict.load_from_disk(data_dir)
        
        # Otherwise load from Hugging Face
        logger.info("Loading dataset from Hugging Face")
        dataset = load_dataset(
            "douy/reddit-self-disclosure",
            cache_dir=cache_dir
        )
        dataset = dataset.map(parse_text_to_tokens_and_tags, remove_columns=["text"])
        return dataset
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def split_dataset(
    dataset: Dataset,
    train_split: float = 0.8,
    val_split: float = 0.1,
    seed: int = 42
) -> DatasetDict:
    """
    Split a dataset into train, validation, and test sets.
    
    Args:
        dataset: Dataset to split
        train_split: Proportion of data to use for training
        val_split: Proportion of data to use for validation
        seed: Random seed for reproducibility
        
    Returns:
        DatasetDict containing train, validation, and test splits
    """
    # Shuffle the dataset
    shuffled_dataset = dataset.shuffle(seed=seed)
    
    # Calculate split sizes
    dataset_size = len(shuffled_dataset)
    train_size = int(dataset_size * train_split)
    val_size = int(dataset_size * val_split)
    
    # Split the dataset
    train_dataset = shuffled_dataset.select(range(train_size))
    val_dataset = shuffled_dataset.select(range(train_size, train_size + val_size))
    test_dataset = shuffled_dataset.select(range(train_size + val_size, dataset_size))
    
    # Create DatasetDict
    return DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })

def read_conll_format(file_path: str) -> List[List[Tuple[str, str]]]:
    """
    Read data in CoNLL format from a file.
    
    Args:
        file_path: Path to the CoNLL format file
        
    Returns:
        List of sentences, where each sentence is a list of (token, tag) tuples
    """
    sentences = []
    current_sentence = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == "" or line == "[SEP]":
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
            elif ".txt" in line:
                continue
            else:
                parts = line.split()
                if len(parts) >= 2:
                    token = parts[0]
                    tag = parts[1]
                    current_sentence.append((token, tag))
    
    if current_sentence:
        sentences.append(current_sentence)
    
    return sentences

def convert_to_iob2_format(sentences: List[List[Tuple[str, str]]]) -> List[List[Tuple[str, str]]]:
    """
    Ensure tags are in IOB2 format.
    
    Args:
        sentences: List of sentences with (token, tag) tuples
        
    Returns:
        List of sentences with tags converted to IOB2 format
    """
    iob2_sentences = []
    
    for sentence in sentences:
        iob2_sentence = []
        prev_tag = "O"
        
        for token, tag in sentence:
            if tag != "O" and not tag.startswith("B-") and not tag.startswith("I-"):
                # If tag is not in IOB format, convert it
                tag = f"B-{tag}"
            
            if tag.startswith("I-") and (prev_tag == "O" or prev_tag.split("-")[1] != tag.split("-")[1]):
                # If I- tag doesn't follow a B- tag of the same type, convert to B-
                tag = f"B-{tag[2:]}"
            
            iob2_sentence.append((token, tag))
            prev_tag = tag
        
        iob2_sentences.append(iob2_sentence)
    
    return iob2_sentences

def prepare_dataset_for_training(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    label_all_tokens: bool = False
) -> DatasetDict:
    """
    Prepare the dataset for training by tokenizing and aligning labels.
    
    Args:
        dataset: Dataset dictionary with splits
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        label_all_tokens: Whether to label all tokens or just the first subword
        
    Returns:
        Processed dataset dictionary
    """
    # Get label list
    label_list = get_labels(dataset["train"])
    
    # Create label mappings
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    
    # Define preprocessing function with batched processing
    def preprocess_function(examples):
        # Tokenize the texts
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",  # Return PyTorch tensors for better compatibility
        )
        
        # Align labels with tokens efficiently
        labels = []
        for i, label in enumerate(examples["tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            
            # Process span-by-span instead of token-by-token for efficiency
            current_tag = None
            current_start = None
            
            for j, word_idx in enumerate(word_ids):
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    # New word
                    label_ids.append(label2id[label[word_idx]])
                else:
                    # Continuation of the same word
                    label_ids.append(label2id[label[word_idx]] if label_all_tokens else -100)
                previous_word_idx = word_idx
            
            labels.append(label_ids)
        
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    # Process datasets with optimized memory usage
    processed_dataset = {}
    for split in dataset.keys():
        # Use batched processing with larger batch size for speed
        processed_dataset[split] = dataset[split].map(
            preprocess_function,
            batched=True,
            batch_size=64,  # Process larger batches for efficiency
            remove_columns=dataset[split].column_names,
            desc=f"Processing {split} dataset",
            num_proc=4,  # Use multiple processors if available
        )
        
        # Set format to PyTorch tensors
        processed_dataset[split].set_format(
            type="torch", 
            columns=["input_ids", "attention_mask", "labels"],
        )
    
    return DatasetDict(processed_dataset)

def get_labels(dataset: Dataset) -> List[str]:
    """
    Get unique labels from the dataset.
    
    Args:
        dataset: Dataset containing tags
        
    Returns:
        List of unique labels
    """
    labels = set()
    for example in dataset:
        for tag in example["tags"]:
            if tag != "O":
                labels.add(tag)
    
    # Sort labels and ensure "O" is first
    return ["O"] + sorted(list(labels))

def compute_metrics(p, label_list):
    """
    Compute metrics for token classification.
    
    Args:
        p: Predictions and labels
        label_list: List of possible labels
        
    Returns:
        Dictionary of metrics
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    # Remove ignored index (special tokens) efficiently using numpy operations
    true_predictions = []
    true_labels = []
    
    # Process in batches to reduce memory pressure
    batch_size = 32
    for i in range(0, len(predictions), batch_size):
        batch_preds = predictions[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        
        for prediction, label in zip(batch_preds, batch_labels):
            mask = label != -100
            true_predictions.append([label_list[p] for p, m in zip(prediction[mask], mask) if m])
            true_labels.append([label_list[l] for l, m in zip(label[mask], mask) if m])
    
    # Calculate metrics using seqeval
    results = {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }
    
    return results

def compute_partial_span_f1(predictions, labels, label_list):
    """
    Compute partial span F1 score as described in the paper.
    
    Args:
        predictions: Model predictions
        labels: True labels
        label_list: List of possible labels
        
    Returns:
        Dictionary of metrics including partial span F1
    """
    # Convert predictions and labels to tag sequences
    pred_tags = []
    true_tags = []
    
    # Process in batches to reduce memory pressure
    batch_size = 32
    for i in range(0, len(predictions), batch_size):
        batch_preds = predictions[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        
        for prediction, label in zip(batch_preds, batch_labels):
            mask = label != -100
            pred_tags.append([label_list[p] for p, m in zip(prediction[mask], mask) if m])
            true_tags.append([label_list[l] for l, m in zip(label[mask], mask) if m])
    
    # Extract spans from tags
    pred_spans = []
    true_spans = []
    
    for pred_seq, true_seq in zip(pred_tags, true_tags):
        pred_spans_sent = extract_spans(pred_seq)
        true_spans_sent = extract_spans(true_seq)
        
        pred_spans.append(pred_spans_sent)
        true_spans.append(true_spans_sent)
    
    # Calculate partial span F1
    tp = 0
    fp = 0
    fn = 0
    
    for pred_spans_sent, true_spans_sent in zip(pred_spans, true_spans):
        for pred_span in pred_spans_sent:
            matched = False
            for true_span in true_spans_sent:
                if spans_overlap(pred_span, true_span, threshold=0.5):
                    tp += 1
                    matched = True
                    break
            if not matched:
                fp += 1
        
        for true_span in true_spans_sent:
            matched = False
            for pred_span in pred_spans_sent:
                if spans_overlap(true_span, pred_span, threshold=0.5):
                    matched = True
                    break
            if not matched:
                fn += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "partial_span_precision": precision,
        "partial_span_recall": recall,
        "partial_span_f1": f1
    }

def extract_spans(tag_seq):
    """
    Extract spans from a sequence of IOB2 tags.
    
    Args:
        tag_seq: Sequence of IOB2 tags
        
    Returns:
        List of spans, where each span is (start_idx, end_idx, type)
    """
    spans = []
    current_span = None
    
    for i, tag in enumerate(tag_seq):
        if tag == "O":
            if current_span is not None:
                spans.append((current_span[0], i - 1, current_span[1]))
                current_span = None
        elif tag.startswith("B-"):
            if current_span is not None:
                spans.append((current_span[0], i - 1, current_span[1]))
            current_span = (i, tag[2:])
        elif tag.startswith("I-"):
            if current_span is None or current_span[1] != tag[2:]:
                # This is an error in the tagging, but we'll handle it by starting a new span
                current_span = (i, tag[2:])
    
    # Add the last span if there is one
    if current_span is not None:
        spans.append((current_span[0], len(tag_seq) - 1, current_span[1]))
    
    return spans

def spans_overlap(span1, span2, threshold=0.5):
    """
    Check if two spans overlap by at least the threshold.
    
    Args:
        span1: First span (start_idx, end_idx, type)
        span2: Second span (start_idx, end_idx, type)
        threshold: Minimum overlap ratio (0.5 means 50% overlap)
        
    Returns:
        Boolean indicating if spans overlap by at least the threshold
    """
    # Spans must be of the same type
    if span1[2] != span2[2]:
        return False
    
    # Calculate overlap
    start1, end1, _ = span1
    start2, end2, _ = span2
    
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    
    if overlap_start <= overlap_end:
        overlap_length = overlap_end - overlap_start + 1
        span1_length = end1 - start1 + 1
        span2_length = end2 - start2 + 1
        
        # Check if overlap is at least threshold of the longer span
        longer_span_length = max(span1_length, span2_length)
        return overlap_length / longer_span_length >= threshold
    
    return False
