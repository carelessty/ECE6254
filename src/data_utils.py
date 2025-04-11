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

def _process_raw_text_split(raw_text_lines: List[str]) -> List[Dict[str, List[str]]]:
    """
    Processes raw CoNLL lines for a split into examples based on separators.

    Args:
        raw_text_lines: A list of strings, where each string is a line from the CoNLL file.

    Returns:
        A list of dictionaries, where each dictionary represents an example
        (sentence/document) with 'tokens' and 'tags' keys.
    """
    examples = []
    current_tokens = []
    current_tags = []

    for line in raw_text_lines:
        line = line.strip()
        # Ignore document markers for now, they don't separate examples in this context
        if ".txt" in line:
            continue
        # [SEP] or empty line acts as a separator between examples
        elif not line or line == "[SEP]":
            if current_tokens:  # Ensure we don't add empty examples
                examples.append({"tokens": list(current_tokens), "tags": list(current_tags)})
                current_tokens = []
                current_tags = []
        # Process lines with token and tag
        else:
            parts = line.rsplit(" ", 1)
            if len(parts) == 2:
                token, tag = parts
            else:
                # Handle potential missing tags, default to 'O'
                token, tag = parts[0], "O"
                # Log a warning if a tag is missing
                # logger.warning(f"Line missing tag, defaulting to 'O': {line}") # Optional: uncomment for debugging
            current_tokens.append(token)
            current_tags.append(tag)

    # Add the last example if the file doesn't end with a separator
    if current_tokens:
        examples.append({"tokens": list(current_tokens), "tags": list(current_tags)})

    return examples

def load_reddit_self_disclosure_dataset(
    token: Optional[str] = None,
    cache_dir: Optional[str] = None,
    data_dir: Optional[str] = None
) -> DatasetDict:
    """
    Load the Reddit self-disclosure dataset from Hugging Face or local directory,
    correctly parsing examples based on separators.
    
    Args:
        token: Hugging Face token for accessing the dataset.
        cache_dir: Directory to cache the dataset.
        data_dir: Directory containing locally saved dataset.
        
    Returns:
        DatasetDict containing dataset splits with correctly parsed examples.
    """
    try:
        raw_dataset: Optional[DatasetDict] = None
        # First try to load from local directory if specified
        if data_dir and os.path.exists(data_dir):
            logger.info(f"Loading dataset from local directory: {data_dir}")
            # Assuming local data is already processed or needs similar raw loading
            # For simplicity, let's assume it needs the same processing for now.
            # If the saved format is different, this part needs adjustment.
            # We might need a flag or check to see if it's already processed.
            # Let's load it as raw text first.
            # This assumes the local format is loadable by `load_dataset` directly
            # e.g., saved using dataset.save_to_disk() with the raw text format.
            # If saved AFTER the incorrect processing, it needs to be regenerated.
            # For now, let's prioritize the Hugging Face loading logic.
            # If loading from disk, we might need to adapt the processing.
            # Reverting to HF load if local load doesn't work as expected for raw text.
            try:
                 # Attempt to load assuming it contains raw 'text' column
                 raw_dataset = DatasetDict.load_from_disk(data_dir)
                 if "text" not in raw_dataset[list(raw_dataset.keys())[0]].column_names:
                      logger.warning(f"Loaded local dataset from {data_dir} does not contain 'text' column. Falling back to Hugging Face.")
                      raw_dataset = None # Force reload from HF
            except Exception as load_err:
                 logger.warning(f"Could not load raw dataset from {data_dir}: {load_err}. Falling back to Hugging Face.")
                 raw_dataset = None


        # Load from Hugging Face if not loaded locally
        if raw_dataset is None:
             logger.info("Loading raw dataset from Hugging Face Hub (douy/reddit-self-disclosure)")
             raw_dataset = load_dataset(
                 "douy/reddit-self-disclosure",
                 token=token, # Pass token if provided
                 cache_dir=cache_dir
             )
             if data_dir: # If a data_dir was specified, save the raw dataset there
                 logger.info(f"Saving raw dataset to {data_dir}")
                 # Ensure parent directories exist
                 os.makedirs(data_dir, exist_ok=True)
                 raw_dataset.save_to_disk(data_dir)


        # Process each split
        processed_splits = {}
        logger.info("Processing raw dataset splits into examples...")
        if len(raw_dataset.keys()) > 1:
            logger.warning(f"Raw dataset has multiple splits: {list(raw_dataset.keys())}. Processing each separately, but splitting logic assumes a single 'train' split as source.")
        
        # Assume the primary data is in the 'train' split of the raw dataset
        # or the first split if 'train' is not present
        source_split_name = 'train' if 'train' in raw_dataset else list(raw_dataset.keys())[0]
        if source_split_name not in raw_dataset:
             raise ValueError(f"Cannot find a source split ('{source_split_name}' or first available) in the raw dataset.")

        logger.info(f"Using raw split '{source_split_name}' as the source for processing.")
        source_split_data = raw_dataset[source_split_name]
        
        # Combine all text lines for the split.
        all_lines = "\n".join(source_split_data["text"]).splitlines()
        # Process the combined lines to get correctly structured examples
        processed_examples = _process_raw_text_split(all_lines)
        
        # Create a single Dataset from the list of processed examples
        processed_dataset = Dataset.from_dict(
            {"tokens": [ex["tokens"] for ex in processed_examples],
             "tags": [ex["tags"] for ex in processed_examples]}
        )
        logger.info(f"Processed source split '{source_split_name}': {len(processed_dataset)} examples.")

        # Now, split the processed dataset into train, validation, and test sets
        logger.info("Splitting the processed dataset into train, validation, and test sets...")
        final_dataset_splits = split_dataset(
            dataset=processed_dataset,
            # Using default splits: 80% train, 10% validation, 10% test
            # train_split=0.8, 
            # val_split=0.1,
            # seed=42 # split_dataset uses seed=42 by default
        )
        logger.info(f"Splitting complete. Train: {len(final_dataset_splits['train'])}, Validation: {len(final_dataset_splits['validation'])}, Test: {len(final_dataset_splits['test'])} examples.")

        # Return the DatasetDict with train, validation, test splits
        return final_dataset_splits

    except Exception as e:
        logger.error(f"Error loading and processing dataset: {e}", exc_info=True) # Add traceback info
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
    
    # Remove ignored index (special tokens) and convert to label strings
    true_predictions = []
    true_labels = []
    
    for i in range(len(predictions)):
        prediction = predictions[i]
        label = labels[i]
        true_prediction_sequence = []
        true_label_sequence = []
        for pred_id, label_id in zip(prediction, label):
            if label_id != -100: # Only consider active labels
                true_prediction_sequence.append(label_list[pred_id])
                true_label_sequence.append(label_list[label_id])
        true_predictions.append(true_prediction_sequence)
        true_labels.append(true_label_sequence)
    
    # Calculate metrics using seqeval
    results = {
        "precision": precision_score(true_labels, true_predictions, zero_division=0),
        "recall": recall_score(true_labels, true_predictions, zero_division=0),
        "f1": f1_score(true_labels, true_predictions, zero_division=0),
    }
    
    return results

def compute_partial_span_f1(predictions, labels, label_list):
    """
    Compute partial span F1 score as described in the paper.
    
    Args:
        predictions: Model predictions (logits or argmaxed IDs)
        labels: True labels
        label_list: List of possible labels
        
    Returns:
        Dictionary of metrics including partial span F1
    """
    # Ensure predictions are class IDs
    if predictions.ndim == 3: # Check if predictions are logits
        predictions = np.argmax(predictions, axis=2)
        
    # Convert predictions and labels to tag sequences, ignoring -100
    pred_tags = []
    true_tags = []
    
    for i in range(len(predictions)):
        prediction = predictions[i]
        label = labels[i]
        pred_tag_sequence = []
        true_tag_sequence = []
        for pred_id, label_id in zip(prediction, label):
            if label_id != -100: # Only consider active labels
                pred_tag_sequence.append(label_list[pred_id])
                true_tag_sequence.append(label_list[label_id])
        pred_tags.append(pred_tag_sequence)
        true_tags.append(true_tag_sequence)
    
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
        # Track matched true spans to avoid double counting for FN calculation
        matched_true_spans = set()
        for pred_span in pred_spans_sent:
            matched_this_pred = False # Track if this specific predicted span found any match
            for idx, true_span in enumerate(true_spans_sent):
                if spans_overlap(pred_span, true_span, threshold=0.5):
                    # Found an overlap! Increment TP for this predicted span.
                    tp += 1
                    # Mark the corresponding true span as matched (used for FN calculation).
                    matched_true_spans.add(idx)
                    # Mark that this predicted span found a match.
                    matched_this_pred = True
                    # A predicted span counts as one TP even if it overlaps multiple ground truths.
                    # Break after finding the first match for this predicted span.
                    break
            if not matched_this_pred:
                # This predicted span did not overlap sufficiently with ANY true span.
                fp += 1
        
        # Calculate false negatives: True spans that were not matched by any prediction span.
        fn += len(true_spans_sent) - len(matched_true_spans)
    
    # Added checks for division by zero
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
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
