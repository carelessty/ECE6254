"""
Data preprocessing utilities for privacy risk detection in user-generated content.
This module handles loading and preprocessing the reddit-self-disclosure dataset.
"""

import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

class SelfDisclosureDataset(Dataset):
    """
    Dataset class for self-disclosure detection task.
    
    This class handles the preprocessing of the reddit-self-disclosure dataset
    for both sentence-level classification and span-level detection tasks.
    """
    
    def __init__(self, dataset, tokenizer, max_length=512, task="classification"):
        """
        Initialize the dataset.
        
        Args:
            dataset: The HuggingFace dataset
            tokenizer: The tokenizer to use for encoding
            max_length: Maximum sequence length
            task: Either "classification" for sentence-level or "span" for span-level detection
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task = task
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Get a single example from the dataset.
        
        For classification task, returns tokenized text and binary label.
        For span task, returns tokenized text and token-level labels.
        """
        item = self.dataset[idx]
        
        if self.task == "classification":
            # For sentence-level classification
            encoding = self.tokenizer(
                item["text"],
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            # Remove batch dimension
            encoding = {k: v.squeeze(0) for k, v in encoding.items()}
            encoding["labels"] = torch.tensor(item["label"], dtype=torch.long)
            
            return encoding
        
        elif self.task == "span":
            # For span-level detection (IOB2 format)
            tokens = item["tokens"]
            iob_tags = item["tags"]
            
            # Convert IOB2 tags to binary labels (1 for B- or I-, 0 for O)
            binary_labels = [1 if tag.startswith(("B-", "I-")) else 0 for tag in iob_tags]
            
            # Tokenize and align labels
            encoding = self.tokenizer(
                tokens,
                is_split_into_words=True,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            # Remove batch dimension
            encoding = {k: v.squeeze(0) for k, v in encoding.items()}
            
            # Align labels with wordpiece tokens
            word_ids = encoding.word_ids()
            aligned_labels = []
            
            for word_id in word_ids:
                if word_id is None:
                    # Special tokens get label -100 (ignored in loss)
                    aligned_labels.append(-100)
                else:
                    # Use the label of the word
                    if word_id < len(binary_labels):
                        aligned_labels.append(binary_labels[word_id])
                    else:
                        # Handle truncation
                        aligned_labels.append(-100)
            
            encoding["labels"] = torch.tensor(aligned_labels, dtype=torch.long)
            
            return encoding
        
        else:
            raise ValueError(f"Unknown task: {self.task}")


def load_self_disclosure_dataset(tokenizer, batch_size=16, task="classification"):
    """
    Load and prepare the reddit-self-disclosure dataset.
    
    Args:
        tokenizer: The tokenizer to use
        batch_size: Batch size for DataLoader
        task: Either "classification" for sentence-level or "span" for span-level detection
    
    Returns:
        train_dataloader, val_dataloader, test_dataloader
    """
    try:
        # Try to load the dataset from Hugging Face
        dataset = load_dataset("douy/reddit-self-disclosure")
        
        # If the dataset requires login, we'll need to handle that separately
        # This is a placeholder for the actual dataset structure
        if "train" not in dataset:
            print("Dataset structure doesn't match expectations. Using mock data for development.")
            # Create mock data for development
            return create_mock_dataloaders(tokenizer, batch_size, task)
        
        # Create dataset objects
        train_dataset = SelfDisclosureDataset(dataset["train"], tokenizer, task=task)
        val_dataset = SelfDisclosureDataset(dataset["validation"], tokenizer, task=task)
        test_dataset = SelfDisclosureDataset(dataset["test"], tokenizer, task=task)
        
        # Create dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
        
        return train_dataloader, val_dataloader, test_dataloader
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Using mock data for development.")
        return create_mock_dataloaders(tokenizer, batch_size, task)


def create_mock_dataloaders(tokenizer, batch_size=16, task="classification"):
    """
    Create mock data for development when the actual dataset is not accessible.
    
    Args:
        tokenizer: The tokenizer to use
        batch_size: Batch size for DataLoader
        task: Either "classification" for sentence-level or "span" for span-level detection
    
    Returns:
        train_dataloader, val_dataloader, test_dataloader with mock data
    """
    from torch.utils.data import TensorDataset
    
    # Example sentences from the paper
    mock_data = [
        {"text": "I am a 23-year-old who is currently going through the last leg of undergraduate school.", "label": 1},
        {"text": "There is a joke in the design industry about that.", "label": 0},
        {"text": "My husband and I live in US.", "label": 1},
        {"text": "I was messing with advanced voice the other day and I was like, 'Oh, I can do this.'", "label": 0},
        {"text": "I'm 16F I think I want to be a bi M", "label": 1},
        {"text": "I am exploring my sexual identity", "label": 1},
        {"text": "I have a desire to explore new options", "label": 1},
        {"text": "I live in New Mexico.", "label": 1},
        {"text": "I live in the Southwest of the United States.", "label": 1},
        {"text": "My father-in-law was not a great father/husband, even my own father was not a great husband (lots of resentment spanning decades), but I digress.", "label": 1},
    ]
    
    # For span detection, we need token-level annotations
    mock_span_data = [
        {
            "tokens": ["I", "am", "a", "23-year-old", "who", "is", "currently", "going", "through", "the", "last", "leg", "of", "undergraduate", "school", "."],
            "tags": ["O", "O", "O", "B-AGE", "O", "O", "O", "O", "O", "O", "O", "O", "O", "B-EDUCATION", "I-EDUCATION", "O"]
        },
        {
            "tokens": ["There", "is", "a", "joke", "in", "the", "design", "industry", "about", "that", "."],
            "tags": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]
        },
        {
            "tokens": ["My", "husband", "and", "I", "live", "in", "US", "."],
            "tags": ["O", "B-RELATIONSHIP_STATUS", "O", "O", "O", "O", "B-LOCATION", "O"]
        }
    ]
    
    if task == "classification":
        # Prepare classification data
        encodings = []
        labels = []
        
        for item in mock_data:
            encoding = tokenizer(
                item["text"],
                truncation=True,
                max_length=128,
                padding="max_length",
                return_tensors="pt"
            )
            
            encodings.append({k: v.squeeze(0) for k, v in encoding.items()})
            labels.append(item["label"])
        
        # Create tensor datasets
        input_ids = torch.stack([item["input_ids"] for item in encodings])
        attention_mask = torch.stack([item["attention_mask"] for item in encodings])
        labels = torch.tensor(labels, dtype=torch.long)
        
        dataset = TensorDataset(input_ids, attention_mask, labels)
    
    else:  # span detection
        # Prepare span detection data
        encodings = []
        all_labels = []
        
        for item in mock_span_data:
            # Convert IOB2 tags to binary labels
            binary_labels = [1 if tag.startswith(("B-", "I-")) else 0 for tag in item["tags"]]
            
            encoding = tokenizer(
                item["tokens"],
                is_split_into_words=True,
                truncation=True,
                max_length=128,
                padding="max_length",
                return_tensors="pt"
            )
            
            # Align labels with wordpiece tokens
            word_ids = encoding.word_ids(batch_index=0)
            aligned_labels = []
            
            for word_id in word_ids:
                if word_id is None:
                    aligned_labels.append(-100)
                else:
                    if word_id < len(binary_labels):
                        aligned_labels.append(binary_labels[word_id])
                    else:
                        aligned_labels.append(-100)
            
            encodings.append({k: v.squeeze(0) for k, v in encoding.items()})
            all_labels.append(torch.tensor(aligned_labels, dtype=torch.long))
        
        # Create tensor datasets
        input_ids = torch.stack([item["input_ids"] for item in encodings])
        attention_mask = torch.stack([item["attention_mask"] for item in encodings])
        labels = torch.stack(all_labels)
        
        dataset = TensorDataset(input_ids, attention_mask, labels)
    
    # Split into train/val/test (60/20/20)
    total_size = len(dataset)
    train_size = int(0.6 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_dataloader, val_dataloader, test_dataloader
