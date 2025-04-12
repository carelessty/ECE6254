"""
Inference script for token classification with NeoBERT or RoBERTa.
"""
import argparse
import logging
import sys
import os
from typing import List, Dict, Any, Optional

import torch
import numpy as np
from transformers import AutoTokenizer
from src.model import NeoBERTForTokenClassification, RobertaForSelfDisclosureDetection

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Inference with NeoBERT or RoBERTa for self-disclosure detection")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to the fine-tuned model directory"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="roberta",
        choices=["roberta", "neobert"],
        help="Model type ('roberta' or 'neobert')"
    )
    parser.add_argument(
        "--text", 
        type=str, 
        required=True,
        help="Text to analyze for self-disclosure"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cuda or cpu)"
    )
    return parser.parse_args()

def load_model_and_tokenizer(model_path: str, model_type: str, device: str):
    """
    Load model and tokenizer based on model type.
    """
    logger.info(f"Loading {model_type} model from {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        use_fast=True, 
        add_prefix_space=True,
        trust_remote_code=True if model_type == "neobert" else False
    )
    
    # Load model based on type
    if model_type == "neobert":
        model = NeoBERTForTokenClassification.from_pretrained(model_path)
    else:  # roberta
        model = RobertaForSelfDisclosureDetection.from_pretrained(model_path)
    
    # Move model to device
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    return model, tokenizer

def get_entity_spans(input_text: str, token_ids: List[int], logits: torch.Tensor, tokenizer, id2label: Dict[int, str], threshold: float = 0.5) -> List[Dict[str, Any]]:
    """
    Convert token-level predictions to entity spans.
    
    Args:
        input_text: Raw input text
        token_ids: List of token IDs
        logits: Model logits
        tokenizer: Tokenizer used for encoding
        id2label: Mapping from prediction ID to label string
        threshold: Confidence threshold for predictions
        
    Returns:
        List of entity spans with type, text, start, end, and confidence
    """
    # Convert logits to predictions
    probabilities = torch.softmax(logits, dim=-1)
    predictions = torch.argmax(logits, dim=-1)
    confidence_scores = torch.max(probabilities, dim=-1).values
    
    # Convert to numpy for easier processing
    predictions = predictions.detach().cpu().numpy()
    confidence_scores = confidence_scores.detach().cpu().numpy()
    
    entities = []
    current_entity = None
    
    # Get word ids to align tokens
    word_ids = tokenizer.encode(input_text, return_tensors="pt", add_special_tokens=True).tolist()[0]
    
    # Decode token IDs to tokens for debugging
    tokens = tokenizer.convert_ids_to_tokens(word_ids)
    
    # Print debug info
    # print("Tokens:", tokens)
    # print("Predictions:", predictions[0])
    # print("Confidence:", confidence_scores[0])
    
    # Process tokens to extract entities
    for i, (token_id, pred_id, conf) in enumerate(zip(word_ids, predictions[0], confidence_scores[0])):
        # Skip special tokens
        if token_id in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
            if current_entity:
                entities.append(current_entity)
                current_entity = None
            continue
        
        # Get label
        label = id2label[pred_id]
        
        # Skip low confidence predictions and "O" labels
        if conf < threshold or label == "O":
            if current_entity:
                entities.append(current_entity)
                current_entity = None
            continue
        
        # Process B- tags (beginning of entity)
        if label.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            
            # Get entity type
            entity_type = label[2:]
            
            # Get token text
            token_text = tokenizer.convert_ids_to_tokens(token_id)
            
            # Create new entity
            current_entity = {
                "type": entity_type,
                "text": token_text,
                "tokens": [token_text],
                "confidence": float(conf),
                "token_indexes": [i]
            }
        
        # Process I- tags (inside of entity)
        elif label.startswith("I-") and current_entity and current_entity["type"] == label[2:]:
            # Get token text
            token_text = tokenizer.convert_ids_to_tokens(token_id)
            
            # Add to current entity
            current_entity["text"] += " " + token_text
            current_entity["tokens"].append(token_text)
            current_entity["token_indexes"].append(i)
            current_entity["confidence"] = min(current_entity["confidence"], float(conf))
        
        # Handle case when I- tag doesn't follow a matching B- tag
        elif label.startswith("I-"):
            if current_entity:
                entities.append(current_entity)
            
            # Get entity type
            entity_type = label[2:]
            
            # Get token text
            token_text = tokenizer.convert_ids_to_tokens(token_id)
            
            # Create new entity (treating I- as B- when no matching B- precedes it)
            current_entity = {
                "type": entity_type,
                "text": token_text,
                "tokens": [token_text],
                "confidence": float(conf),
                "token_indexes": [i]
            }
    
    # Add the last entity if there is one
    if current_entity:
        entities.append(current_entity)
    
    # Clean up the results and convert to character offsets
    filtered_entities = []
    for entity in entities:
        # Reconstruct text from tokens (this is simplified; in practice you'd need 
        # more sophisticated alignment between tokens and original text)
        filtered_entities.append({
            "type": entity["type"],
            "text": entity["text"].replace(" ##", ""),
            "confidence": entity["confidence"],
        })
    
    return filtered_entities

def main():
    args = parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.model_type, args.device)
    
    # Tokenize input text
    inputs = tokenizer(
        args.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=args.max_length
    )
    
    # Move inputs to device
    inputs = {k: v.to(args.device) for k, v in inputs.items()}
    
    # Get model id2label mapping
    if hasattr(model, "config") and hasattr(model.config, "id2label"):
        id2label = model.config.id2label
    else:
        # Default mapping for demonstration
        id2label = {
            0: "O",
            # Add other labels as needed based on your model
        }
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract logits
    if isinstance(outputs, dict):
        logits = outputs["logits"]
    else:
        logits = outputs.logits
    
    # Process the results to extract entity spans
    entities = get_entity_spans(
        args.text, 
        inputs["input_ids"][0].tolist(), 
        logits, 
        tokenizer, 
        id2label
    )
    
    # Print results
    print("\nText:", args.text)
    print("\nDetected Self-Disclosures:")
    if not entities:
        print("No self-disclosures detected.")
    else:
        for i, entity in enumerate(entities, 1):
            print(f"{i}. Type: {entity['type']}")
            print(f"   Text: \"{entity['text']}\"")
            print(f"   Confidence: {entity['confidence']:.4f}")
            print()

if __name__ == "__main__":
    main()
