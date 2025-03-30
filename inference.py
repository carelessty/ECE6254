"""
Inference script for using the fine-tuned RoBERTa model for self-disclosure detection.
"""

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline
)

from src.data_utils import extract_spans

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned self-disclosure detection model")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model directory",
    )
    parser.add_argument(
        "--input_text",
        type=str,
        default=None,
        help="Text to analyze for self-disclosure",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Path to file containing text to analyze",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save the results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="Device to run inference on (-1 for CPU, 0+ for GPU)",
    )
    
    args = parser.parse_args()
    return args

def load_model_and_tokenizer(model_path: str):
    """
    Load the fine-tuned model and tokenizer.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model and tokenizer from {model_path}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    
    return model, tokenizer

def create_token_classifier(model, tokenizer, device: int = -1):
    """
    Create a token classification pipeline.
    
    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
        device: Device to run inference on (-1 for CPU, 0+ for GPU)
        
    Returns:
        Token classification pipeline
    """
    return pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        aggregation_strategy="simple"  # Group entities
    )

def preprocess_text(text: str) -> List[str]:
    """
    Preprocess text for inference by splitting into sentences.
    
    Args:
        text: Text to preprocess
        
    Returns:
        List of sentences
    """
    # Simple sentence splitting by punctuation
    sentences = []
    for sent in text.replace('\n', ' ').split('.'):
        sent = sent.strip()
        if sent:
            sentences.append(sent + '.')
    
    return sentences if sentences else [text]

def process_text(classifier, text: str):
    """
    Process text and identify self-disclosures.
    
    Args:
        classifier: Token classification pipeline
        text: Text to analyze
        
    Returns:
        List of identified self-disclosures
    """
    # Preprocess text into sentences
    sentences = preprocess_text(text)
    
    # Run inference on each sentence
    all_disclosures = []
    current_offset = 0
    
    for sentence in sentences:
        # Run inference
        result = classifier(sentence)
        
        # Process results
        for entity in result:
            if entity["entity_group"] != "O":
                disclosure = {
                    "text": entity["word"],
                    "start": current_offset + entity["start"],
                    "end": current_offset + entity["end"],
                    "type": entity["entity_group"],
                    "score": entity["score"]
                }
                all_disclosures.append(disclosure)
        
        # Update offset for next sentence
        current_offset += len(sentence)
    
    return all_disclosures

def format_results(text: str, disclosures: List[Dict]):
    """
    Format results for display.
    
    Args:
        text: Original text
        disclosures: List of identified self-disclosures
        
    Returns:
        Formatted string with results
    """
    if not disclosures:
        return f"Text: {text}\nNo self-disclosures detected."
    
    # Sort disclosures by start position
    disclosures = sorted(disclosures, key=lambda x: x["start"])
    
    # Format results
    result = f"Text: {text}\n\nSelf-disclosures detected:\n"
    for i, disclosure in enumerate(disclosures, 1):
        result += f"{i}. Type: {disclosure['type']}\n"
        result += f"   Text: \"{disclosure['text']}\"\n"
        result += f"   Position: {disclosure['start']}-{disclosure['end']}\n"
        result += f"   Confidence: {disclosure['score']:.4f}\n\n"
    
    return result

def main():
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Check input
    if args.input_text is None and args.input_file is None:
        logger.error("Either --input_text or --input_file must be provided.")
        sys.exit(1)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    
    # Create token classifier
    classifier = create_token_classifier(model, tokenizer, args.device)
    
    # Process input
    if args.input_text:
        # Process single text
        logger.info("Processing input text...")
        disclosures = process_text(classifier, args.input_text)
        formatted_result = format_results(args.input_text, disclosures)
        
        # Print or save results
        if args.output_file:
            with open(args.output_file, "w", encoding="utf-8") as f:
                f.write(formatted_result)
            logger.info(f"Results saved to {args.output_file}")
        else:
            print("\n" + formatted_result)
    
    elif args.input_file:
        # Process file
        logger.info(f"Processing texts from {args.input_file}...")
        with open(args.input_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        
        all_results = []
        for i, text in enumerate(texts, 1):
            logger.info(f"Processing text {i}/{len(texts)}...")
            disclosures = process_text(classifier, text)
            formatted_result = format_results(text, disclosures)
            all_results.append(formatted_result)
        
        # Join results
        combined_results = "\n" + "="*80 + "\n\n".join(all_results)
        
        # Print or save results
        if args.output_file:
            with open(args.output_file, "w", encoding="utf-8") as f:
                f.write(combined_results)
            logger.info(f"Results saved to {args.output_file}")
        else:
            print(combined_results)
    
    logger.info("Done!")

if __name__ == "__main__":
    main()
