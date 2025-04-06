import logging
import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel
from typing import List, Optional
from src.model import NeoBERTForSelfDisclosureDetection, create_neobert_config

def load_neobert_model(args, num_labels, label_list=None):
    """
    Load the NeoBERT model for training or inference.
    Args:
        args: Arguments containing model configuration
        num_labels: Number of labels for classification
        label_list: List of label names
    Returns:
        NeoBERT model
    """
    logging.info(f"Loading NeoBERT model from {args.model_name_or_path}")
    # Create NeoBERT configuration
    config = create_neobert_config(
        model_name_or_path=args.model_name_or_path,
        num_labels=num_labels,
        label_list=label_list,
        hidden_dropout_prob=args.hidden_dropout_prob if hasattr(args, 'hidden_dropout_prob') else 0.1,
        attention_probs_dropout_prob=args.attention_probs_dropout_prob if hasattr(args, 'attention_probs_dropout_prob') else 0.1,
        classifier_dropout=args.classifier_dropout if hasattr(args, 'classifier_dropout') else None,
        cache_dir=args.cache_dir if hasattr(args, 'cache_dir') else None,
        local_files_only=args.local_files_only if hasattr(args, 'local_files_only') else False
    )
    # Initialize model with pretrained weights if available
    try:
        model = NeoBERTForSelfDisclosureDetection.from_pretrained(
            args.model_name_or_path,
            config=config,
            cache_dir=args.cache_dir if hasattr(args, 'cache_dir') else None,
            local_files_only=args.local_files_only if hasattr(args, 'local_files_only') else False
        )
        logging.info("Loaded pretrained weights for NeoBERT")
    except Exception as e:
        logging.warning(f"Could not load pretrained weights for NeoBERT: {e}")
        logging.info("Initializing NeoBERT with RoBERTa weights")
        # Initialize from scratch
        model = NeoBERTForSelfDisclosureDetection(config)
        # If the model_name_or_path is a pretrained RoBERTa, load those weights into the RoBERTa part
        try:
            roberta_model = RobertaModel.from_pretrained(args.model_name_or_path)
            model.roberta = roberta_model
            logging.info("Successfully loaded RoBERTa weights into NeoBERT backbone")
        except Exception as e:
            logging.warning(f"Could not load RoBERTa weights: {e}. Using random initialization.")
    return model
