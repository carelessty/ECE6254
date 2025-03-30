"""
Custom model implementation for self-disclosure detection.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    RobertaConfig,
    RobertaModel,
    RobertaPreTrainedModel,
)

logger = logging.getLogger(__name__)

class RobertaForSelfDisclosureDetection(RobertaPreTrainedModel):
    """
    RoBERTa model for self-disclosure detection as a token classification task.
    
    This model is based on RoBERTa and adds a token classification head on top
    for detecting self-disclosures in text.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        
        # RoBERTa model
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        
        # Classification head
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # Initialize weights
        self.init_weights()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            position_ids: Position IDs
            head_mask: Head mask
            inputs_embeds: Input embeddings
            labels: Labels for computing the token classification loss
            output_attentions: Whether to return attentions
            output_hidden_states: Whether to return hidden states
            return_dict: Whether to return a dict or tuple
            
        Returns:
            Token classification outputs
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get RoBERTa outputs
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # Get sequence outputs
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        
        # Get logits
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
        }

def create_model_config(
    model_name_or_path: str = "roberta-large",
    num_labels: int = 0,
    label_list: Optional[List[str]] = None,
    hidden_dropout_prob: float = 0.1,
    attention_probs_dropout_prob: float = 0.1,
    classifier_dropout: Optional[float] = None,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False
) -> RobertaConfig:
    """
    Create a RoBERTa configuration for self-disclosure detection.
    
    Args:
        model_name_or_path: Name or path of the pre-trained model
        num_labels: Number of labels for classification
        label_list: List of labels
        hidden_dropout_prob: Hidden dropout probability
        attention_probs_dropout_prob: Attention dropout probability
        classifier_dropout: Classifier dropout probability
        cache_dir: Directory to cache the model
        local_files_only: Whether to use only local files
        
    Returns:
        RoBERTa configuration
    """
    if label_list is not None:
        num_labels = len(label_list)
    
    # Load config
    config = RobertaConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        id2label={i: label for i, label in enumerate(label_list)} if label_list else None,
        label2id={label: i for i, label in enumerate(label_list)} if label_list else None,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        classifier_dropout=classifier_dropout,
        cache_dir=cache_dir,
        local_files_only=local_files_only
    )
    
    return config
