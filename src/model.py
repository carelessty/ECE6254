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
from transformers.modeling_outputs import TokenClassifierOutput

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
        
        # Classification head with optimized memory usage
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
        
        # Get RoBERTa outputs with memory optimization options
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
                # More efficient masking with vectorized operations
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                
                # Skip computation if no active labels
                if active_labels.numel() > 0:
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return TokenClassifierOutput(
                    loss=loss,
                    logits=logits,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
            )

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
    
    # Load config with performance optimizations
    config = RobertaConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        id2label={i: label for i, label in enumerate(label_list)} if label_list else None,
        label2id={label: i for i, label in enumerate(label_list)} if label_list else None,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        classifier_dropout=classifier_dropout,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        # Additional optimization settings
        gradient_checkpointing=True,  # Default to gradient checkpointing
        use_cache=False  # Disable KV caching during training
    )
    
    return config


class NeoBERTForSelfDisclosureDetection(RobertaPreTrainedModel):
    """
    NeoBERT model for self-disclosure detection as a token classification task.
    This model is an enhanced version of RoBERTa with additional components
    specifically designed for self-disclosure detection in text.
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        # RoBERTa model
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        # Enhanced features for NeoBERT
        self.feature_enhancement = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size)
        )
        # Self-attention mechanism for contextual understanding
        num_heads = config.neobert_attention_heads if hasattr(config, "neobert_attention_heads") else 8
        self.self_attention = nn.MultiheadAttention(
        embed_dim=config.hidden_size,
        num_heads=num_heads,
        dropout=config.attention_probs_dropout_prob
)
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
        Forward pass of the NeoBERT model.
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
        sequence_output = outputs[0]  # [batch_size, seq_len, hidden_size]
        # Apply feature enhancement
        enhanced_features = self.feature_enhancement(sequence_output)
        # Apply self-attention for contextual dependency modeling
        # Need to transpose for MultiheadAttention (seq_len, batch_size, hidden_size)
        enhanced_features_t = enhanced_features.transpose(0, 1)
        # Create attention mask for padding tokens
        if attention_mask is not None:
            # Convert to boolean mask where 0s (padding) become True (to be masked)
            # and 1s (tokens) become False (not masked)
            attention_padding_mask = attention_mask.eq(0)
        else:
            attention_padding_mask = None
        attended_output, _ = self.self_attention(
            enhanced_features_t,
            enhanced_features_t,
            enhanced_features_t,
            key_padding_mask=attention_padding_mask
        )
        # Transpose back to [batch_size, seq_len, hidden_size]
        attended_output = attended_output.transpose(0, 1)
        # Residual connection
        sequence_output = sequence_output + attended_output
        # Apply dropout
        sequence_output = self.dropout(sequence_output)
        # Get logits
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                if active_labels.numel() > 0:
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
def create_neobert_config(
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
    Create a NeoBERT configuration for self-disclosure detection.
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
        RoBERTa configuration with NeoBERT settings
    """
    if label_list is not None:
        num_labels = len(label_list)
    # Load config with performance optimizations
    config = RobertaConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        id2label={i: label for i, label in enumerate(label_list)} if label_list else None,
        label2id={label: i for i, label in enumerate(label_list)} if label_list else None,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        classifier_dropout=classifier_dropout,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        # Additional optimization settings
        gradient_checkpointing=True,
        use_cache=False
    )
    # Add NeoBERT specific config parameters if needed
    config.neobert_attention_heads = 8
    config.neobert_feature_enhancement = True
    return config
    