"""
Custom model implementation for self-disclosure detection.
"""

import logging
import os
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


class NeoBERTForTokenClassification(nn.Module):
    """
    Custom wrapper for NeoBERT that adds a token classification head.
    
    Since NeoBERT is only available as a base model through AutoModel,
    we need to manually add a classification head for token classification.
    """
    
    def __init__(self, model_name_or_path, num_labels, id2label=None, label2id=None, cache_dir=None):
        super().__init__()
        from transformers import AutoModel, AutoConfig
        
        # Load the base NeoBERT model
        self.config = AutoConfig.from_pretrained(
            model_name_or_path, 
            trust_remote_code=True,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            cache_dir=cache_dir
        )
        
        # Store num_labels for later use
        self.num_labels = num_labels
        
        # Load the base model
        self.neobert = AutoModel.from_pretrained(
            model_name_or_path,
            config=self.config,
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        
        # Get hidden size from config
        self.hidden_size = self.config.hidden_size
        
        # Add classification head
        # Check if hidden_dropout_prob exists, otherwise use default value
        dropout_prob = 0.1  # Default dropout probability
        if hasattr(self.config, 'hidden_dropout_prob'):
            dropout_prob = self.config.hidden_dropout_prob
        
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        
        # Initialize weights for the classifier
        self._init_weights(self.classifier)
    
    def _init_weights(self, module):
        """Initialize the weights - copied from similar transformer models."""
        if isinstance(module, nn.Linear):
            # Slightly different initialization from PyTorch's default
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
    
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
        Forward pass for token classification with NeoBERT.
        
        Args match the standard interface of transformer models for token classification.
        """
        # Set default for return_dict if not provided
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Forward pass through NeoBERT base model
        outputs = self.neobert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # Get sequence output and apply dropout
        sequence_output = outputs[0]  # [batch_size, seq_len, hidden_size]
        sequence_output = self.dropout(sequence_output)
        
        # Apply classification head
        logits = self.classifier(sequence_output)  # [batch_size, seq_len, num_labels]
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            
            # Only keep active parts of the loss (non-padding)
            if attention_mask is not None:
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
        
        # Prepare return values using TokenClassifierOutput
        if not return_dict:
            # 对于非return_dict模式，保持一致性返回元组
            output = (logits,)
            # 小心处理其他输出，确保它们存在
            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                output = output + (outputs.hidden_states,)
            else:
                output = output + (None,)
                
            if hasattr(outputs, "attentions") and outputs.attentions is not None:
                output = output + (outputs.attentions,)
            else:
                output = output + (None,)
                
            return ((loss,) + output) if loss is not None else output
        
        # 创建一个与TokenClassifierOutput兼容的字典
        # 确保hidden_states和attentions始终有值，如果没有则设为None
        hidden_states = outputs.hidden_states if hasattr(outputs, "hidden_states") else None
        attentions = outputs.attentions if hasattr(outputs, "attentions") else None
        
        # 直接返回TokenClassifierOutput对象而不是字典
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            attentions=attentions
        )
    
    def gradient_checkpointing_enable(self, *args, **kwargs):
        """Enable gradient checkpointing for efficient memory usage."""
        try:
            if hasattr(self.neobert, "gradient_checkpointing_enable"):
                # 尝试将参数传递给基础模型
                gradient_checkpointing_kwargs = kwargs.get("gradient_checkpointing_kwargs", {})
                self.neobert.gradient_checkpointing_enable(*args, **kwargs)
            else:
                # Attempt manual implementation if method not available
                logger.warning("NeoBERT implementation does not have gradient_checkpointing_enable method. "
                              "Using custom implementation.")
                if hasattr(self.neobert, "config"):
                    self.neobert.config.use_cache = False
                
                # Define custom checkpoint function
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                
                # Try to set custom checkpointing attribute
                if hasattr(self.neobert, "model"):
                    modules = self.neobert.model.encoder.layer if hasattr(self.neobert.model, "encoder") else []
                    for module in modules:
                        if hasattr(module, "forward"):
                            module._forward = module.forward
                            setattr(module, "forward", create_custom_forward(module._forward))
        except Exception as e:
            # Gracefully handle any exceptions by logging and continuing
            logger.warning(f"Failed to enable gradient checkpointing for NeoBERT: {e}")
            logger.warning("Training will continue without gradient checkpointing.")
    
    @classmethod
    def from_pretrained(cls, model_name_or_path, *args, **kwargs):
        """
        Load a pre-trained NeoBERT model and add a token classification head.
        Follows the from_pretrained pattern used in transformers.
        """
        # Extract parameters needed for initialization
        num_labels = kwargs.pop("num_labels", 2)
        id2label = kwargs.pop("id2label", None)
        label2id = kwargs.pop("label2id", None)
        cache_dir = kwargs.pop("cache_dir", None)
        
        # Initialize model
        model = cls(
            model_name_or_path=model_name_or_path,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            cache_dir=cache_dir,
        )
        
        # Return initialized model
        return model
    
    def save_pretrained(self, save_directory, *args, **kwargs):
        """
        Save the model to a directory. Mimics Hugging Face's save_pretrained.
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Save configuration
        self.config.save_pretrained(save_directory)
        
        # Save model weights - only save neobert
        self.neobert.save_pretrained(save_directory)
        
        # Save classifier separately
        torch.save(
            {
                "classifier.weight": self.classifier.weight.data,
                "classifier.bias": self.classifier.bias.data if self.classifier.bias is not None else None,
            },
            os.path.join(save_directory, "token_classifier.pt")
        )
    
    def gradient_checkpointing_disable(self, *args, **kwargs):
        """Disable gradient checkpointing."""
        try:
            if hasattr(self.neobert, "gradient_checkpointing_disable"):
                self.neobert.gradient_checkpointing_disable(*args, **kwargs)
            else:
                logger.warning("NeoBERT implementation does not have gradient_checkpointing_disable method. "
                             "Using custom implementation.")
                if hasattr(self.neobert, "config"):
                    self.neobert.config.use_cache = True
                    
                # Try to restore original forwards if they were modified
                if hasattr(self.neobert, "model"):
                    modules = self.neobert.model.encoder.layer if hasattr(self.neobert.model, "encoder") else []
                    for module in modules:
                        if hasattr(module, "_forward"):
                            setattr(module, "forward", module._forward)
                            delattr(module, "_forward")
        except Exception as e:
            logger.warning(f"Failed to disable gradient checkpointing for NeoBERT: {e}")
    