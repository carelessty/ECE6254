"""
Training script for fine-tuning RoBERTa-large on the Reddit self-disclosure dataset.
"""

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional
from peft import get_peft_model, LoraConfig, TaskType

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset, DatasetDict, load_from_disk
from transformers import (
    Trainer,
    TrainingArguments,
    set_seed,
    TrainerCallback,
    AutoTokenizer,
    AutoModelForTokenClassification,
)
# Import seqeval scoring functions directly for use in compute_metrics_fn
from seqeval.metrics import precision_score, recall_score, f1_score

from src.data_utils import (
    load_reddit_self_disclosure_dataset,
    prepare_dataset_for_training,
    get_labels,
    compute_metrics,
    compute_partial_span_f1,
)
from src.model_utils import get_model_and_tokenizer
# Import our custom NeoBERT implementation
from src.model import NeoBERTForTokenClassification

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa-large for self-disclosure detection")
    
    # Dataset arguments
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
        help="Directory to cache the dataset and models",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing preprocessed dataset",
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help="Proportion of data to use for training (if dataset needs splitting)",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Proportion of data to use for validation (if dataset needs splitting)",
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="roberta-large",
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save the model",
    )
    
    # Training arguments
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=128,
        help="Batch size for training",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=128,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before backward pass",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use mixed precision",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Whether to use bfloat16 mixed precision",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--gradient_checkpointing", 
        action="store_true",
        help="Enable gradient checkpointing to save memory"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Run evaluation every X steps",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X steps",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Log every X steps",
    )
    # Argument to specify model type
    parser.add_argument(
        "--model_type",
        type=str,
        default="roberta",
        choices=["roberta", "neobert"], # Allow specifying model type
        help="Model type to load ('roberta' or 'neobert')",
    )
    
    args = parser.parse_args()
    return args

def main():
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Prevent using both FP16 and BF16 at the same time, which can cause issues
    if args.fp16 and args.bf16:
        logger.warning("Both fp16 and bf16 were enabled. Disabling bf16.")
        args.bf16 = False
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_reddit_self_disclosure_dataset(
        token=args.hf_token,
        cache_dir=args.cache_dir,
        data_dir=args.data_dir
    )
    logger.info(f"Dataset loaded. Splits: {list(dataset.keys())}")
    
    # Get label list
    label_list = get_labels(dataset["train"])
    num_labels = len(label_list)
    logger.info(f"Number of labels: {num_labels}")
    
    # --- Model and Tokenizer Loading --- 
    logger.info(f"Loading model type '{args.model_type}' from {args.model_name_or_path}...")
    
    # Create label mappings
    id2label={i: l for i, l in enumerate(label_list)}
    label2id={l: i for i, l in enumerate(label_list)}
    
    if args.model_type.lower() == "neobert":
        # Use our custom NeoBERTForTokenClassification wrapper
        logger.info("Loading NeoBERT using custom NeoBERTForTokenClassification wrapper")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, 
            cache_dir=args.cache_dir,
            use_fast=True,
            add_prefix_space=True, # Good practice for RoBERTa-like tokenizers
            trust_remote_code=True
        )
        model = NeoBERTForTokenClassification.from_pretrained(
            args.model_name_or_path,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            cache_dir=args.cache_dir,
        )
        logger.info("NeoBERT model and tokenizer loaded.")
    
    else: # Default to standard RoBERTa
        logger.info("Loading standard RoBERTa model for Token Classification.")
        model, tokenizer = get_model_and_tokenizer(
             model_name_or_path=args.model_name_or_path,
             num_labels=num_labels,
             label_list=label_list, # Pass label_list here
             cache_dir=args.cache_dir,
        )
        logger.info("Standard RoBERTa model and tokenizer loaded.")

    logger.info(f"Model Class: {type(model).__name__}")
    logger.info(f"Tokenizer Class: {type(tokenizer).__name__}")
    
    # --- PEFT (LoRA) Configuration --- 
    logger.info("Configuring LoRA...")
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        logger.info("Enabling gradient checkpointing")
        try:
            model.gradient_checkpointing_enable()
        except Exception as e:
            logger.warning(f"Could not enable gradient checkpointing: {e}")
            logger.warning("This may affect memory usage but training will continue.")
    
    # Determine target modules based on model type
    if args.model_type.lower() == "neobert":
        # For NeoBERT, we need to find target modules by examining the model structure
        logger.info("Examining NeoBERT structure to determine target modules for LoRA...")
        
        # Find Linear layers to use as target modules
        linear_modules = []
        encoder_modules = {}  # 用于按层分组的字典
        
        # 首先收集所有线性层
        for name, module in model.named_modules():
            # Only use full module paths with parameters
            if isinstance(module, nn.Linear) and hasattr(module, 'weight'):
                linear_modules.append(name)
                if len(linear_modules) <= 10:  # Print a few for debugging
                    logger.info(f"Found linear module: {name}")
                
                # 识别层号和模块类型
                if "transformer_encoder" in name:
                    parts = name.split(".")
                    for i, part in enumerate(parts):
                        if part == "transformer_encoder" and i+1 < len(parts) and parts[i+1].isdigit():
                            layer_num = int(parts[i+1])
                            if layer_num not in encoder_modules:
                                encoder_modules[layer_num] = []
                            encoder_modules[layer_num].append(name)
        
        # 按层号分析每个encoder层的组件
        logger.info(f"Found modules in {len(encoder_modules)} encoder layers")
        
        # 确定应用LoRA的层数
        num_layers = len(encoder_modules)
        # 选择要应用LoRA的层数（可以是全部或部分层）
        target_layer_count = num_layers  # 默认全部应用
        
        # 如果层数超过10，可以选择一部分重要的层（例如前面几层、后面几层或跳跃选择）
        if num_layers > 10:
            # 选项1：仅应用于前后各几层（通常最重要）
            # target_layers = list(range(3)) + list(range(num_layers-3, num_layers))
            
            # 选项2：均匀选择层
            target_layers = sorted(set([0, num_layers-1] + [i for i in range(num_layers) if i % 3 == 0]))
            target_layer_count = len(target_layers)
            logger.info(f"Applying LoRA to {target_layer_count} of {num_layers} layers: {target_layers}")
        else:
            # 层数不多时应用到所有层
            target_layers = list(range(num_layers))
            logger.info(f"Applying LoRA to all {num_layers} encoder layers")
        
        # 确定每层中应用LoRA的模块类型
        target_modules = []
        
        # 分析模块类型并选择应用LoRA的模块
        module_types = set()
        for layer_modules in encoder_modules.values():
            for module in layer_modules:
                if "qkv" in module:
                    module_types.add("qkv")
                elif "wo" in module:
                    module_types.add("wo") 
                elif "ffn.w12" in module:
                    module_types.add("ffn.w12")
                elif "ffn.w3" in module:
                    module_types.add("ffn.w3")
        
        logger.info(f"Found module types: {module_types}")
        
        # 为选定的层构建目标模块列表
        for layer_num in target_layers:
            if layer_num in encoder_modules:
                layer_modules = encoder_modules[layer_num]
                # 优先选择注意力相关的模块
                qkv_modules = [m for m in layer_modules if "qkv" in m]
                wo_modules = [m for m in layer_modules if "wo" in m]
                ffn_modules = [m for m in layer_modules if "ffn" in m]
                
                # 添加注意力模块（QKV和WO投影）
                target_modules.extend(qkv_modules)
                target_modules.extend(wo_modules)
                
                # 可选：添加前馈网络模块（取决于计算资源和需求）
                if len(target_modules) < 30:  # 限制总模块数以避免内存问题
                    target_modules.extend(ffn_modules)
        
        if not target_modules:
            # 如果通过层分析没有找到模块，回退到简单的模式匹配
            logger.warning("Could not find modules through layer analysis. Falling back to pattern matching.")
            target_modules = [name for name in linear_modules if any(
                pattern in name for pattern in ['qkv', 'wo', 'ffn.w12', 'ffn.w3'])]
            # 限制模块数量
            if len(target_modules) > 30:
                target_modules = target_modules[:30]
        
        logger.info(f"Selected {len(target_modules)} modules for LoRA fine-tuning")
        logger.info(f"LoRA target modules: {target_modules}")
    else:
        # For RoBERTa, use the standard query/key/value names
        target_modules = ["query", "key", "value"]
        logger.info("Using standard attention module names for RoBERTa")

    # 根据选择的目标模块数量调整LoRA配置
    lora_r = 16  # 默认rank
    lora_alpha = 32  # 默认alpha
    
    # 如果模块太多，可以降低rank以节省内存
    if len(target_modules) > 20:
        lora_r = 8
        lora_alpha = 16
        logger.info(f"Reducing LoRA rank to {lora_r} due to high number of target modules")
    
    # Configure LoRA with the determined target modules
    lora_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        inference_mode=False,
        r=lora_r,                   # Rank of the update matrices
        lora_alpha=lora_alpha,      # Alpha scaling factor
        lora_dropout=0.1,           # Dropout probability for LoRA layers
        target_modules=target_modules,
    )

    # Apply LoRA to model
    logger.info("Applying LoRA adaptation to model")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Log trainable parameters %
    
    # Prepare dataset for training efficiently
    logger.info("Preparing dataset for training...")
    processed_dataset = prepare_dataset_for_training(
        dataset=dataset,
        tokenizer=tokenizer,
        max_length=512,
        label_all_tokens=False
    )
    
    # Log dataset sizes
    logger.info(f"Train dataset size: {len(processed_dataset['train'])}")
    logger.info(f"Validation dataset size: {len(processed_dataset['validation'])}")
    logger.info(f"Test dataset size: {len(processed_dataset['test'])}")
    
    # --- Undersample 'O'-only sentences in the training set --- 
    logger.info("Analyzing training set for undersampling...")
    train_dataset = processed_dataset["train"]
    entity_indices = []
    only_O_indices = []

    # Import random here, only needed if undersampling occurs
    import random 
    
    for i, example in enumerate(train_dataset):
        has_entity = False
        # Check labels, ignore padding -100
        for label_id in example['labels']:
            if label_id != -100:
                # Use label_list which should be available in this scope
                label_str = label_list[label_id] 
                if label_str != 'O':
                    has_entity = True
                    break
        if has_entity:
            entity_indices.append(i)
        else:
            # Only append if it's not padding-only (though unlikely after processing)
            if any(l != -100 for l in example['labels']):
                 only_O_indices.append(i)
    
    n_entity = len(entity_indices)
    n_only_O = len(only_O_indices)
    logger.info(f"Found {n_entity} sentences with entities and {n_only_O} sentences with only 'O'.")

    # Assign the dataset to be used for training
    training_dataset_to_use = train_dataset # Default to original

    if n_only_O > n_entity and n_entity > 0: # Only undersample if O-only is majority and entities exist
        logger.info(f"Undersampling 'O'-only sentences from {n_only_O} down to {n_entity}.")
        # Randomly select n_entity indices from the only_O_indices
        random.seed(args.seed) # Ensure reproducibility
        sampled_O_indices = random.sample(only_O_indices, n_entity)
        
        # Combine indices and select the balanced dataset
        balanced_indices = entity_indices + sampled_O_indices
        balanced_train_dataset = train_dataset.select(balanced_indices)
        
        # Shuffle the balanced dataset
        balanced_train_dataset = balanced_train_dataset.shuffle(seed=args.seed)
        logger.info(f"Created balanced training dataset with {len(balanced_train_dataset)} examples.")
        training_dataset_to_use = balanced_train_dataset # Use the balanced one

    elif n_entity == 0:
        logger.warning("No sentences with entities found in the training set. Cannot balance. Using original training set.")
        # training_dataset_to_use remains train_dataset
    else:
        logger.info("No undersampling needed (O-only sentences are not the majority or no entities found). Using original training set.")
        # training_dataset_to_use remains train_dataset

    # ----------------------------------------------------------
    
    # Configure training arguments AFTER dataset preparation/undersampling
    logger.info("Configuring training arguments...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        save_strategy="steps",
        eval_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=f"{args.output_dir}/logs",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_total_limit=2,  # Only keep the 2 best checkpoints
        overwrite_output_dir=True,
        dataloader_num_workers=4,  # Reduced workers for less memory pressure
        group_by_length=True,  # Group similar length sequences for efficiency
        report_to="tensorboard",
        label_names=["labels"],
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        seed=args.seed,
        # Memory optimization
        optim="adamw_torch",  # Use torch implementation for better memory usage
        ddp_find_unused_parameters=False,  # Improve DDP efficiency
        torch_compile=False,  # Disable torch compile to avoid memory spikes
        gradient_checkpointing=args.gradient_checkpointing,
        # Add max_grad_norm to handle gradient explosions
        max_grad_norm=1.0,
        # Restrict evaluation size to prevent OOMs during validation
        eval_accumulation_steps=8,  # Accumulate gradients during eval to reduce memory
        # Set eval batch size reduction to use less memory during evaluation
    )

    # Create a smaller validation dataset if the evaluation set is too large
    # This helps prevent OOM during validation
    eval_dataset = processed_dataset["validation"]
    if len(eval_dataset) > 1000:  # If validation set is large
        logger.info(f"Creating smaller validation dataset for frequent evaluation")
        # Use a stratified subset for validation to maintain distribution
        eval_indices = np.random.choice(
            len(eval_dataset), 
            size=min(1000, len(eval_dataset)), 
            replace=False
        )
        validation_subset = eval_dataset.select(eval_indices)
        logger.info(f"Using validation subset of size {len(validation_subset)} for training")
    else:
        validation_subset = eval_dataset
    
    # Define compute_metrics_fn and helper functions BEFORE Trainer initialization
    def cleanup_memory():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        import gc
        gc.collect()

    def compute_metrics_fn(eval_preds):
        logger.info("--- Inside compute_metrics_fn ---") # DEBUG PRINT
        preds, labels = eval_preds
        logger.info(f"Raw preds shape: {preds.shape}, type: {preds.dtype}") # DEBUG PRINT
        logger.info(f"Raw labels shape: {labels.shape}, type: {labels.dtype}") # DEBUG PRINT

        # 安全地检查预测结果是否有效
        if preds is None or labels is None:
            logger.error("Received None for predictions or labels")
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "partial_span_precision": 0.0,
                "partial_span_recall": 0.0,
                "partial_span_f1": 0.0
            }
            
        # Get predicted IDs
        try:
            preds_ids = np.argmax(preds, axis=2)
            logger.info(f"Argmaxed preds_ids shape: {preds_ids.shape}") # DEBUG PRINT
        except Exception as e:
            logger.error(f"Error during argmax: {e}")
            # 返回默认值
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "partial_span_precision": 0.0,
                "partial_span_recall": 0.0,
                "partial_span_f1": 0.0
            }

        # Convert to label sequences, ignoring -100
        true_predictions_list = []
        true_labels_list = []
        
        try:
            for i in range(len(preds_ids)):
                prediction = preds_ids[i]
                label = labels[i]
                true_prediction_sequence = []
                true_label_sequence = []
                for pred_id, label_id in zip(prediction, label):
                    if label_id != -100:
                        true_prediction_sequence.append(label_list[pred_id])
                        true_label_sequence.append(label_list[label_id])
                true_predictions_list.append(true_prediction_sequence)
                true_labels_list.append(true_label_sequence)

            # DEBUG PRINT: Print first example's labels and predictions
            if len(true_labels_list) > 0 and len(true_predictions_list) > 0:
                logger.info(f"Sample True Labels (first example, len={len(true_labels_list[0])}): {true_labels_list[0][:50]}...")
                logger.info(f"Sample Pred Labels (first example, len={len(true_predictions_list[0])}): {true_predictions_list[0][:50]}...")
                is_pred_all_O = all(p == 'O' for p in true_predictions_list[0])
                num_non_O_true = sum(t != 'O' for t in true_labels_list[0]) # Added count
                num_non_O_pred = sum(p != 'O' for p in true_predictions_list[0]) # Added count
                logger.info(f"First sample analysis: True non-O count={num_non_O_true}, Pred non-O count={num_non_O_pred}, Is prediction all 'O'? {is_pred_all_O}")
            else:
                logger.warning("Could not get sample labels/predictions, lists might be empty.")
        except Exception as e:
            logger.error(f"Error during sequence conversion: {e}", exc_info=True)
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "partial_span_precision": 0.0,
                "partial_span_recall": 0.0,
                "partial_span_f1": 0.0
            }

        # Calculate metrics using seqeval functions
        try:
            precision = precision_score(true_labels_list, true_predictions_list, zero_division=0)
            recall = recall_score(true_labels_list, true_predictions_list, zero_division=0)
            f1 = f1_score(true_labels_list, true_predictions_list, zero_division=0)
            logger.info(f"Seqeval results: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}") # DEBUG PRINT
        except Exception as e:
            logger.error(f"Error during seqeval calculation: {e}", exc_info=True)
            precision, recall, f1 = 0.0, 0.0, 0.0

        metrics = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        # Compute partial span F1 separately
        try:
            partial_metrics = compute_partial_span_f1(
                preds_ids, 
                labels,
                label_list
            )
            logger.info(f"Partial Span F1 results: {partial_metrics}") # DEBUG PRINT
        except Exception as e:
            logger.error(f"Error during partial span F1 calculation: {e}", exc_info=True)
            partial_metrics = {"partial_span_precision": 0.0, "partial_span_recall": 0.0, "partial_span_f1": 0.0}
        
        metrics.update(partial_metrics)
        
        logger.info("--- Exiting compute_metrics_fn ---") # DEBUG PRINT
        cleanup_memory()
        
        return metrics
    
    # Define MemoryCleanupCallback BEFORE Trainer initialization
    class MemoryCleanupCallback(TrainerCallback):
        """Callback to clean up memory after evaluation."""
        
        def on_evaluate(self, args, state, control, **kwargs):
            """Called after evaluation."""
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            import gc
            gc.collect()
            return control
        
        def on_prediction_step(self, args, state, control, **kwargs):
            """Called after each prediction step."""
            if torch.cuda.is_available() and state.global_step % 10 == 0:
                torch.cuda.empty_cache()
            return control
        
        def on_save(self, args, state, control, **kwargs):
            """Called after model saving."""
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            import gc
            gc.collect()
            return control

    # 自定义Trainer类，添加更多的错误处理和日志记录
    class CustomTrainer(Trainer):
        """自定义Trainer以增强与自定义模型的兼容性"""
        
        def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
            """重写评估循环以处理NoneType输出的边缘情况"""
            try:
                # 调用原始实现
                return super().evaluation_loop(
                    dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
                )
            except TypeError as e:
                # 处理NoneType错误
                logger.error(f"TypeError in evaluation_loop: {e}")
                logger.warning("返回空评估结果，跳过当前评估步骤")
                
                # 返回一个最小可行的评估结果以避免训练中断
                return {
                    f"{metric_key_prefix}_loss": 0.0,
                    f"{metric_key_prefix}_precision": 0.0,
                    f"{metric_key_prefix}_recall": 0.0,
                    f"{metric_key_prefix}_f1": 0.0,
                    f"{metric_key_prefix}_runtime": 0.0,
                    f"{metric_key_prefix}_samples_per_second": 0.0,
                    f"{metric_key_prefix}_steps_per_second": 0.0,
                }, [], []

    # 使用自定义Trainer替换标准Trainer
    # trainer = Trainer(...)
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        # Use the potentially balanced training dataset
        train_dataset=training_dataset_to_use,
        eval_dataset=validation_subset,  # Use smaller validation set for eval
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_fn,
        callbacks=[
            MemoryCleanupCallback() # Add the callback instance
        ],
    )
    
    # Train the model
    logger.info("Training the model...")
    trainer.train()
    
    # Evaluate the model
    logger.info("Evaluating the model...")
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_results}")
    
    # Save evaluation results
    with open(os.path.join(args.output_dir, "eval_results.txt"), "w") as f:
        for key, value in eval_results.items():
            f.write(f"{key} = {value}\n")
    
    # Save the model
    logger.info(f"Saving the model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Test the model
    logger.info("Testing the model...")
    test_results = trainer.evaluate(processed_dataset["test"])
    logger.info(f"Test results: {test_results}")
    
    # Save test results
    with open(os.path.join(args.output_dir, "test_results.txt"), "w") as f:
        for key, value in test_results.items():
            f.write(f"{key} = {value}\n")
    
    logger.info("Done!")

if __name__ == "__main__":
    main()
