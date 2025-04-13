"""
Inference script for token classification with NeoBERT or RoBERTa.
Run predictions on text using fine-tuned self-disclosure detection models.
"""

import argparse
import logging
import sys
import os
from typing import List, Dict, Any, Optional, Tuple, Union

import torch
import numpy as np
from transformers import AutoTokenizer
from peft import PeftModel, PeftConfig

# 导入项目所需的模块
from src.model import NeoBERTForTokenClassification, RobertaForSelfDisclosureDetection
from src.data_utils import get_labels, extract_spans

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
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
        default=None,
        help="Text to analyze for self-disclosure"
    )
    parser.add_argument(
        "--input_file", 
        type=str, 
        default=None,
        help="Input file containing text to analyze (one text per line)"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default=None,
        help="Output file to save results"
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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for processing multiple texts"
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for entity predictions"
    )
    parser.add_argument(
        "--color_output",
        action="store_true",
        help="Use colored output for console display"
    )
    return parser.parse_args()

def load_model_and_tokenizer(model_path: str, model_type: str, device: str):
    """
    Load model and tokenizer based on model type.
    
    Args:
        model_path: Path to the fine-tuned model directory
        model_type: Type of model ('roberta' or 'neobert')
        device: Device to run model on ('cuda' or 'cpu')
        
    Returns:
        Tuple of (model, tokenizer, id2label)
    """
    logger.info(f"Loading {model_type} model from {model_path}")
    
    # 检查是否是PEFT/LoRA模型
    is_peft_model = os.path.exists(os.path.join(model_path, "adapter_config.json"))
    logger.info(f"Is PEFT/LoRA model: {is_peft_model}")
    
    # 获取标签映射
    if os.path.exists(os.path.join(model_path, "label_list.txt")):
        with open(os.path.join(model_path, "label_list.txt"), "r") as f:
            label_list = [line.strip() for line in f.readlines()]
        id2label = {i: label for i, label in enumerate(label_list)}
        label2id = {label: i for i, label in enumerate(label_list)}
        logger.info(f"Loaded {len(label_list)} labels from label_list.txt")
    else:
        # 尝试从模型配置中获取
        if model_type == "neobert":
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            if hasattr(config, "id2label"):
                id2label = config.id2label
                label2id = config.label2id
            else:
                # 使用默认BIO格式标签
                from src.data_utils import get_default_labels
                label_list = get_default_labels()
                id2label = {i: label for i, label in enumerate(label_list)}
                label2id = {label: i for i, label in enumerate(label_list)}
            logger.info(f"Using {len(id2label)} labels from model config or defaults")
        else:
            # RoBERTa模型通常会在config中包含id2label
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_path)
            if hasattr(config, "id2label"):
                id2label = config.id2label
                label2id = config.label2id
            else:
                # 使用默认BIO格式标签
                from src.data_utils import get_default_labels
                label_list = get_default_labels()
                id2label = {i: label for i, label in enumerate(label_list)}
                label2id = {label: i for i, label in enumerate(label_list)}
            logger.info(f"Using {len(id2label)} labels from model config or defaults")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        use_fast=True, 
        add_prefix_space=True,
        trust_remote_code=True if model_type == "neobert" else False
    )
    
    # 根据模型类型加载基础模型
    if model_type == "neobert":
        # 检查是否有adapter目录，表明是PEFT模型
        if is_peft_model:
            logger.info("Loading NeoBERT with PEFT/LoRA adapter")
            # 首先加载基础模型
            num_labels = len(id2label)
            base_model = NeoBERTForTokenClassification.from_pretrained(
                model_path, 
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id
            )
            # 然后加载PEFT适配器
            model = PeftModel.from_pretrained(base_model, model_path)
        else:
            logger.info("Loading regular NeoBERT model")
            model = NeoBERTForTokenClassification.from_pretrained(model_path)
    else:  # roberta
        if is_peft_model:
            logger.info("Loading RoBERTa with PEFT/LoRA adapter")
            from transformers import AutoModelForTokenClassification
            # 首先加载基础模型
            base_model = AutoModelForTokenClassification.from_pretrained(
                model_path, 
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True  # 允许加载不同大小的模型（PEFT会使头部改变）
            )
            # 然后加载PEFT适配器
            model = PeftModel.from_pretrained(base_model, model_path)
        else:
            logger.info("Loading regular RoBERTa model")
            from transformers import AutoModelForTokenClassification
            model = AutoModelForTokenClassification.from_pretrained(model_path)
    
    # 移动模型到指定设备并设置为评估模式
    model = model.to(device)
    model.eval()  # 设置为评估模式
    
    # 获取或确认id2label映射
    if not hasattr(model, "config") or not hasattr(model.config, "id2label"):
        logger.warning("Model does not have id2label in its config. Using the one we created.")
    else:
        # 使用模型中的id2label，它可能更准确
        id2label = model.config.id2label
        logger.info(f"Using {len(id2label)} labels from the loaded model")
    
    return model, tokenizer, id2label

def preprocess_batch(texts: List[str], tokenizer, max_length: int = 512, device: str = "cuda"):
    """
    对一批文本进行预处理。
    
    Args:
        texts: 文本列表
        tokenizer: 用于分词的tokenizer
        max_length: 最大序列长度
        device: 运行设备
        
    Returns:
        编码后的输入，以及原始文本到token的对应信息
    """
    # 使用tokenizer对文本进行编码
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length
    )
    
    # 移动到指定设备
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    return inputs

def perform_inference(model, inputs, id2label):
    """
    对输入执行推理。
    
    Args:
        model: 模型
        inputs: 模型输入
        id2label: ID到标签的映射
        
    Returns:
        预测的logits和概率
    """
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 提取logits
    if hasattr(outputs, "logits"):
        logits = outputs.logits
    elif isinstance(outputs, tuple) and len(outputs) > 0:
        logits = outputs[0]
    elif isinstance(outputs, dict) and "logits" in outputs:
        logits = outputs["logits"]
    else:
        raise ValueError(f"Unable to extract logits from model outputs: {type(outputs)}")
    
    # 计算概率和预测
    probabilities = torch.softmax(logits, dim=-1)
    predictions = torch.argmax(logits, dim=-1)
    confidence_scores = torch.max(probabilities, dim=-1).values
    
    return predictions, confidence_scores, probabilities

def extract_entities(
    texts: List[str],
    input_ids: torch.Tensor,
    predictions: torch.Tensor, 
    confidence_scores: torch.Tensor,
    tokenizer,
    id2label: Dict[int, str],
    confidence_threshold: float = 0.5
) -> List[List[Dict[str, Any]]]:
    """
    从预测结果中提取实体。
    
    Args:
        texts: 原始文本列表
        input_ids: 输入token ID
        predictions: 模型预测的标签索引
        confidence_scores: 置信度分数
        tokenizer: 用于将token ID转换回token的tokenizer
        id2label: ID到标签的映射
        confidence_threshold: 置信度阈值
        
    Returns:
        每个文本的实体列表
    """
    batch_entities = []
    
    # 将张量转换为numpy数组以便处理
    predictions_np = predictions.cpu().numpy()
    confidence_scores_np = confidence_scores.cpu().numpy()
    input_ids_np = input_ids.cpu().numpy()
    
    # 逐文本处理
    for text_idx, (text, text_preds, text_conf, text_ids) in enumerate(
        zip(texts, predictions_np, confidence_scores_np, input_ids_np)
    ):
        text_entities = []
        current_entity = None
        
        # 获取token到原始文本的映射（简化版）
        tokens = tokenizer.convert_ids_to_tokens(text_ids)
        
        # 处理每个token
        for i, (token_id, pred_id, conf) in enumerate(zip(text_ids, text_preds, text_conf)):
            # 跳过特殊token
            if token_id in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
                if current_entity:
                    text_entities.append(current_entity)
                    current_entity = None
                continue
            
            # 获取预测的标签
            label = id2label.get(int(pred_id), "O")
            
            # 跳过低置信度预测和"O"标签
            if conf < confidence_threshold or label == "O":
                if current_entity:
                    text_entities.append(current_entity)
                    current_entity = None
                continue
            
            # 处理B-标签（实体开始）
            if label.startswith("B-"):
                if current_entity:
                    text_entities.append(current_entity)
                
                # 获取实体类型
                entity_type = label[2:]
                
                # 获取token文本
                token_text = tokenizer.convert_ids_to_tokens(token_id)
                
                # 创建新实体
                current_entity = {
                    "type": entity_type,
                    "text": token_text,
                    "tokens": [token_text],
                    "confidence": float(conf),
                    "token_indexes": [i]
                }
            
            # 处理I-标签（实体内部）
            elif label.startswith("I-") and current_entity and current_entity["type"] == label[2:]:
                # 获取token文本
                token_text = tokenizer.convert_ids_to_tokens(token_id)
                
                # 添加到当前实体
                current_entity["text"] += " " + token_text
                current_entity["tokens"].append(token_text)
                current_entity["token_indexes"].append(i)
                current_entity["confidence"] = min(current_entity["confidence"], float(conf))
            
            # 处理不跟随匹配的B-标签的I-标签
            elif label.startswith("I-"):
                if current_entity:
                    text_entities.append(current_entity)
                
                # 获取实体类型
                entity_type = label[2:]
                
                # 获取token文本
                token_text = tokenizer.convert_ids_to_tokens(token_id)
                
                # 创建新实体（将I-视为B-）
                current_entity = {
                    "type": entity_type,
                    "text": token_text,
                    "tokens": [token_text],
                    "confidence": float(conf),
                    "token_indexes": [i]
                }
        
        # 添加最后一个实体（如果有）
        if current_entity:
            text_entities.append(current_entity)
        
        # 清理结果
        cleaned_entities = []
        for entity in text_entities:
            # 清理token文本，去除特殊符号
            clean_text = entity["text"].replace(" ##", "")
            if "word_ids" in entity:
                # 如果有原始文本位置信息，可以从原始文本中提取实体文本
                start, end = entity["start"], entity["end"]
                clean_text = text[start:end]
            
            # 创建清理后的实体
            cleaned_entity = {
                "type": entity["type"],
                "text": clean_text,
                "confidence": entity["confidence"],
            }
            cleaned_entities.append(cleaned_entity)
        
        batch_entities.append(cleaned_entities)
    
    return batch_entities

def format_results(texts, batch_entities, color_output=False):
    """
    格式化结果用于显示。
    
    Args:
        texts: 原始文本列表
        batch_entities: 每个文本的实体列表
        color_output: 是否使用彩色输出
    
    Returns:
        格式化的结果字符串
    """
    # ANSI颜色代码
    colors = {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
    }
    
    # 为不同类型的实体分配颜色
    entity_colors = {}
    color_list = ["red", "green", "blue", "magenta", "cyan", "yellow"]
    
    results = []
    
    for text_idx, (text, entities) in enumerate(zip(texts, batch_entities)):
        result = []
        result.append(f"Text {text_idx + 1}: {text}")
        
        if not entities:
            result.append("No self-disclosures detected.")
        else:
            result.append("\nSelf-disclosures detected:")
            for i, entity in enumerate(entities, 1):
                entity_type = entity["type"]
                
                # 为新实体类型分配颜色
                if entity_type not in entity_colors and color_output:
                    entity_colors[entity_type] = color_list[len(entity_colors) % len(color_list)]
                
                # 添加彩色输出
                if color_output:
                    color = colors[entity_colors[entity_type]]
                    result.append(f"{i}. Type: {color}{entity_type}{colors['reset']}")
                    result.append(f"   Text: \"{color}{entity['text']}{colors['reset']}\"")
                    result.append(f"   Confidence: {color}{entity['confidence']:.4f}{colors['reset']}")
                else:
                    result.append(f"{i}. Type: {entity_type}")
                    result.append(f"   Text: \"{entity['text']}\"")
                    result.append(f"   Confidence: {entity['confidence']:.4f}")
        
        results.append("\n".join(result))
    
    return "\n\n" + "\n\n".join(results)

def run_inference_batch(
    model, 
    tokenizer, 
    id2label, 
    texts, 
    max_length, 
    device, 
    confidence_threshold,
    color_output
):
    """
    对一批文本运行推理。
    
    Args:
        model: 模型
        tokenizer: tokenizer
        id2label: ID到标签的映射
        texts: 文本列表
        max_length: 最大序列长度
        device: 运行设备
        confidence_threshold: 置信度阈值
        color_output: 是否使用彩色输出
        
    Returns:
        格式化的结果字符串
    """
    # 预处理文本
    inputs = preprocess_batch(texts, tokenizer, max_length, device)
    
    # 执行推理
    predictions, confidence_scores, _ = perform_inference(model, inputs, id2label)
    
    # 提取实体
    batch_entities = extract_entities(
        texts, 
        inputs["input_ids"],
        predictions,
        confidence_scores,
        tokenizer,
        id2label,
        confidence_threshold
    )
    
    # 格式化结果
    results = format_results(texts, batch_entities, color_output)
    
    return results, batch_entities

def main():
    """
    主函数。
    """
    # 解析命令行参数
    args = parse_args()
    
    # 检查输入
    if args.text is None and args.input_file is None:
        logger.error("Either --text or --input_file must be provided.")
        sys.exit(1)
    
    # 加载模型和tokenizer
    model, tokenizer, id2label = load_model_and_tokenizer(args.model_path, args.model_type, args.device)
    
    # 准备输入文本
    if args.input_file:
        # 从文件读取文本
        with open(args.input_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(texts)} texts from {args.input_file}")
    else:
        # 使用命令行提供的文本
        texts = [args.text]
    
    # 按批次处理文本
    all_results = []
    all_entities = []
    
    for i in range(0, len(texts), args.batch_size):
        batch_texts = texts[i:i + args.batch_size]
        logger.info(f"Processing batch {i // args.batch_size + 1}/{(len(texts) + args.batch_size - 1) // args.batch_size}")
        
        # 执行推理
        batch_results, batch_entities = run_inference_batch(
            model,
            tokenizer,
            id2label,
            batch_texts,
            args.max_length,
            args.device,
            args.confidence_threshold,
            args.color_output
        )
        
        all_results.append(batch_results)
        all_entities.extend(batch_entities)
    
    # 汇总结果
    combined_results = "\n".join(all_results)
    
    # 输出或保存结果
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            # 在文件中我们不使用彩色输出
            f.write(combined_results)
        logger.info(f"Results saved to {args.output_file}")
    else:
        print(combined_results)
    
    logger.info("Done!")

if __name__ == "__main__":
    main()
