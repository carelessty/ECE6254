import logging
import sys
from typing import List, Optional, Dict
import torch
from transformers import AutoTokenizer, pipeline
from src.neobert_utils import load_neobert_model

logger = logging.getLogger(__name__)

class NeoBERTInference:
    def __init__(
        self,
        model_path: str,
        device: int = -1,
        num_labels: int = 2,
        label_list: Optional[List[str]] = None,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        classifier_dropout: Optional[float] = None,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
    ):
        """
        初始化 NeoBERTInference 类，并加载模型、分词器和 token 分类管道。
        """
        if label_list is None:
            label_list = ["0", "1"]

        self.model_path = model_path
        self.device = device
        self.num_labels = num_labels
        self.label_list = label_list
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.classifier_dropout = classifier_dropout

        # 构造一个简单的 args 对象以传入 load_neobert_model
        class Args:
            pass
        args = Args()
        args.model_name_or_path = model_path
        args.hidden_dropout_prob = hidden_dropout_prob
        args.attention_probs_dropout_prob = attention_probs_dropout_prob
        args.classifier_dropout = classifier_dropout
        args.cache_dir = cache_dir
        args.local_files_only = local_files_only

        # 加载 NeoBERT 模型
        logger.info(f"Loading NeoBERT model from {model_path}...")
        self.model = load_neobert_model(args, num_labels, label_list)

        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            use_fast=True,
            model_max_length=512,
            add_prefix_space=True,
        )

        # 创建 token-classification 管道
        self.classifier = pipeline(
            "token-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
            aggregation_strategy="simple"  # 对连续实体进行简单聚合
        )
        logger.info("NeoBERTInference 初始化完成.")

    def preprocess_text(self, text: str) -> List[str]:
        """
        简单地将输入文本按句号拆分为句子列表。
        """
        sentences = []
        for sent in text.replace('\n', ' ').split('.'):
            sent = sent.strip()
            if sent:
                sentences.append(sent + '.')
        return sentences if sentences else [text]

    def run_inference(self, text: str) -> List[Dict]:
        """
        对输入文本进行推断，返回所有检测到的自我披露信息。
        """
        sentences = self.preprocess_text(text)
        all_disclosures = []
        current_offset = 0
        for sentence in sentences:
            results = self.classifier(sentence)
            for entity in results:
                # 如果实体组不是 "O"，则认为是一个自我披露的实体
                if entity.get("entity_group", "O") != "O":
                    disclosure = {
                        "text": entity["word"],
                        "start": current_offset + entity["start"],
                        "end": current_offset + entity["end"],
                        "type": entity["entity_group"],
                        "score": entity["score"]
                    }
                    all_disclosures.append(disclosure)
            current_offset += len(sentence)
        return all_disclosures

    def format_results(self, text: str, disclosures: List[Dict]) -> str:
        """
        格式化推断结果为可读字符串。
        """
        if not disclosures:
            return f"Text: {text}\nNo self-disclosures detected."
        # 按照实体出现位置排序
        disclosures = sorted(disclosures, key=lambda x: x["start"])
        result = f"Text: {text}\n\nSelf-disclosures detected:\n"
        for i, disclosure in enumerate(disclosures, 1):
            result += f"{i}. Type: {disclosure['type']}\n"
            result += f"   Text: \"{disclosure['text']}\"\n"
            result += f"   Position: {disclosure['start']}-{disclosure['end']}\n"
            result += f"   Confidence: {disclosure['score']:.4f}\n\n"
        return result

    def infer_and_format(self, text: str) -> str:
        """
        对输入文本进行推断并格式化输出结果。
        """
        disclosures = self.run_inference(text)
        return self.format_results(text, disclosures)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Run inference with fine-tuned NeoBERT model for self-disclosure detection"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the fine-tuned model or pretrained model"
    )
    parser.add_argument(
        "--input_text", type=str, default=None, help="Text to analyze for self-disclosure"
    )
    parser.add_argument(
        "--input_file", type=str, default=None, help="Path to file containing text to analyze"
    )
    parser.add_argument(
        "--output_file", type=str, default=None, help="Path to save the results"
    )
    parser.add_argument(
        "--device", type=int, default=-1, help="Device to run inference on (-1 for CPU, 0+ for GPU)"
    )
    parser.add_argument(
        "--neobert_attention_heads", type=int, default=8, help="Attention heads for NeoBERT"
    )
    parser.add_argument(
        "--hidden_dropout_prob", type=float, default=0.1, help="Hidden dropout probability"
    )
    parser.add_argument(
        "--attention_probs_dropout_prob", type=float, default=0.1, help="Attention dropout probability"
    )
    parser.add_argument(
        "--classifier_dropout", type=float, default=None, help="Classifier dropout probability"
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None, help="Cache directory"
    )
    parser.add_argument(
        "--local_files_only", action="store_true", help="Use local files only"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    logger.info("Starting NeoBERT Inference...")

    if not args.input_text and not args.input_file:
        logger.error("Either --input_text or --input_file must be provided.")
        sys.exit(1)

    # 创建 NeoBERTInference 对象
    inference_obj = NeoBERTInference(
        model_path=args.model_path,
        device=args.device,
        hidden_dropout_prob=args.hidden_dropout_prob,
        attention_probs_dropout_prob=args.attention_probs_dropout_prob,
        classifier_dropout=args.classifier_dropout,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only
    )

    # 处理单条文本或文件
    if args.input_text:
        logger.info("Processing input text...")
        result = inference_obj.infer_and_format(args.input_text)
        if args.output_file:
            with open(args.output_file, "w", encoding="utf-8") as f:
                f.write(result)
            logger.info(f"Results saved to {args.output_file}")
        else:
            print(result)
    elif args.input_file:
        logger.info(f"Processing texts from {args.input_file}...")
        with open(args.input_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        all_results = []
        for i, text in enumerate(texts, 1):
            logger.info(f"Processing text {i}/{len(texts)}...")
            res = inference_obj.infer_and_format(text)
            all_results.append(res)
        combined_results = "\n" + "=" * 80 + "\n".join(all_results)
        if args.output_file:
            with open(args.output_file, "w", encoding="utf-8") as f:
                f.write(combined_results)
            logger.info(f"Results saved to {args.output_file}")
        else:
            print(combined_results)

    logger.info("Inference completed.")

if __name__ == "__main__":
    main()
