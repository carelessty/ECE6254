import os
import torch
from typing import Tuple, Dict, Any, Union
from datasets import load_dataset, DatasetDict
from datasets import Dataset
from datasets.exceptions import DatasetNotFoundError
from transformers import AutoTokenizer
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader, TensorDataset

# Field name constants for dataset compatibility
TEXT_FIELD = "text"
TAGS_FIELD = "tags"
LABEL_FIELD = "label"
SPLIT_NAMES = ("train", "validation", "test")

def parse_text_into_tokens_and_tags(example):
    """
    将示例中 'text' 字段按行解析出 tokens 和 tags。
    假设数据格式：每行的最后一个空格右侧是标签，其余部分是一个 token。
    
    如:
    "New O
    year/New O
    me! O
    I've B-Health
    spent I-Health
    ..."
    将拆分成:
      tokens = ["New", "year/New", "me!", "I've", "spent", ...]
      tags   = ["O",   "O",        "O",     "B-Health", "I-Health", ...]
    """
    text_lines = example[TEXT_FIELD].splitlines()
    tokens, tags = [], []
    for line in text_lines:
        line = line.strip()
        if not line:
            # 跳过空行
            continue
        # 以最后一个空格分隔：左边为 token，右边为 tag
        parts = line.rsplit(" ", 1)
        if len(parts) == 2:
            token, tag = parts
            tokens.append(token)
            tags.append(tag)
        else:
            # 若无法正常分割标签，就给一个默认O标签
            tokens.append(line)
            tags.append("O")

    return {
        "tokens": tokens,
        TAGS_FIELD: tags
    }

class SelfDisclosureDataset(TorchDataset):
    """
    Dataset class for self-disclosure detection task.

    Handles preprocessing for both sentence-level classification and span-level detection tasks.
    """

    def __init__(
        self, 
        dataset,   # HuggingFace Dataset 对象
        tokenizer: AutoTokenizer, 
        max_length: int = 512, 
        task: str = "classification"
    ):
        """
        Initialize the dataset with enhanced error handling.

        Args:
            dataset: HuggingFace Dataset object
            tokenizer: Initialized tokenizer with pad_token set
            max_length: Maximum sequence length (default: 512)
            task: Task type ("classification" or "span")
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task = task

        # Validate task type
        if task not in ("classification", "span"):
            raise ValueError(f"Invalid task type: {task}. Choose 'classification' or 'span'")

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "[PAD]"

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.dataset[idx]
        if self.task == "classification":
            return self._process_classification_item(item)
        else:
            return self._process_span_item(item)

    def _process_classification_item(self, item: Dict) -> Dict[str, torch.Tensor]:
        """Process item for sentence-level classification task."""
        encoding = self.tokenizer(
            item[TEXT_FIELD],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        # 若 label 不存在，则默认为 0；或者可自定义
        label_value = item[LABEL_FIELD] if LABEL_FIELD in item else 0
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_value, dtype=torch.long)
        }

    def _process_span_item(self, item: Dict) -> Dict[str, torch.Tensor]:
        """
        Process item for span-level detection task with improved label alignment.
        - Expects `tokens` as a list of tokens
        - Expects `tags` as a list of IOB/BIO labels, which we'll map to binary 0/1
          (tag starts with 'B-' or 'I-' => 1, else => 0)
        """
        if "tokens" in item and TAGS_FIELD in item:
            tokens = item["tokens"]
            iob_tags = item[TAGS_FIELD]
        else:
            # 如果仍然没有 tokens/tags，可自行 fallback：
            # tokens = item[TEXT_FIELD].split()
            # iob_tags = ["O"] * len(tokens)
            raise KeyError("Item does not contain 'tokens' or 'tags'. Please parse them first.")

        # Convert IOB2 to binary labels with truncation
        max_allowed = self.max_length - 2  # [CLS]/[SEP] 留两个位置
        binary_labels = [
            1 if tag.startswith(("B-", "I-")) else 0
            for tag in iob_tags[:max_allowed]
        ]

        # Tokenize with word alignment
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        # Align labels with subword tokens
        word_ids = encoding.word_ids()
        aligned_labels = []
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)  # special tokens [CLS], [SEP], padding
            elif word_id >= len(binary_labels):
                aligned_labels.append(-100)
            else:
                aligned_labels.append(binary_labels[word_id])

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(aligned_labels, dtype=torch.long)
        }


def load_self_disclosure_dataset(
    tokenizer: AutoTokenizer,
    batch_size: int = 16,
    task: str = "classification"
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Enhanced data loading function.
    - Loads the reddit-self-disclosure dataset
    - Auto-splits if needed
    - If task == 'span', will parse each item['text'] into tokens/tags
    - Returns (train_dataloader, val_dataloader, test_dataloader)
    """
    try:
        # 加载原始数据集
        dataset = load_dataset("douy/reddit-self-disclosure")

        # 统一转换为 DatasetDict 格式
        if not isinstance(dataset, DatasetDict):
            dataset = DatasetDict({'train': dataset})

        # 如果缺少 validation/test，就自动划分
        if not all(split in dataset for split in SPLIT_NAMES):
            print("自动划分验证集和测试集...")
            dataset = self_disclosure_train_test_split(dataset)

        # 若是 span-level 任务，进一步解析 text -> tokens/tags
        if task == "span":
            for split in dataset.keys():
                dataset[split] = dataset[split].map(parse_text_into_tokens_and_tags)

        return create_dataloaders(dataset, tokenizer, batch_size, task)

    except Exception as e:
        print(f"数据加载失败: {str(e)}, 使用模拟数据")
        return create_mock_dataloaders(tokenizer, batch_size, task)


def self_disclosure_train_test_split(dataset: DatasetDict) -> DatasetDict:
    """
    将原始训练集划分为train/val/test (80/10/10)
    """
    # 首次拆分：80%训练，20%临时
    train_test = dataset['train'].train_test_split(
        test_size=0.2, 
        seed=42,
        shuffle=True
    )

    # 二次拆分：20%中分 50%验证，50%测试
    val_test = train_test['test'].train_test_split(
        test_size=0.5, 
        seed=42
    )

    return DatasetDict({
        'train': train_test['train'],
        'validation': val_test['train'],
        'test': val_test['test']
    })


def create_dataloaders(
    dataset: DatasetDict,
    tokenizer: AutoTokenizer,
    batch_size: int,
    task: str
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create dataloaders from validated dataset."""
    train_set = SelfDisclosureDataset(dataset[SPLIT_NAMES[0]], tokenizer, task=task)
    val_set = SelfDisclosureDataset(dataset[SPLIT_NAMES[1]], tokenizer, task=task)
    test_set = SelfDisclosureDataset(dataset[SPLIT_NAMES[2]], tokenizer, task=task)

    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(val_set, batch_size=batch_size),
        DataLoader(test_set, batch_size=batch_size)
    )


def create_mock_dataloaders(
    tokenizer: AutoTokenizer,
    batch_size: int = 16,
    task: str = "classification"
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create reproducible mock data with fixed random seed.

    Returns:
        (train_dataloader, val_dataloader, test_dataloader)
    """
    torch.manual_seed(42)
    generator = torch.Generator().manual_seed(42)

    # Mock data samples for classification
    mock_data = [
        {TEXT_FIELD: "I'm a 23-year-old undergrad.", LABEL_FIELD: 1},
        {TEXT_FIELD: "Design industry joke.", LABEL_FIELD: 0},
        {TEXT_FIELD: "Live in US with husband.", LABEL_FIELD: 1},
    ]

    # ✅ Enhanced mock data samples for span detection
    mock_span_data = [
        {
            "tokens": ["I", "am", "23", "years", "old", "and", "studying", "medicine"],
            TAGS_FIELD: ["O", "O", "B-AGE", "I-AGE", "I-AGE", "O", "O", "B-HEALTH"]
        },
        {
            "tokens": ["Weather", "is", "nice", "today"],
            TAGS_FIELD: ["O", "O", "O", "O"]
        },
        {
            "tokens": ["My", "husband", "and", "I", "live", "in", "Canada"],
            TAGS_FIELD: ["O", "B-FAMILY", "O", "O", "O", "O", "B-LOC"]
        },
        {
            "tokens": ["I", "don't", "want", "to", "share", "anything", "personal"],
            TAGS_FIELD: ["O", "O", "O", "O", "O", "O", "O"]
        },
    ]

    if task == "classification":
        encodings = [
            tokenizer(
                item[TEXT_FIELD],
                truncation=True,
                max_length=128,
                padding="max_length",
                return_tensors="pt"
            ) for item in mock_data
        ]

        dataset = TensorDataset(
            torch.stack([e["input_ids"].squeeze(0) for e in encodings]),
            torch.stack([e["attention_mask"].squeeze(0) for e in encodings]),
            torch.tensor([item[LABEL_FIELD] for item in mock_data], dtype=torch.long)
        )
    else:
        # span-level
        all_encodings = []
        for item in mock_span_data:
            encoding = tokenizer(
                item["tokens"],
                is_split_into_words=True,
                truncation=True,
                max_length=128,
                padding="max_length",
                return_tensors="pt"
            )
            word_ids = encoding.word_ids()
            binary_labels = [1 if tag.startswith(("B-", "I-")) else 0 for tag in item[TAGS_FIELD]]
            aligned_labels = [
                binary_labels[word_id] if word_id is not None and word_id < len(binary_labels) else -100
                for word_id in word_ids
            ]

            all_encodings.append((
                encoding["input_ids"].squeeze(0),
                encoding["attention_mask"].squeeze(0),
                torch.tensor(aligned_labels, dtype=torch.long)
            ))

        dataset = TensorDataset(
            torch.stack([e[0] for e in all_encodings]),
            torch.stack([e[1] for e in all_encodings]),
            torch.stack([e[2] for e in all_encodings])
        )

    # Reproducible split
    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    splits = [train_size, val_size, len(dataset) - train_size - val_size]

    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, splits, generator=generator
    )

    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=generator),
        DataLoader(val_set, batch_size=batch_size),
        DataLoader(test_set, batch_size=batch_size)
    )
