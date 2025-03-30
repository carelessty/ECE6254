# Self-Disclosure Detection

This repository contains code for fine-tuning RoBERTa-large on the Reddit self-disclosure dataset for self-disclosure detection, as described in the paper [Reducing Privacy Risks in Online Self-Disclosures with Language Models](https://arxiv.org/pdf/2311.09538).

## Table of Contents

- [Requirements](#requirements)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Dataset Preprocessing](#dataset-preprocessing)
  - [Training](#training)
  - [Inference](#inference)
- [Model Details](#model-details)
- [Evaluation Metrics](#evaluation-metrics)
- [Citation](#citation)

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Dataset

The code uses the [Reddit self-disclosure dataset](https://huggingface.co/datasets/douy/reddit-self-disclosure/) from Hugging Face, which is formatted in CoNLL IOB2 format. The dataset contains 19 self-disclosure categories.

If the dataset only contains a training split (which is common), the code will automatically create validation and test splits using the specified proportions.

To access the dataset, you may need to provide a Hugging Face token.

Alternatively, you can use your own data in CoNLL format, with each line containing a token and its tag separated by a space, and sentences separated by empty lines or `[SEP]` tokens.

## Project Structure

```
self_disclosure_detection/
├── requirements.txt          # Required packages
├── preprocess_dataset.py     # Script for dataset preprocessing
├── train.py                  # Script for model training
├── inference.py              # Script for model inference
├── src/
│   ├── data_utils.py         # Utilities for data loading and processing
│   ├── model_utils.py        # Utilities for model configuration
│   └── model.py              # Custom model implementation
```

## Usage

### Dataset Preprocessing

To preprocess the dataset:

```bash
python preprocess_dataset.py --hf_token YOUR_HF_TOKEN --output_dir ./data
```

For local data:

```bash
python preprocess_dataset.py --local_data_file path/to/your/data.conll --output_dir ./data
```

The preprocessing script will:
1. Load the dataset from Hugging Face or a local file
2. If the dataset only has a training split, create validation and test splits
3. Save the processed dataset to the specified output directory
4. Analyze and print dataset statistics

### Training

To fine-tune RoBERTa-large on the dataset:

```bash
python train.py \
  --hf_token YOUR_HF_TOKEN \
  --model_name_or_path roberta-large \
  --output_dir ./output \
  --data_dir ./data \
  --num_train_epochs 5 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --learning_rate 5e-5 \
  --warmup_steps 500 \
  --weight_decay 0.01
```

The training script will:
1. First try to load the preprocessed dataset from `data_dir/processed_dataset`
2. If not found, try to load from `data_dir/original_dataset`
3. If still not found, load from Hugging Face
4. If the dataset only has a training split, automatically create validation and test splits
5. Prepare the dataset for training by tokenizing and aligning labels
6. Train and evaluate the model

Additional options:
- `--fp16`: Enable mixed precision training
- `--seed`: Set random seed (default: 42)
- `--cache_dir`: Directory to cache the dataset and models
- `--train_split`: Proportion of data to use for training (default: 0.8)
- `--val_split`: Proportion of data to use for validation (default: 0.1)

### Inference

To run inference with a fine-tuned model:

```bash
python inference.py \
  --model_path ./output \
  --input_text "I've spent almost all of 2022 in treatment. Inpatient, intensive outpatient, and sober living."
```

For batch inference from a file:

```bash
python inference.py \
  --model_path ./output \
  --input_file path/to/input.txt \
  --output_file path/to/output.txt
```

Additional options:
- `--device`: Device to run inference on (-1 for CPU, 0+ for GPU)
- `--batch_size`: Batch size for inference

## Model Details

The implementation fine-tunes RoBERTa-large for self-disclosure detection as a sequence tagging task. The model is trained to identify 19 different categories of self-disclosure in text, including health, age, gender, location, and other personal attributes.

Key configuration parameters:
- Model: RoBERTa-large
- Learning rate: 5e-5
- Batch size: 16
- Training epochs: 5
- Weight decay: 0.01
- Warmup steps: 500

## Evaluation Metrics

The model is evaluated using the following metrics:
- Standard precision, recall, and F1 score
- Partial span F1 score: A more lenient metric that considers a predicted span as correct if it overlaps with a reference span by at least 50% of the longer span's length

## Citation

If you use this code, please cite the original paper:

```
@article{dou2023reducing,
  title={Reducing Privacy Risks in Online Self-Disclosures with Language Models},
  author={Dou, Yao and Krsek, Isadora and Naous, Tarek and Kabra, Anubha and Das, Sauvik and Ritter, Alan and Xu, Wei},
  journal={arXiv preprint arXiv:2311.09538},
  year={2023}
}
```
