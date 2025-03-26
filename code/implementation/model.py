"""
Model implementation for privacy risk detection using DeepSeek-R1-Distill-Qwen-1.5B.
This module handles the adaptation of the model for self-disclosure detection tasks.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

class PrivacyRiskClassifier:
    """
    Wrapper class for privacy risk detection models.
    
    This class handles different approaches for adapting DeepSeek-R1-Distill-Qwen-1.5B
    for privacy risk detection tasks, including fine-tuning and few-shot learning.
    """
    
    def __init__(
        self,
        model_name="deepseek-ai/deepseek-r1-distill-qwen-1.5b",
        task="classification",
        use_lora=True,
        device=None
    ):
        """
        Initialize the privacy risk classifier.
        
        Args:
            model_name: Name or path of the pre-trained model
            task: Either "classification" for sentence-level or "span" for span-level detection
            use_lora: Whether to use LoRA for parameter-efficient fine-tuning
            device: Device to use (None for auto-detection)
        """
        self.model_name = model_name
        self.task = task
        self.use_lora = use_lora
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Ensure the tokenizer has padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize model based on task
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model based on the specified task."""
        if self.task == "classification":
            # For sentence-level classification
            if "deepseek" in self.model_name.lower():
                # For DeepSeek models, we need to adapt the causal LM for classification
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,
                    device_map=self.device
                )
                
                # Apply LoRA for parameter-efficient fine-tuning
                if self.use_lora:
                    lora_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        r=16,
                        lora_alpha=32,
                        lora_dropout=0.1,
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    )
                    
                    # Prepare model for LoRA fine-tuning
                    self.model = prepare_model_for_kbit_training(self.model)
                    self.model = get_peft_model(self.model, lora_config)
                
                # We'll use a prompt-based approach for classification
                self.is_generative = True
            
            else:
                # For traditional classification models like RoBERTa
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=2,  # Binary classification
                    device_map=self.device
                )
                self.is_generative = False
        
        elif self.task == "span":
            # For span-level detection
            if "deepseek" in self.model_name.lower():
                # For DeepSeek models, we need to adapt the causal LM for token classification
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,
                    device_map=self.device
                )
                
                # Apply LoRA for parameter-efficient fine-tuning
                if self.use_lora:
                    lora_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        r=16,
                        lora_alpha=32,
                        lora_dropout=0.1,
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    )
                    
                    # Prepare model for LoRA fine-tuning
                    self.model = prepare_model_for_kbit_training(self.model)
                    self.model = get_peft_model(self.model, lora_config)
                
                # We'll use a prompt-based approach for token classification
                self.is_generative = True
            
            else:
                # For traditional token classification models
                self.model = AutoModelForTokenClassification.from_pretrained(
                    self.model_name,
                    num_labels=2,  # Binary classification (disclosure or not)
                    device_map=self.device
                )
                self.is_generative = False
        
        else:
            raise ValueError(f"Unknown task: {self.task}")
    
    def train(self, train_dataloader, val_dataloader, output_dir, num_epochs=3):
        """
        Train the model on the provided data.
        
        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            output_dir: Directory to save the model
            num_epochs: Number of training epochs
        
        Returns:
            Training metrics
        """
        if self.is_generative:
            # For generative models, we use a custom training loop
            return self._train_generative(train_dataloader, val_dataloader, output_dir, num_epochs)
        else:
            # For classification models, we use the Hugging Face Trainer
            return self._train_classification(train_dataloader, val_dataloader, output_dir, num_epochs)
    
    def _train_classification(self, train_dataloader, val_dataloader, output_dir, num_epochs):
        """Training loop for classification models."""
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            push_to_hub=False,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataloader.dataset,
            eval_dataset=val_dataloader.dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        trainer.train()
        trainer.save_model(f"{output_dir}/best_model")
        
        return trainer.state.log_history
    
    def _train_generative(self, train_dataloader, val_dataloader, output_dir, num_epochs=3):
        """
        Custom training loop for generative models using a prompt-based approach,
        aligned with the data structure from SelfDisclosureDataset.
        
        We assume:
        - train_dataloader/val_dataloader yield batches with 'input_ids', 'attention_mask', and 'labels'.
        - 'labels' is 0 or 1 (binary classification), which we map to 'No' or 'Yes'.
        """
        print("Training generative model with prompt-based approach...")

        # Basic label-to-text mapping for demonstration
        label2text = {0: "No", 1: "Yes"}

        # Put the model in training mode
        self.model.train()

        # Example optimizer (modify hyperparameters or add a scheduler as needed)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)

        for epoch in range(num_epochs):
            self.model.train()
            total_train_loss = 0.0

            # ------------------
            # TRAINING LOOP
            # ------------------
            for step, batch in enumerate(train_dataloader):
                # Move batch to the correct device
                batch_input_ids = batch["input_ids"].to(self.device)
                batch_attention_mask = batch["attention_mask"].to(self.device)
                batch_labels = batch["labels"].to(self.device)

                # We'll accumulate the loss over each item in the batch
                batch_loss = 0.0

                # ----------------------------------------------------------------
                # For prompt-based teacher forcing, we decode each example, append
                # the textual label (Yes/No), and re-tokenize. This is illustrative
                # but not the most efficient for large batches.
                # ----------------------------------------------------------------
                for i in range(len(batch_input_ids)):
                    # 1) Decode the tokens back into text
                    #    (skip special tokens to keep it cleaner)
                    decoded_text = self.tokenizer.decode(
                        batch_input_ids[i],
                        skip_special_tokens=True
                    ).strip()

                    # 2) Convert numeric label to text label
                    label_text = label2text[batch_labels[i].item()]

                    # 3) Create the prompt. Example:
                    #    "Text: <decoded_text>\nDoes this text contain self-disclosure? Answer: Yes/No"
                    # You can refine the prompt as needed.
                    prompt = (
                        "Determine if the following text contains self-disclosure. "
                        "Answer with Yes or No.\n\n"
                        f"Text: {decoded_text}\nAnswer:"
                    )

                    # 4) Combine the prompt + label for teacher forcing
                    #    e.g.  "...Answer: Yes"
                    full_sequence = prompt + " " + label_text

                    # 5) Re-tokenize
                    inputs = self.tokenizer(full_sequence, return_tensors="pt").to(self.device)
                    labels_tensors = inputs["input_ids"].clone()

                    # 6) Forward pass with teacher forcing
                    outputs = self.model(**inputs, labels=labels_tensors)
                    loss = outputs.loss
                    batch_loss += loss.item()

                    # 7) Backprop
                    loss.backward()

                # 8) Optimizer step
                optimizer.step()
                optimizer.zero_grad()

                total_train_loss += batch_loss

            avg_train_loss = total_train_loss / len(train_dataloader)

            # ------------------
            # VALIDATION LOOP
            # ------------------
            self.model.eval()
            total_val_loss = 0.0

            with torch.no_grad():
                for step, batch in enumerate(val_dataloader):
                    batch_input_ids = batch["input_ids"].to(self.device)
                    batch_attention_mask = batch["attention_mask"].to(self.device)
                    batch_labels = batch["labels"].to(self.device)

                    for i in range(len(batch_input_ids)):
                        decoded_text = self.tokenizer.decode(
                            batch_input_ids[i],
                            skip_special_tokens=True
                        ).strip()

                        label_text = label2text[batch_labels[i].item()]

                        prompt = (
                            "Determine if the following text contains self-disclosure. "
                            "Answer with Yes or No.\n\n"
                            f"Text: {decoded_text}\nAnswer:"
                        )
                        full_sequence = prompt + " " + label_text

                        inputs = self.tokenizer(full_sequence, return_tensors="pt").to(self.device)
                        labels_tensors = inputs["input_ids"].clone()

                        outputs = self.model(**inputs, labels=labels_tensors)
                        total_val_loss += outputs.loss.item()

            avg_val_loss = total_val_loss / len(val_dataloader)

            # Print or log the epoch stats
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val   Loss: {avg_val_loss:.4f}")

        # Optionally save the model at the end
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        return {
            "message": "Generative model training completed",
            "final_train_loss": avg_train_loss,
            "final_val_loss": avg_val_loss
        }

    
    def predict(self, texts):
        """
        Make predictions on the provided texts.
        
        Args:
            texts: List of texts to classify
        
        Returns:
            Predictions (labels for classification, spans for span detection)
        """
        if self.is_generative:
            # For generative models, we use a prompt-based approach
            return self._predict_generative(texts)
        else:
            # For classification models, we use the standard approach
            return self._predict_classification(texts)
    
    def _predict_classification(self, texts):
        """Make predictions using classification models."""
        # Tokenize inputs
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        return predictions.cpu().numpy()
    
    def _predict_generative(self, texts):
        """Make predictions using generative models with prompt-based approach."""
        results = []
        
        for text in texts:
            if self.task == "classification":
                # Prompt for classification
                prompt = f"Determine if the following text contains self-disclosure (personal information). Answer with 'Yes' or 'No'.\n\nText: {text}\n\nContains self-disclosure:"
            else:
                # Prompt for span detection
                prompt = f"Identify spans of text that contain self-disclosure (personal information) in the following text. Mark each word as either 'D' for disclosure or 'N' for non-disclosure.\n\nText: {text}\n\nLabels:"
            
            # Tokenize prompt
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate response
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.1,
                    do_sample=False
                )
            
            # Decode response
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract prediction from response
            if self.task == "classification":
                # For classification, extract Yes/No
                prediction = 1 if "Yes" in response.split(prompt)[1] else 0
            else:
                # For span detection, extract D/N labels
                # This is a simplified approach and would need to be refined
                labels_text = response.split(prompt)[1].strip()
                prediction = [1 if label == 'D' else 0 for label in labels_text.split()]
            
            results.append(prediction)
        
        return results
    
    def save(self, path):
        """Save the model and tokenizer."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    @classmethod
    def load(cls, path, task="classification", device=None):
        """Load a saved model."""
        instance = cls(model_name=path, task=task, device=device)
        return instance


class FewShotPrivacyRiskClassifier:
    """
    Few-shot learning approach for privacy risk detection.
    
    This class implements few-shot learning using DeepSeek-R1-Distill-Qwen-1.5B
    without fine-tuning, leveraging the model's in-context learning capabilities.
    """
    
    def __init__(
        self,
        model_name="deepseek-ai/deepseek-r1-distill-qwen-1.5b",
        task="classification",
        device=None
    ):
        """
        Initialize the few-shot privacy risk classifier.
        
        Args:
            model_name: Name or path of the pre-trained model
            task: Either "classification" for sentence-level or "span" for span-level detection
            device: Device to use (None for auto-detection)
        """
        self.model_name = model_name
        self.task = task
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device
        )
        
        # Ensure the tokenizer has padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Example prompts for few-shot learning
        self._initialize_examples()
    
    def _initialize_examples(self):
        """Initialize example prompts for few-shot learning."""
        if self.task == "classification":
            # Examples for sentence-level classification
            self.examples = [
                {
                    "text": "I am a 23-year-old who is currently going through the last leg of undergraduate school.",
                    "label": "Yes"
                },
                {
                    "text": "There is a joke in the design industry about that.",
                    "label": "No"
                },
                {
                    "text": "My husband and I live in US.",
                    "label": "Yes"
                },
                {
                    "text": "I was messing with advanced voice the other day and I was like, 'Oh, I can do this.'",
                    "label": "No"
                },
                {
                    "text": "I'm 16F I think I want to be a bi M",
                    "label": "Yes"
                }
            ]
        else:
            # Examples for span-level detection
            self.examples = [
                {
                    "text": "I am a 23-year-old who is currently going through the last leg of undergraduate school.",
                    "labels": "N N N D D D N N N N N N N D D N"
                },
                {
                    "text": "My husband and I live in US.",
                    "labels": "N D N N N N D N"
                },
                {
                    "text": "I'm 16F I think I want to be a bi M",
                    "labels": "N D N N N N N D D"
                }
            ]
    
    def predict(self, texts):
        """
        Make predictions on the provided texts using few-shot learning.
        
        Args:
            texts: List of texts to classify
        
        Returns:
            Predictions (labels for classification, spans for span detection)
        """
        results = []
        
        for text in texts:
            # Create prompt with examples
            prompt = self._create_prompt(text)
            
            # Tokenize prompt
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate response
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.1,
                    do_sample=False
                )
            
            # Decode response
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract prediction from response
            prediction = self._extract_prediction(response, prompt)
            results.append(prediction)
        
        return results
    
    def _create_prompt(self, text):
        """Create a prompt with examples for few-shot learning."""
        if self.task == "classification":
            # Prompt for classification
            prompt = "Determine if the following texts contain self-disclosure (personal information). Answer with 'Yes' or 'No'.\n\n"
            
            # Add examples
            for example in self.examples:
                prompt += f"Text: {example['text']}\nContains self-disclosure: {example['label']}\n\n"
            
            # Add the text to classify
            prompt += f"Text: {text}\nContains self-disclosure:"
        
        else:
            # Prompt for span detection
            prompt = "Identify spans of text that contain self-disclosure (personal information) in the following texts. Mark each word as either 'D' for disclosure or 'N' for non-disclosure.\n\n"
            
            # Add examples
            for example in self.examples:
                prompt += f"Text: {example['text']}\nLabels: {example['labels']}\n\n"
            
            # Add the text to classify
            prompt += f"Text: {text}\nLabels:"
        
        return prompt
    
    def _extract_prediction(self, response, prompt):
        """Extract prediction from model response."""
        # Get the part after the prompt
        result = response[len(prompt):].strip()
        
        if self.task == "classification":
            # For classification, extract Yes/No
            if result.startswith("Yes"):
                return 1
            else:
                return 0
        else:
            # For span detection, extract D/N labels
            # This is a simplified approach and would need to be refined
            labels = result.split()
            return [1 if label == 'D' else 0 for label in labels]
