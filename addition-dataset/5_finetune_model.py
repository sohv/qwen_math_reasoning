"""
Fine-tuning script using regular transformers for ModelGenie addition dataset.
This script uses Qwen2.5-0.5B-Instruct for efficient fine-tuning.
"""

import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments
)
from trl import SFTTrainer
from datasets import Dataset
import os

class AdditionFineTuner:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", max_seq_length: int = 2048):
        """
        Initialize the fine-tuner with a specific model.
        
        Args:
            model_name: Model to fine-tune (default: "Qwen/Qwen2.5-0.5B-Instruct")
            max_seq_length: Maximum sequence length for training
        """
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load the model and tokenizer using transformers."""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,  # Use float32 to avoid precision issues
            device_map="auto" if torch.cuda.is_available() else None,
        )
        
        # Ensure model is in training mode
        self.model.train()
        
        # Enable gradient checkpointing to save memory
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
    def load_dataset(self, train_file: str, val_file: str = None):
        """Load the training and validation datasets."""
        with open(train_file, 'r') as f:
            train_data = json.load(f)
        
        # Convert to text format
        if isinstance(train_data[0], dict) and "instruction" in train_data[0]:
            # Qwen format
            train_texts = self._convert_qwen_to_text(train_data)
        else:
            # Convert other formats to text
            train_texts = self._convert_to_text_format(train_data)
        
        self.train_dataset = Dataset.from_dict({"text": train_texts})
        
        # Load validation data if provided
        if val_file and os.path.exists(val_file):
            with open(val_file, 'r') as f:
                val_data = json.load(f)
            
            if isinstance(val_data[0], dict) and "instruction" in val_data[0]:
                val_texts = self._convert_qwen_to_text(val_data)
            else:
                val_texts = self._convert_to_text_format(val_data)
                
            self.val_dataset = Dataset.from_dict({"text": val_texts})
        else:
            self.val_dataset = None
        
    def check_dataset_sample(self):
        """Check a sample from the dataset to ensure proper formatting."""
        if hasattr(self, 'train_dataset'):
            print("Dataset sample:")
            sample_text = self.train_dataset[0]['text']
            print(f"First example: {sample_text[:300]}...")
            print(f"Training dataset size: {len(self.train_dataset)}")
            if self.val_dataset:
                print(f"Validation dataset size: {len(self.val_dataset)}")
            
            # Check tokenization
            tokens = self.tokenizer.encode(sample_text)
            print(f"Sample token count: {len(tokens)}")
            print(f"Sample tokens (first 10): {tokens[:10]}")
            
            # Check if tokenizer can decode properly
            decoded = self.tokenizer.decode(tokens)
            print(f"Decoded sample matches: {decoded == sample_text}")
        else:
            print("No dataset loaded yet.")
    
    def _convert_qwen_to_text(self, data):
        """Convert Qwen instruction format to text format for training."""
        texts = []
        
        for item in data:
            # Format as a proper conversation for Qwen
            text = f"<|im_start|>user\n{item['instruction']}: {item['input']}<|im_end|>\n<|im_start|>assistant\n{item['output']}<|im_end|><|endoftext|>"
            texts.append(text)
        
        return texts
    
    def _convert_to_text_format(self, data):
        """Convert different data formats to text format for training."""
        texts = []
        
        for item in data:
            if "messages" in item:  # Chat format
                text = ""
                for msg in item["messages"]:
                    if msg["role"] == "user":
                        text += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
                    else:
                        text += f"<|im_start|>assistant\n{msg['content']}<|im_end|>"
                texts.append(text)
            
            elif "text" in item:  # Direct text format
                texts.append(item["text"])
                
            elif "instruction" in item:  # Instruction format
                text = f"<|im_start|>user\n{item['instruction']}: {item['input']}<|im_end|>\n<|im_start|>assistant\n{item['output']}<|im_end|>"
                texts.append(text)
        
        return texts
    
    def train(self, 
              output_dir: str = "./results",
              num_epochs: int = 3,
              batch_size: int = 1,  # Reduced from 2 to 1
              learning_rate: float = 2e-4,
              warmup_steps: int = 5,
              logging_steps: int = 1):
        """Fine-tune the model."""
        
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if not hasattr(self, 'train_dataset'):
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        # Training arguments
        training_args = TrainingArguments(
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=warmup_steps,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            logging_steps=logging_steps,
            optim="adamw_torch",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output_dir,
            save_steps=max(1, len(self.train_dataset) // (batch_size * 4)),
            save_total_limit=3,
            eval_steps=max(1, len(self.train_dataset) // batch_size) if self.val_dataset else None,
            dataloader_pin_memory=False,
        )
        
        # Create SFT trainer
        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
        )
        
        # Start training
        trainer.train()
        
        # Save the model
        trainer.save_model()
        
        return trainer

def main():
    """Main training function."""
    # Default model
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # Check if data files exist (use Qwen format)
    train_file = "data/sft_pairs_qwen_train.json"
    val_file = "data/sft_pairs_qwen_val.json"
    
    if not os.path.exists(train_file):
        print(f"Training file not found: {train_file}")
        print("Please run step 4 (SFT pair generation) first.")
        return
    
    # Initialize fine-tuner
    finetuner = AdditionFineTuner(model_name=model_name)
    
    try:
        # Load model
        finetuner.load_model()
        
        # Load dataset
        finetuner.load_dataset(train_file, val_file)
        
        # Check dataset formatting
        finetuner.check_dataset_sample()
        
        # Start training
        output_dir = "./finetuned_qwen_addition_model"
        finetuner.train(
            output_dir=output_dir,
            num_epochs=3,
            batch_size=1,
            learning_rate=5e-5,  # Much lower learning rate
            warmup_steps=10,
            logging_steps=1
        )
        
        print(f"Training complete. Model saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
