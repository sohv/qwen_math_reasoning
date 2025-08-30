#!/usr/bin/env python3
"""
Test script to verify training setup and data formatting.
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_data_format():
    """Test if the training data is properly formatted."""
    print("Testing data format...")
    
    # Load a sample from the training data
    with open("data/sft_pairs_qwen_train.json", 'r') as f:
        data = json.load(f)
    
    print(f"Dataset size: {len(data)}")
    print(f"Sample item keys: {data[0].keys()}")
    
    # Test the conversion
    item = data[0]
    text = f"<|im_start|>user\n{item['instruction']}: {item['input']}<|im_end|>\n<|im_start|>assistant\n{item['output']}<|im_end|><|endoftext|>"
    
    print(f"Formatted text length: {len(text)}")
    print(f"Formatted text sample: {text[:200]}...")
    
    return text

def test_tokenizer():
    """Test tokenizer functionality."""
    print("\nTesting tokenizer...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Get sample text
    text = test_data_format()
    
    # Tokenize
    tokens = tokenizer.encode(text)
    print(f"Token count: {len(tokens)}")
    print(f"First 10 tokens: {tokens[:10]}")
    
    # Test decode
    decoded = tokenizer.decode(tokens)
    print(f"Decode successful: {decoded == text}")
    
    return tokenizer, tokens

def test_model_loading():
    """Test if model loads correctly."""
    print("\nTesting model loading...")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct",
            torch_dtype=torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        
        print(f"Model loaded successfully")
        print(f"Model dtype: {model.dtype}")
        print(f"Model device: {next(model.parameters()).device}")
        
        # Test a forward pass
        tokenizer, tokens = test_tokenizer()
        
        # Create input tensor
        input_ids = torch.tensor([tokens[:50]]).to(next(model.parameters()).device)  # Truncate to avoid memory issues
        
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            print(f"Test forward pass loss: {loss.item()}")
            print(f"Loss is finite: {torch.isfinite(loss)}")
        
        return True
        
    except Exception as e:
        print(f"Model loading failed: {e}")
        return False

if __name__ == "__main__":
    test_data_format()
    test_tokenizer()
    test_model_loading()
