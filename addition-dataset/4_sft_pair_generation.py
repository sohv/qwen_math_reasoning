import json
import os
from typing import Dict, List, Tuple

def create_sft_pairs_qwen_format(corrected_file: str, output_file: str):
    """Create SFT pairs in the format used for Qwen model fine-tuning."""
    
    with open(corrected_file, 'r') as f:
        dataset = json.load(f)
    
    sft_pairs = []
    
    for entry in dataset:
        # Format for Qwen (instruction-input-response format)
        sft_pair = {
            "instruction": "add",
            "input": f"{entry['numbers']}<think>{entry['thought']}",
            "output": f"<think>{entry['new_thought']}</think><answer>{entry['actual_answer']}</answer>"
        }
        
        sft_pairs.append(sft_pair)
    
    # Save SFT pairs
    with open(output_file, 'w') as f:
        json.dump(sft_pairs, f, indent=2)
    
    print(f"Created {len(sft_pairs)} SFT pairs for Qwen format")
    print(f"Saved to: {output_file}")

def create_sft_pairs_llama_format(corrected_file: str, output_file: str):
    """Create SFT pairs in chat format for Llama model fine-tuning."""
    
    with open(corrected_file, 'r') as f:
        dataset = json.load(f)
    
    sft_pairs = []
    
    for entry in dataset:
        # Format for Llama (user-assistant chat format)
        conversation = [
            {
                "role": "user",
                "content": f"add {entry['numbers']}\n{entry['thought']}"
            },
            {
                "role": "assistant", 
                "content": f"<think>{entry['new_thought']}</think>\n<answer>{entry['actual_answer']}</answer>"
            }
        ]
        
        sft_pairs.append({"messages": conversation})
    
    # Save SFT pairs
    with open(output_file, 'w') as f:
        json.dump(sft_pairs, f, indent=2)
    
    print(f"Created {len(sft_pairs)} SFT pairs for Llama format")
    print(f"Saved to: {output_file}")

def create_unsloth_format(corrected_file: str, output_file: str):
    """Create SFT pairs in Unsloth format for easy fine-tuning."""
    
    with open(corrected_file, 'r') as f:
        dataset = json.load(f)
    
    sft_pairs = []
    
    for entry in dataset:
        # Unsloth format with special tokens
        conversation = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Add these numbers: {', '.join(map(str, entry['numbers']))}

Original reasoning:
{entry['thought']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

<think>
{entry['new_thought']}
</think>

<answer>{entry['actual_answer']}</answer><|eot_id|>"""
        
        sft_pairs.append({"text": conversation})
    
    # Save SFT pairs
    with open(output_file, 'w') as f:
        json.dump(sft_pairs, f, indent=2)
    
    print(f"Created {len(sft_pairs)} SFT pairs for Unsloth format")
    print(f"Saved to: {output_file}")

def create_training_validation_split(sft_file: str, train_ratio: float = 0.8):
    """Split the SFT pairs into training and validation sets."""
    
    with open(sft_file, 'r') as f:
        sft_pairs = json.load(f)
    
    # Shuffle the data
    import random
    random.seed(42)
    random.shuffle(sft_pairs)
    
    # Split into train/val
    split_idx = int(len(sft_pairs) * train_ratio)
    train_data = sft_pairs[:split_idx]
    val_data = sft_pairs[split_idx:]
    
    # Save splits
    base_name = sft_file.replace('.json', '')
    train_file = f"{base_name}_train.json"
    val_file = f"{base_name}_val.json"
    
    with open(train_file, 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(val_file, 'w') as f:
        json.dump(val_data, f, indent=2)
    
    print(f"Training data: {len(train_data)} samples -> {train_file}")
    print(f"Validation data: {len(val_data)} samples -> {val_file}")

def preview_sft_pairs(sft_file: str, num_examples: int = 2):
    """Preview some SFT pairs to check formatting."""
    
    with open(sft_file, 'r') as f:
        sft_pairs = json.load(f)
    
    print(f"\n=== SFT Pair Examples ===")
    
    for i, pair in enumerate(sft_pairs[:num_examples]):
        print(f"\n--- Example {i+1} ---")
        
        if "instruction" in pair:  # Qwen format
            print("Format: Qwen (instruction-input-output)")
            print(f"Instruction: {pair['instruction']}")
            print(f"Input: {pair['input'][:100]}...")
            print(f"Output: {pair['output'][:100]}...")
            
        elif "messages" in pair:  # Llama format
            print("Format: Llama (chat)")
            for msg in pair["messages"]:
                print(f"{msg['role'].upper()}: {msg['content'][:100]}...")
                
        elif "text" in pair:  # Unsloth format
            print("Format: Unsloth")
            print(f"Text: {pair['text'][:200]}...")
        
        print("-" * 50)

def create_dataset_statistics(corrected_file: str):
    """Generate statistics about the corrected dataset."""
    
    with open(corrected_file, 'r') as f:
        dataset = json.load(f)
    
    print(f"\n=== Dataset Statistics ===")
    print(f"Total error entries: {len(dataset)}")
    
    # Analyze original vs corrected answers
    answer_improvements = 0
    for entry in dataset:
        if entry.get('new_generated_answer') == entry['actual_answer']:
            answer_improvements += 1
    
    print(f"Answers corrected by critic: {answer_improvements}/{len(dataset)}")
    
    # Analyze thought length
    original_thoughts = [len(entry['thought']) for entry in dataset]
    new_thoughts = [len(entry['new_thought']) for entry in dataset]
    
    print(f"Average original thought length: {sum(original_thoughts)/len(original_thoughts):.0f} chars")
    print(f"Average corrected thought length: {sum(new_thoughts)/len(new_thoughts):.0f} chars")
    
    # Number range analysis
    all_numbers = []
    for entry in dataset:
        all_numbers.extend(entry['numbers'])
    
    print(f"Number range: {min(all_numbers):,} to {max(all_numbers):,}")
    print(f"Average number: {sum(all_numbers)/len(all_numbers):,.0f}")

if __name__ == "__main__":
    corrected_file = "data/critic_corrected_additions.json"
    
    print("Generating SFT pairs for fine-tuning...")
    
    # Create dataset statistics
    if os.path.exists(corrected_file):
        create_dataset_statistics(corrected_file)
    else:
        print(f"Corrected file not found: {corrected_file}")
        print("Please run step 3 (critic model) first.")
        exit(1)
    
    # Create different formats
    print("\nCreating SFT pairs in different formats...")
    
    # Qwen format
    create_sft_pairs_qwen_format(corrected_file, "data/sft_pairs_qwen.json")
    
    # Llama format
    create_sft_pairs_llama_format(corrected_file, "data/sft_pairs_llama.json")
    
    # Unsloth format
    create_unsloth_format(corrected_file, "data/sft_pairs_unsloth.json")
    
    # Create train/val splits for each format
    print("\nCreating train/validation splits...")
    create_training_validation_split("data/sft_pairs_qwen.json")
    create_training_validation_split("data/sft_pairs_llama.json")
    create_training_validation_split("data/sft_pairs_unsloth.json")
    
    # Preview examples
    preview_sft_pairs("data/sft_pairs_qwen.json", 1)
    preview_sft_pairs("data/sft_pairs_llama.json", 1)
