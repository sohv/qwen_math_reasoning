import json
import re
from typing import Dict, List, Optional

def extract_thought_from_response(response: str) -> str:
    """Extract the thought process from the model response."""
    # Look for content between <think> and </think> tags
    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    if think_match:
        return think_match.group(1).strip()
    
    # If no <think> tags, try to extract from the beginning until a clear separation
    lines = response.split('\n')
    thought_lines = []
    for line in lines:
        if line.strip() and not line.startswith('\\[') and 'boxed' not in line:
            thought_lines.append(line.strip())
        else:
            break
    
    return '\n'.join(thought_lines)

def extract_answer_from_response(response: str) -> Optional[int]:
    """Extract the final numerical answer from the model response."""
    # Look for boxed answer first
    boxed_match = re.search(r'\\boxed\{(\d+)\}', response)
    if boxed_match:
        return int(boxed_match.group(1))
    
    # Look for numbers in the response, prefer the last large number
    numbers = re.findall(r'\b(\d{6,})\b', response)
    if numbers:
        return int(numbers[-1])
    
    # Look for any large numbers
    all_numbers = re.findall(r'\b(\d+)\b', response)
    if all_numbers:
        # Filter for numbers that could be realistic sums (6+ digits)
        large_numbers = [int(n) for n in all_numbers if len(n) >= 6]
        if large_numbers:
            return large_numbers[-1]
    
    return None

def calculate_actual_answer(numbers: List[int]) -> int:
    """Calculate the correct sum of the numbers."""
    return sum(numbers)

def detect_errors_in_dataset(input_file: str, output_file: str) -> None:
    """Process the dataset to detect errors and extract information."""
    import os
    
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(input_file, 'r') as f:
        dataset = json.load(f)
    
    processed_dataset = []
    mistake_count = 0
    total_count = len(dataset)
    
    print(f"Processing {total_count} entries...")
    
    for i, entry in enumerate(dataset):
        numbers = entry["numbers"]
        response = entry["response"]
        
        # Extract thought and generated answer
        thought = extract_thought_from_response(response)
        generated_answer = extract_answer_from_response(response)
        
        # Calculate actual answer
        actual_answer = calculate_actual_answer(numbers)
        
        # Check if there's a mistake
        is_mistake = False
        if generated_answer is None or generated_answer != actual_answer:
            is_mistake = True
            mistake_count += 1
        
        processed_entry = {
            "numbers": numbers,
            "response": response,
            "thought": thought,
            "generated_answer": generated_answer,
            "actual_answer": actual_answer,
            "error": is_mistake
        }
        
        processed_dataset.append(processed_entry)
        
        # Print progress
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{total_count} entries...")
    
    # Save processed dataset
    with open(output_file, 'w') as f:
        json.dump(processed_dataset, f, indent=2)
    
    # Print statistics
    accuracy = (total_count - mistake_count) / total_count * 100
    print(f"\n=== Error Detection Results ===")
    print(f"Total entries: {total_count}")
    print(f"Correct answers: {total_count - mistake_count}")
    print(f"Incorrect answers: {mistake_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Error rate: {100 - accuracy:.2f}%")
    print(f"Results saved to: {output_file}")

def analyze_error_types(processed_file: str) -> None:
    """Analyze the types of errors made by the model."""
    with open(processed_file, 'r') as f:
        dataset = json.load(f)
    
    errors = [entry for entry in dataset if entry["error"]]
    
    print(f"\n=== Error Analysis ===")
    print(f"Total errors: {len(errors)}")
    
    if not errors:
        print("No errors found!")
        return
    
    # Analyze magnitude of errors
    error_magnitudes = []
    for entry in errors:
        if entry["generated_answer"] is not None:
            diff = abs(entry["actual_answer"] - entry["generated_answer"])
            error_magnitudes.append(diff)
    
    if error_magnitudes:
        avg_error = sum(error_magnitudes) / len(error_magnitudes)
        max_error = max(error_magnitudes)
        min_error = min(error_magnitudes)
        
        print(f"Average error magnitude: {avg_error:,.0f}")
        print(f"Maximum error magnitude: {max_error:,.0f}")
        print(f"Minimum error magnitude: {min_error:,.0f}")
    
    # Show some examples
    print(f"\n=== Error Examples ===")
    for i, entry in enumerate(errors[:3]):
        print(f"\nExample {i+1}:")
        print(f"Numbers: {entry['numbers']}")
        print(f"Actual answer: {entry['actual_answer']}")
        print(f"Generated answer: {entry['generated_answer']}")
        if entry['generated_answer']:
            diff = entry['actual_answer'] - entry['generated_answer']
            print(f"Difference: {diff:+}")

if __name__ == "__main__":
    import os
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    input_file = os.path.join(script_dir, "data", "additions_r1half-4.json")
    output_file = os.path.join(script_dir, "results", "processed_additions_with_errors.json")
    
    # Step 1: Detect errors
    detect_errors_in_dataset(input_file, output_file)
    
    # Step 2: Analyze error types
    analyze_error_types(output_file)
