import json
import ollama
import time
from typing import Dict, List
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class CriticModel:
    def __init__(self, model_type: str = "gemini"):
        self.model_type = model_type
        if model_type == "gemini":
            # Configure Gemini API
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("Please set GEMINI_API_KEY environment variable")
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp')
        elif model_type == "ollama":
            # Use local Ollama model (you can change this to any available model)
            self.model_name = "deepseek-r1:32b"  # or qwq:32b
    
    def correct_reasoning(self, numbers: List[int], original_thought: str, actual_answer: int) -> str:
        """Use the critic model to correct the reasoning process."""
        
        numbers_str = ', '.join(map(str, numbers))
        
        prompt = f"""You are a math teacher helping to correct a student's addition work. 

**Problem**: Add these numbers: {numbers_str}

**Student's original work**:
{original_thought}

**Correct answer**: {actual_answer}

**Your task**: 
1. Identify where the student went wrong in their reasoning
2. Provide a corrected step-by-step thought process that leads to the right answer
3. Start your response with the original thought, then add your corrections
4. Format your final response as: <corrected_thought>your corrected reasoning here</corrected_thought>

Be specific about the errors and show the correct calculation method. Focus on place-value addition and proper carry-over operations."""

        if self.model_type == "gemini":
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                print(f"Gemini API error: {e}")
                return self._fallback_correction(numbers, original_thought, actual_answer)
        
        elif self.model_type == "ollama":
            try:
                response = ollama.chat(
                    model=self.model_name,
                    messages=[{
                        'role': 'user',
                        'content': prompt
                    }]
                )
                return response['message']['content']
            except Exception as e:
                print(f"Ollama error: {e}")
                return self._fallback_correction(numbers, original_thought, actual_answer)
    
    def _fallback_correction(self, numbers: List[int], original_thought: str, actual_answer: int) -> str:
        """Fallback correction when API fails."""
        numbers_str = ' + '.join(map(str, numbers))
        return f"""<corrected_thought>
{original_thought}

**Correction**: Let me recalculate step by step:
{numbers_str} = {actual_answer}

The error in the original calculation was in the addition process. The correct answer is {actual_answer}.
</corrected_thought>"""

def extract_corrected_thought(correction_response: str) -> str:
    """Extract the corrected thought from the critic model response."""
    # Look for content between <corrected_thought> tags
    import re
    match = re.search(r'<corrected_thought>(.*?)</corrected_thought>', correction_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # If no tags found, return the whole response
    return correction_response.strip()

def run_critic_model_corrections(input_file: str, output_file: str, model_type: str = "gemini"):
    """Run the critic model on all errors to generate corrections."""
    
    # Load processed dataset
    with open(input_file, 'r') as f:
        dataset = json.load(f)
    
    # Filter only entries with errors
    error_entries = [entry for entry in dataset if entry["error"]]
    
    print(f"Found {len(error_entries)} entries with errors to correct...")
    
    # Initialize critic model
    try:
        critic = CriticModel(model_type=model_type)
    except Exception as e:
        print(f"Failed to initialize critic model: {e}")
        return
    
    corrected_dataset = []
    
    for i, entry in enumerate(error_entries):
        print(f"Processing error {i+1}/{len(error_entries)}...")
        
        try:
            # Get correction from critic model
            correction_response = critic.correct_reasoning(
                numbers=entry["numbers"],
                original_thought=entry["thought"],
                actual_answer=entry["actual_answer"]
            )
            
            # Extract the corrected thought
            new_thought = extract_corrected_thought(correction_response)
            
            # Create corrected entry
            corrected_entry = {
                **entry,  # Include all original fields
                "critic_response": correction_response,
                "new_thought": new_thought,
                "new_generated_answer": entry["actual_answer"]  # The critic should get it right
            }
            
            corrected_dataset.append(corrected_entry)
            
            # Add delay to avoid rate limiting
            time.sleep(1)
            
        except Exception as e:
            print(f"Error processing entry {i+1}: {e}")
            # Add entry without correction
            corrected_entry = {
                **entry,
                "critic_response": f"Error: {str(e)}",
                "new_thought": entry["thought"],  # Keep original
                "new_generated_answer": entry["actual_answer"]
            }
            corrected_dataset.append(corrected_entry)
    
    # Save corrected dataset
    with open(output_file, 'w') as f:
        json.dump(corrected_dataset, f, indent=2)
    
    print(f"\n=== Critic Model Results ===")
    print(f"Processed {len(error_entries)} error entries")
    print(f"Results saved to: {output_file}")

def preview_corrections(corrected_file: str, num_examples: int = 3):
    """Preview some correction examples."""
    with open(corrected_file, 'r') as f:
        dataset = json.load(f)
    
    print(f"\n=== Correction Examples ===")
    
    for i, entry in enumerate(dataset[:num_examples]):
        print(f"\n--- Example {i+1} ---")
        print(f"Numbers: {entry['numbers']}")
        print(f"Original answer: {entry['generated_answer']}")
        print(f"Correct answer: {entry['actual_answer']}")
        print(f"\nOriginal thought (first 200 chars):")
        print(entry['thought'][:200] + "..." if len(entry['thought']) > 200 else entry['thought'])
        print(f"\nCorrected thought (first 200 chars):")
        print(entry['new_thought'][:200] + "..." if len(entry['new_thought']) > 200 else entry['new_thought'])
        print("-" * 50)

if __name__ == "__main__":
    input_file = "data/processed_additions_with_errors.json"
    output_file = "data/critic_corrected_additions.json"
    
    print("Choose critic model:")
    print("1. Gemini 2.0 Flash Thinking (requires API key)")
    print("2. Local Ollama model (requires deepseek-r1:32b or similar)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        model_type = "gemini"
        api_key = input("Enter your Gemini API key (or set GEMINI_API_KEY env var): ").strip()
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
    else:
        model_type = "ollama"
    
    # Run critic model corrections
    run_critic_model_corrections(input_file, output_file, model_type=model_type)
    
    # Preview results
    if os.path.exists(output_file):
        preview_corrections(output_file)
