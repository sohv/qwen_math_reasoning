import os
import json
import ollama
from math_llm_pipeline import validate_dataset, evaluate_llm, generate_direct_prompt, generate_cot_prompt

def ollama_deepseek_call(prompt: str) -> str:
    response = ollama.chat(
        model='deepseek-r1:1.5b',
        messages=[{
            'role': 'user',
            'content': prompt
        }]
    )
    return response['message']['content']

if __name__ == "__main__":
    dataset_path = os.path.join("data", "additions_r1half-4.json")
    dataset = validate_dataset(dataset_path)
    print("running LLM evaluation .....")
    evaluate_llm(dataset, prompt_type="direct", model_call_fn=ollama_deepseek_call, output_path="llm_results.json")
    print("evaluation complete. results saved to llm_results.json.") 