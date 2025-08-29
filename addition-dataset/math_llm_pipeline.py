import json
import os
import re
from typing import List, Dict
import dspy

class MathAddition(dspy.Signature):
    """Add a list of numbers and provide the sum."""
    numbers = dspy.InputField(desc="List of numbers to add")
    result = dspy.OutputField(desc="The sum of the numbers")

class MathAdditionWithReasoning(dspy.Signature):
    """Add numbers step by step with reasoning."""
    numbers = dspy.InputField(desc="List of numbers to add")
    reasoning = dspy.OutputField(desc="Step-by-step calculation process")
    result = dspy.OutputField(desc="The final sum")

class DirectMathSolver(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.Predict(MathAddition)
    
    def forward(self, numbers):
        numbers_str = ', '.join(str(n) for n in numbers)
        return self.generate_answer(numbers=numbers_str)

class ChainOfThoughtMathSolver(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(MathAdditionWithReasoning)
    
    def forward(self, numbers):
        numbers_str = ', '.join(str(n) for n in numbers)
        return self.generate_answer(numbers=numbers_str)

def validate_dataset(json_path: str) -> List[Dict]:
    required_fields = ["numbers", "response"]
    with open(json_path, 'r') as f:
        data = json.load(f)
    cleaned = []
    for i, entry in enumerate(data):
        if all(field in entry for field in required_fields):
            cleaned.append(entry)
        else:
            print(f"Entry {i} missing required fields: {entry}")
    print(f"Validated {len(cleaned)} out of {len(data)} entries.")
    return cleaned

def extract_number_from_response(response: str) -> int:
    """Extract the final number from model response."""
    numbers = re.findall(r'-?\d+', str(response))
    return int(numbers[-1]) if numbers else 0

def accuracy_metric(gold, pred, trace=None):
    """Metric to evaluate accuracy of math predictions."""
    try:
        gold_result = int(gold.result) if hasattr(gold, 'result') else int(gold)
        pred_result = extract_number_from_response(pred.result if hasattr(pred, 'result') else pred)
        return gold_result == pred_result
    except:
        return False

def evaluate_llm_dspy(dataset: List[Dict], solver_type: str = "direct", lm=None, optimize: bool = False, output_path="llm_results.json"):
    """Evaluate using DSPy framework."""
    if lm:
        dspy.settings.configure(lm=lm)
    
    # Choose solver
    if solver_type == "direct":
        solver = DirectMathSolver()
    else:
        solver = ChainOfThoughtMathSolver()
    
    # Convert dataset to DSPy format
    examples = []
    for entry in dataset:
        examples.append(dspy.Example(
            numbers=entry["numbers"],
            result=str(entry["response"])
        ).with_inputs('numbers'))
    
    # Optimize if requested
    if optimize and len(examples) > 5:
        trainset = examples[:len(examples)//2]
        testset = examples[len(examples)//2:]
        
        optimizer = dspy.BootstrapFewShot(metric=accuracy_metric, max_bootstrapped_demos=4)
        optimized_solver = optimizer.compile(solver, trainset=trainset)
        solver = optimized_solver
    else:
        testset = examples
    
    # Evaluate
    results = []
    correct = 0
    for example in testset:
        try:
            prediction = solver(numbers=example.numbers)
            is_correct = accuracy_metric(example, prediction)
            correct += is_correct
            
            results.append({
                "numbers": example.numbers,
                "prediction": str(prediction.result if hasattr(prediction, 'result') else prediction),
                "reasoning": str(prediction.reasoning if hasattr(prediction, 'reasoning') else ""),
                "ground_truth": example.result,
                "correct": is_correct
            })
        except Exception as e:
            results.append({
                "numbers": example.numbers,
                "prediction": f"Error: {str(e)}",
                "reasoning": "",
                "ground_truth": example.result,
                "correct": False
            })
    
    accuracy = correct / len(results) if results else 0
    
    # Save results
    output_data = {
        "accuracy": accuracy,
        "total_examples": len(results),
        "correct_predictions": correct,
        "solver_type": solver_type,
        "optimized": optimize,
        "results": results
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Accuracy: {accuracy:.2%} ({correct}/{len(results)})")
    print(f"Saved results to {output_path}")
    
    return solver, accuracy

# Backward compatibility functions
def generate_direct_prompt(numbers: List[int]) -> str:
    return f"What is the sum of the following numbers? {', '.join(str(n) for n in numbers)}"

def generate_cot_prompt(numbers: List[int]) -> str:
    return (
        "Let's solve this step by step. "
        f"Add the following numbers one by one: {', '.join(str(n) for n in numbers)}. "
        "Show your reasoning and give the final answer."
    )

def evaluate_llm(dataset: List[Dict], prompt_type: str = "direct", model_call_fn=None, output_path="llm_results.json"):
    """Legacy evaluation function for backward compatibility."""
    results = []
    for entry in dataset:
        numbers = entry["numbers"]
        if prompt_type == "direct":
            prompt = generate_direct_prompt(numbers)
        else:
            prompt = generate_cot_prompt(numbers)
        if model_call_fn:
            model_response = model_call_fn(prompt)
        else:
            model_response = "<model output here>"
        results.append({
            "numbers": numbers,
            "prompt": prompt,
            "model_response": model_response,
            "ground_truth": entry["response"]
        })
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved LLM results to {output_path}")

if __name__ == "__main__":
    dataset_path = os.path.join("data", "additions_r1half-4.json")
    dataset = validate_dataset(dataset_path)
    print("\nExample Direct Prompt:")
    print(generate_direct_prompt(dataset[0]["numbers"]))
    print("\nExample Chain-of-Thought Prompt:")
    print(generate_cot_prompt(dataset[0]["numbers"]))
    print("\nEvaluation pipeline ready") 