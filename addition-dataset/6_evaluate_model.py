import json
import ollama
import random
from typing import List, Dict, Tuple
import re

def generate_test_problems(num_problems: int = 100, num_digits: int = 6) -> List[Dict]:
    """Generate fresh test problems for evaluation."""
    test_problems = []
    
    for _ in range(num_problems):
        numbers = [random.randint(10**(num_digits-1), 10**num_digits-1) for _ in range(6)]
        actual_answer = sum(numbers)
        
        test_problems.append({
            "numbers": numbers,
            "actual_answer": actual_answer
        })
    
    return test_problems

def extract_answer_from_response(response: str) -> int:
    """Extract the numerical answer from model response."""
    # Look for answer tags
    answer_match = re.search(r'<answer>(\d+)</answer>', response)
    if answer_match:
        return int(answer_match.group(1))
    
    # Look for boxed answer
    boxed_match = re.search(r'\\boxed\{(\d+)\}', response)
    if boxed_match:
        return int(boxed_match.group(1))
    
    # Look for the last large number in the response
    numbers = re.findall(r'\b(\d{6,})\b', response)
    if numbers:
        return int(numbers[-1])
    
    return None

def test_model_with_ollama(model_name: str, problems: List[Dict]) -> List[Dict]:
    """Test a model using Ollama."""
    results = []
    
    print(f"Testing {len(problems)} problems with {model_name}...")
    
    for i, problem in enumerate(problems):
        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{len(problems)}")
        
        numbers_str = ', '.join(map(str, problem['numbers']))
        prompt = f"Add these numbers: {numbers_str}\nFormat your final answer as <answer>result</answer>"
        
        try:
            response = ollama.chat(
                model=model_name,
                messages=[{
                    'role': 'user',
                    'content': prompt
                }]
            )
            
            model_response = response['message']['content']
            generated_answer = extract_answer_from_response(model_response)
            
            is_correct = generated_answer == problem['actual_answer']
            
            result = {
                **problem,
                "model_response": model_response,
                "generated_answer": generated_answer,
                "correct": is_correct
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error with problem {i+1}: {e}")
            result = {
                **problem,
                "model_response": f"Error: {str(e)}",
                "generated_answer": None,
                "correct": False
            }
            results.append(result)
    
    return results

def analyze_results(results: List[Dict]) -> Dict:
    """Analyze the evaluation results."""
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    accuracy = correct / total * 100 if total > 0 else 0
    
    # Analyze error types
    errors = [r for r in results if not r['correct']]
    error_magnitudes = []
    
    for error in errors:
        if error['generated_answer'] is not None:
            diff = abs(error['actual_answer'] - error['generated_answer'])
            error_magnitudes.append(diff)
    
    analysis = {
        "total_problems": total,
        "correct_answers": correct,
        "accuracy": accuracy,
        "error_rate": 100 - accuracy,
        "total_errors": len(errors),
        "parsing_errors": sum(1 for e in errors if e['generated_answer'] is None),
        "calculation_errors": len(error_magnitudes)
    }
    
    if error_magnitudes:
        analysis.update({
            "avg_error_magnitude": sum(error_magnitudes) / len(error_magnitudes),
            "max_error_magnitude": max(error_magnitudes),
            "min_error_magnitude": min(error_magnitudes)
        })
    
    return analysis

def compare_models(original_model: str, finetuned_model: str, test_problems: List[Dict]):
    """Compare original and fine-tuned models."""
    print("=== Model Comparison ===")
    
    # Test original model
    print(f"\nTesting original model: {original_model}")
    original_results = test_model_with_ollama(original_model, test_problems)
    original_analysis = analyze_results(original_results)
    
    # Test fine-tuned model
    print(f"\nTesting fine-tuned model: {finetuned_model}")
    finetuned_results = test_model_with_ollama(finetuned_model, test_problems)
    finetuned_analysis = analyze_results(finetuned_results)
    
    # Print comparison
    print("\n" + "="*50)
    print("COMPARISON RESULTS")
    print("="*50)
    
    print(f"{'Metric':<25} {'Original':<15} {'Fine-tuned':<15} {'Improvement':<15}")
    print("-" * 70)
    
    metrics = [
        ("Accuracy (%)", "accuracy"),
        ("Error Rate (%)", "error_rate"),
        ("Correct Answers", "correct_answers"),
        ("Parsing Errors", "parsing_errors"),
        ("Calculation Errors", "calculation_errors")
    ]
    
    for metric_name, metric_key in metrics:
        orig_val = original_analysis[metric_key]
        fine_val = finetuned_analysis[metric_key]
        
        if metric_key in ["accuracy"]:
            improvement = fine_val - orig_val
            improvement_str = f"+{improvement:.2f}%" if improvement > 0 else f"{improvement:.2f}%"
        elif metric_key in ["error_rate", "parsing_errors", "calculation_errors"]:
            improvement = orig_val - fine_val
            improvement_str = f"-{abs(improvement):.0f}" if improvement > 0 else f"+{abs(improvement):.0f}"
        else:
            improvement = fine_val - orig_val
            improvement_str = f"+{improvement:.0f}" if improvement > 0 else f"{improvement:.0f}"
        
        print(f"{metric_name:<25} {orig_val:<15.2f} {fine_val:<15.2f} {improvement_str:<15}")
    
    # Error magnitude comparison
    if "avg_error_magnitude" in original_analysis and "avg_error_magnitude" in finetuned_analysis:
        print(f"\nError Magnitude Analysis:")
        print(f"Original avg error: {original_analysis['avg_error_magnitude']:,.0f}")
        print(f"Fine-tuned avg error: {finetuned_analysis['avg_error_magnitude']:,.0f}")
        
        magnitude_improvement = original_analysis['avg_error_magnitude'] - finetuned_analysis['avg_error_magnitude']
        print(f"Improvement: {magnitude_improvement:,.0f}")
    
    return original_results, finetuned_results, original_analysis, finetuned_analysis

def save_evaluation_results(results: List[Dict], analysis: Dict, filename: str):
    """Save evaluation results to file."""
    output = {
        "analysis": analysis,
        "results": results
    }
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved to: {filename}")

def main():
    """Main evaluation function."""
    print("=== ModelGenie Addition Model Evaluation ===")
    
    # Generate test problems
    print("Generating fresh test problems...")
    test_problems = generate_test_problems(num_problems=50)  # Start with smaller set
    print(f"Generated {len(test_problems)} test problems")
    
    # Available models for testing
    print("\nAvailable models for comparison:")
    print("1. deepseek-r1:1.5b (original)")
    print("2. Custom fine-tuned model")
    print("3. Both (comparison)")
    
    choice = input("Select option (1-3): ").strip()
    
    if choice == "1":
        # Test only original model
        model_name = "deepseek-r1:1.5b"
        results = test_model_with_ollama(model_name, test_problems)
        analysis = analyze_results(results)
        
        print("\n" + "="*40)
        print("EVALUATION RESULTS")
        print("="*40)
        print(f"Model: {model_name}")
        print(f"Accuracy: {analysis['accuracy']:.2f}%")
        print(f"Correct: {analysis['correct_answers']}/{analysis['total_problems']}")
        print(f"Parsing errors: {analysis['parsing_errors']}")
        print(f"Calculation errors: {analysis['calculation_errors']}")
        
        if "avg_error_magnitude" in analysis:
            print(f"Average error magnitude: {analysis['avg_error_magnitude']:,.0f}")
        
        save_evaluation_results(results, analysis, "evaluation_results_original.json")
    
    elif choice == "2":
        # Test fine-tuned model
        finetuned_model = input("Enter fine-tuned model name: ").strip()
        results = test_model_with_ollama(finetuned_model, test_problems)
        analysis = analyze_results(results)
        
        print("\n" + "="*40)
        print("EVALUATION RESULTS")
        print("="*40)
        print(f"Model: {finetuned_model}")
        print(f"Accuracy: {analysis['accuracy']:.2f}%")
        print(f"Correct: {analysis['correct_answers']}/{analysis['total_problems']}")
        print(f"Parsing errors: {analysis['parsing_errors']}")
        print(f"Calculation errors: {analysis['calculation_errors']}")
        
        if "avg_error_magnitude" in analysis:
            print(f"Average error magnitude: {analysis['avg_error_magnitude']:,.0f}")
        
        save_evaluation_results(results, analysis, "evaluation_results_finetuned.json")
    
    elif choice == "3":
        # Compare both models
        original_model = "deepseek-r1:1.5b"
        finetuned_model = input("Enter fine-tuned model name: ").strip()
        
        orig_results, fine_results, orig_analysis, fine_analysis = compare_models(
            original_model, finetuned_model, test_problems
        )
        
        # Save both results
        save_evaluation_results(orig_results, orig_analysis, "evaluation_results_original.json")
        save_evaluation_results(fine_results, fine_analysis, "evaluation_results_finetuned.json")
        
        # Save comparison
        comparison = {
            "original_model": original_model,
            "finetuned_model": finetuned_model,
            "original_analysis": orig_analysis,
            "finetuned_analysis": fine_analysis,
            "test_problems": test_problems
        }
        
        with open("evaluation_comparison.json", 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print("\nComparison results saved to: evaluation_comparison.json")

if __name__ == "__main__":
    main()
