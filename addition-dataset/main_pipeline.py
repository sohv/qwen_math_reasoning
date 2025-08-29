#!/usr/bin/env python3
"""
ModelGenie Addition Dataset Pipeline
Replicates the experiment from https://shukraditya.notion.site/can-llms-do-math

Steps:
1. Dataset Generation (already done via Jupyter notebook)
2. Error Detection
3. Critic Model Correction
4. SFT Pair Generation
5. Fine-tuning (optional)
6. Evaluation

Usage:
    python main_pipeline.py --step 2  # Run specific step
    python main_pipeline.py --all     # Run all steps
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

def run_step(step_number: int, description: str, script_name: str):
    """Run a specific step in the pipeline."""
    print(f"\n{'='*60}")
    print(f"STEP {step_number}: {description}")
    print('='*60)
    
    if not os.path.exists(script_name):
        print(f"Error: {script_name} not found!")
        return False
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}:")
        print(f"Return code: {e.returncode}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False

def check_prerequisites():
    """Check if required files exist."""
    required_files = [
        "data/additions_r1half-4.json",  # Generated dataset
        "2_error_detection.py",
        "3_critic_model.py", 
        "4_sft_pair_generation.py",
        "5_finetune_model.py",
        "6_evaluate_model.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        
        if "data/additions_r1half-4.json" in missing_files:
            print("\nPlease run the Jupyter notebook first to generate the dataset:")
            print("  jupyter notebook '1. dataset_generation_script.ipynb'")
        
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="ModelGenie Addition Dataset Pipeline")
    parser.add_argument("--step", type=int, choices=[2, 3, 4, 5, 6], 
                       help="Run specific step (2-6)")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    parser.add_argument("--skip-finetuning", action="store_true", 
                       help="Skip fine-tuning step (requires GPU)")
    
    args = parser.parse_args()
    
    if not args.step and not args.all:
        parser.print_help()
        return
    
    print("ModelGenie Addition Dataset Pipeline")
    print("Replicating: https://shukraditya.notion.site/can-llms-do-math")
    
    # Check prerequisites
    if not check_prerequisites():
        print("\nPlease fix the missing files before continuing.")
        return
    
    # Define pipeline steps
    steps = [
        (2, "Error Detection", "2_error_detection.py"),
        (3, "Critic Model Correction", "3_critic_model.py"),
        (4, "SFT Pair Generation", "4_sft_pair_generation.py"),
        (5, "Fine-tuning", "5_finetune_model.py"),
        (6, "Model Evaluation", "6_evaluate_model.py")
    ]
    
    if args.skip_finetuning:
        steps = [s for s in steps if s[0] != 5]
    
    # Run specific step or all steps
    if args.step:
        step_info = next((s for s in steps if s[0] == args.step), None)
        if step_info:
            success = run_step(*step_info)
            if not success:
                print(f"Step {args.step} failed!")
                return
        else:
            print(f"Invalid step: {args.step}")
            return
    
    elif args.all:
        print("\nRunning complete pipeline...")
        
        for step_num, description, script in steps:
            if args.skip_finetuning and step_num == 5:
                print(f"\nSkipping Step {step_num}: {description} (--skip-finetuning)")
                continue
                
            success = run_step(step_num, description, script)
            if not success:
                print(f"\nPipeline failed at Step {step_num}!")
                print("You can resume from this step using:")
                print(f"  python main_pipeline.py --step {step_num}")
                return
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nGenerated files:")
        
        output_files = [
            "data/processed_additions_with_errors.json",
            "data/critic_corrected_additions.json", 
            "data/sft_pairs_qwen.json",
            "data/sft_pairs_llama.json",
            "data/sft_pairs_unsloth.json",
            "evaluation_results_original.json"
        ]
        
        for file in output_files:
            if os.path.exists(file):
                print(f"  [FOUND] {file}")
            else:
                print(f"  [MISSING] {file} (not found)")

def show_usage():
    """Show detailed usage instructions."""
    print("""
ModelGenie Addition Dataset Pipeline Usage:

1. Generate Dataset (Step 1):
   jupyter notebook '1. dataset_generation_script.ipynb'

2. Run Error Detection (Step 2):
   python main_pipeline.py --step 2

3. Run Critic Model Correction (Step 3):
   python main_pipeline.py --step 3
   (Requires Gemini API key or local Ollama model)

4. Generate SFT Pairs (Step 4):
   python main_pipeline.py --step 4

5. Fine-tune Model (Step 5 - Optional):
   python main_pipeline.py --step 5
   (Requires GPU and additional packages)

6. Evaluate Model (Step 6):
   python main_pipeline.py --step 6

Run all steps at once:
   python main_pipeline.py --all
   python main_pipeline.py --all --skip-finetuning

Individual steps:
   python 2_error_detection.py
   python 3_critic_model.py
   python 4_sft_pair_generation.py
   python 5_finetune_model.py
   python 6_evaluate_model.py
""")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        show_usage()
    else:
        main()
