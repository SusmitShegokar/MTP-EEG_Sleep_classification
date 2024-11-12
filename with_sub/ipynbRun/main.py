import subprocess
import os
from datetime import datetime
import glob
import torch

# Check if CUDA is available and print whether using CUDA or CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Define paths to scripts
dataloader_script = './dataloader.py'
model_runner_script = './model_runner.py'
evaluator_script = './evaluator.py'

# Log file path
log_dir = './output/'
log_file = os.path.join(log_dir, 'pipeline_log.txt')

# Ensure the log directory exists
os.makedirs(log_dir, exist_ok=True)

# Function to log both to the console and to a file
def log(message):
    print(message)
    with open(log_file, 'a') as f:
        f.write(f"{message}\n")

# Function to run a Python script and log the output
def run_script(script_path, *args):
    command = ['python3', script_path] + list(args)
    try:
        log(f"Running: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        log(result.stdout)
    except subprocess.CalledProcessError as e:
        log(f"Error running {script_path}:")
        log(e.stderr)

def main():
    # Start logging
    log(f"\n{'='*40}\nStarting pipeline at {datetime.now()}\n{'='*40}\n")

    # Step 1: Run dataloader.py to load data
    log("Running dataloader.py...")
    run_script(dataloader_script)

    # Step 2: Iterate over each model in the modelpy/ directory
    model_files = glob.glob('./modelpy/*.py')  # List all .py files in modelpy/

    for model_py_path in model_files:
        # Extract the base model name (without .py extension)
        model_name = os.path.basename(model_py_path).replace('.py', '')

        log(f"\nRunning model: {model_name}")

        # Step 3: Run model_runner.py with model.py as argument
        log(f"Running model_runner.py with argument: {model_py_path}")
        run_script(model_runner_script, model_py_path)
        
        # Path for the corresponding .pt file (adjust this according to how your models are saved)
        model_pt_path = f'/output/{model_name}_final_model.pt'


        # Step 4: Run evaluator.py with model.pt and model.py as arguments
        log(f"Running evaluator.py with arguments: {model_pt_path}, {model_py_path}")
        run_script(evaluator_script, model_pt_path, model_py_path)

    log(f"\nPipeline completed at {datetime.now()}\n{'='*40}\n")

if __name__ == '__main__':
    main()
