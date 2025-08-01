# launcher.py
import subprocess
import sys
import os

if __name__ == "__main__":
    print("--- Custom SageMaker Launcher Script (launcher.py) ---")

    # Log all SM_ environment variables to confirm they are present
    print("--- Environment Variables Seen by Custom Launcher ---")
    for key, value in os.environ.items():
        if key.startswith("SM_") or key.startswith("SAGEMAKER_"):
            print(f"  LAUNCHER SEES: {key}={value}")
    if not any(key.startswith("SM_") for key in os.environ):
         print("  LAUNCHER WARNING: No SM_ prefixed environment variables found!")
    print("--- End Environment Variables Seen by Custom Launcher ---")

    training_module = "src.train" # The module we want to run
    command = [sys.executable, "-m", training_module]

    # Pass through command-line arguments received by this launcher script
    # (from sagemaker_training.cli.train, which gets them from Estimator hyperparameters)
    if len(sys.argv) > 1:
        command.extend(sys.argv[1:])
    
    print(f"Custom Launcher executing command: {' '.join(command)}")
    
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        env=os.environ.copy() # Ensure environment is inherited
    )
    
    if process.stdout:
        for line in iter(process.stdout.readline, ''):
            print(line, end='', flush=True)
            
    process.wait()
    
    print(f"--- Custom Launcher: Subprocess '{training_module}' finished with exit code: {process.returncode} ---")
    
    if process.returncode != 0:
        sys.exit(process.returncode)