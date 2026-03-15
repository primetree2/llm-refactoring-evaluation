
import subprocess
import os
import sys
from datetime import datetime


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
LOG_DIR = os.path.join(RESULTS_DIR, "logs")

EVALUATE_SCRIPT = os.path.join(SCRIPTS_DIR, "evaluate_outputs.py")
AGGREGATE_SCRIPT = os.path.join(SCRIPTS_DIR, "aggregate_results.py")
GENERATE_TABLES_SCRIPT = os.path.join(SCRIPTS_DIR, "generate_tables.py")

LOG_FILE = os.path.join(LOG_DIR, "experiment_log.txt")


def log(message):
    """Write message to console and log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {message}"

    print(formatted)

    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(formatted + "\n")


def run_script(script_path):
    """Execute a Python script and capture errors."""

    log(f"Running {os.path.basename(script_path)}")

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=True,
            text=True
        )

        if result.stdout:
            print(result.stdout)

        if result.stderr:
            print(result.stderr)

        log(f"{os.path.basename(script_path)} completed successfully")

    except subprocess.CalledProcessError as e:

        log(f"ERROR running {os.path.basename(script_path)}")

        error_file = os.path.join(LOG_DIR, "errors.txt")

        with open(error_file, "a") as f:
            f.write(str(e.stderr))
            f.write("\n")

        print(e.stderr)
        sys.exit(1)


def main():

    log("=====================================")
    log("LLM Refactoring Evaluation Started")
    log("=====================================")

    # Step 1: Evaluate outputs
    run_script(EVALUATE_SCRIPT)

    # Step 2: Aggregate metrics
    run_script(AGGREGATE_SCRIPT)

    # Step 3: Generate tables for paper
    run_script(GENERATE_TABLES_SCRIPT)

    log("=====================================")
    log("Experiment completed successfully")
    log("Results stored in /results")
    log("=====================================")


if __name__ == "__main__":
    main()
