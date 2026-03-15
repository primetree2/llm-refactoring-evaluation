import os
import pandas as pd

from compute_text_metrics import compute_text_metrics
from pylint_runner import get_pylint_score


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
REFERENCE_DIR = os.path.join(BASE_DIR, "data", "reference_code")
RESULTS_FILE = os.path.join(BASE_DIR, "results", "raw_metrics.csv")


def evaluate():

    rows = []

    for benchmark in os.listdir(OUTPUT_DIR):

        benchmark_path = os.path.join(OUTPUT_DIR, benchmark)

        if not os.path.isdir(benchmark_path):
            continue

        reference_file = os.path.join(
            REFERENCE_DIR,
            f"{benchmark}_clean.py"
        )

        for file in os.listdir(benchmark_path):

            if not file.endswith(".py"):
                continue

            candidate_file = os.path.join(benchmark_path, file)

            name = file.replace(".py", "")
            model, prompt = name.split("_")

            print(f"Evaluating {benchmark} | {model} | {prompt}")

            bleu, rouge = compute_text_metrics(reference_file, candidate_file)

            pylint_score = get_pylint_score(candidate_file)

            overall = (bleu + rouge + (pylint_score / 10)) / 3

            rows.append({
                "benchmark": benchmark,
                "model": model,
                "prompt": prompt,
                "bleu": bleu,
                "rouge": rouge,
                "pylint": pylint_score,
                "overall": overall
            })

    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

    df.to_csv(RESULTS_FILE, index=False)

    print("\nEvaluation complete.")
    print(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    evaluate()