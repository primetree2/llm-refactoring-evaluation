import os
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_RESULTS = os.path.join(BASE_DIR, "results", "raw_metrics.csv")

AGG_RESULTS = os.path.join(BASE_DIR, "results", "aggregated_metrics.csv")
MODEL_RANKING = os.path.join(BASE_DIR, "results", "model_ranking.csv")
PROMPT_COMPARISON = os.path.join(BASE_DIR, "results", "prompt_comparison.csv")


def aggregate():

    df = pd.read_csv(RAW_RESULTS)

    # Average metrics by model + prompt
    agg = df.groupby(["model", "prompt"]).mean(numeric_only=True).reset_index()

    agg.to_csv(AGG_RESULTS, index=False)

    print("Saved aggregated metrics.")

    # Model ranking by overall score
    ranking = agg.sort_values("overall", ascending=False)

    ranking.to_csv(MODEL_RANKING, index=False)

    print("Saved model ranking.")

    # Prompt comparison
    prompt_comp = df.groupby("prompt").mean(numeric_only=True).reset_index()

    prompt_comp.to_csv(PROMPT_COMPARISON, index=False)

    print("Saved prompt comparison.")


if __name__ == "__main__":
    aggregate()