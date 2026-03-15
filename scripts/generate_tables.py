import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AGG_RESULTS = os.path.join(BASE_DIR, "results", "aggregated_metrics.csv")

TABLE_DIR = os.path.join(BASE_DIR, "analysis", "tables_for_paper")
PLOT_DIR = os.path.join(BASE_DIR, "analysis", "plots")


def generate_tables():

    os.makedirs(TABLE_DIR, exist_ok=True)

    df = pd.read_csv(AGG_RESULTS)

    # Sort by overall score (better presentation)
    df = df.sort_values("overall", ascending=False)

    # Evaluation matrix table
    df.to_csv(
        os.path.join(TABLE_DIR, "evaluation_matrix.csv"),
        index=False
    )

    # Prompt effectiveness comparison
    prompt_effect = df.groupby("prompt").mean(numeric_only=True).reset_index()

    prompt_effect.to_csv(
        os.path.join(TABLE_DIR, "prompt_effectiveness.csv"),
        index=False
    )

    print("Saved paper tables.")


def generate_plots():

    os.makedirs(PLOT_DIR, exist_ok=True)

    df = pd.read_csv(AGG_RESULTS)

    sns.set_theme(style="whitegrid")

    # BLEU plot
    plt.figure(figsize=(8,5))
    sns.barplot(data=df, x="model", y="bleu", hue="prompt")
    plt.title("BLEU Score Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "bleu_scores.png"))
    plt.close()

    # ROUGE plot
    plt.figure(figsize=(8,5))
    sns.barplot(data=df, x="model", y="rouge", hue="prompt")
    plt.title("ROUGE-L Score Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "rouge_scores.png"))
    plt.close()

    # Pylint plot
    plt.figure(figsize=(8,5))
    sns.barplot(data=df, x="model", y="pylint", hue="prompt")
    plt.title("Pylint Score Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "pylint_scores.png"))
    plt.close()

    # Overall score
    plt.figure(figsize=(8,5))
    sns.barplot(data=df, x="model", y="overall", hue="prompt")
    plt.title("Overall Performance Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "overall_scores.png"))
    plt.close()

    print("Saved plots.")


if __name__ == "__main__":

    generate_tables()

    generate_plots()