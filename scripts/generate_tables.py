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
    os.makedirs(PLOT_DIR, exist_ok=True)

    df = pd.read_csv(AGG_RESULTS)

    # Evaluation matrix table
    eval_matrix = df.copy()

    eval_matrix.to_csv(
        os.path.join(TABLE_DIR, "evaluation_matrix.csv"),
        index=False
    )

    # Prompt effectiveness
    prompt_effect = df.groupby("prompt").mean(numeric_only=True).reset_index()

    prompt_effect.to_csv(
        os.path.join(TABLE_DIR, "prompt_effectiveness.csv"),
        index=False
    )

    print("Saved paper tables.")


def generate_plots():

    df = pd.read_csv(AGG_RESULTS)

    sns.set(style="whitegrid")

    # BLEU plot
    plt.figure()
    sns.barplot(data=df, x="model", y="bleu", hue="prompt")
    plt.title("BLEU Score Comparison")
    plt.savefig(os.path.join(PLOT_DIR, "bleu_scores.png"))

    # ROUGE plot
    plt.figure()
    sns.barplot(data=df, x="model", y="rouge", hue="prompt")
    plt.title("ROUGE Score Comparison")
    plt.savefig(os.path.join(PLOT_DIR, "rouge_scores.png"))

    # Pylint plot
    plt.figure()
    sns.barplot(data=df, x="model", y="pylint", hue="prompt")
    plt.title("Pylint Score Comparison")
    plt.savefig(os.path.join(PLOT_DIR, "pylint_scores.png"))

    # Overall score
    plt.figure()
    sns.barplot(data=df, x="model", y="overall", hue="prompt")
    plt.title("Overall Performance Comparison")
    plt.savefig(os.path.join(PLOT_DIR, "overall_scores.png"))

    print("Saved plots.")


if __name__ == "__main__":

    generate_tables()

    generate_plots()