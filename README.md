# LLM Refactoring Evaluation Framework

This repository contains the experimental framework used to evaluate **Large Language Models (LLMs)** for automated Python code refactoring.

The evaluation measures how different LLMs perform when refactoring messy Python code into cleaner, maintainable implementations using two prompt strategies.

---

# Overview

The experiment compares **5 LLMs** and **2 prompt styles** across **5 benchmark programs**.

### Models evaluated

* GPT-4o
* GPT-5
* Gemini
* Claude
* Grok

### Prompt styles

* **P1 – Monolithic instruction**
* **P2 – Checklist constraints**

### Benchmarks

* basic_string
* sequential_minimum_optimization
* linear_discriminant_analysis
* simplex
* jacobi_iteration_method

Total experiment runs:

5 benchmarks × 5 models × 2 prompts = **50 refactoring outputs**

---

# Evaluation Metrics

The framework computes three metrics for each generated program.

### BLEU

Measures n-gram overlap between generated documentation and reference documentation.

### ROUGE-L

Measures longest common subsequence similarity between documentation artifacts.

### Pylint Score

Static analysis score indicating code quality, maintainability, and adherence to Python conventions.

---

# Repository Structure

```
llm_refactoring_evaluation
│
├── data
│   ├── messy_code
│   ├── reference_code
│   └── metadata
│
├── outputs
│   └── <benchmark folders containing LLM outputs>
│
├── scripts
│   ├── extract_docs.py
│   ├── compute_text_metrics.py
│   ├── pylint_runner.py
│   ├── evaluate_outputs.py
│   ├── aggregate_results.py
│   └── generate_tables.py
│
├── results
│   ├── raw_metrics.csv
│   ├── aggregated_metrics.csv
│   └── logs
│
└── analysis
    ├── plots
    └── tables_for_paper
```

---

# Running the Experiment

## 1 Install dependencies

```
pip install -r requirements.txt
```

Download tokenizer resources:

```
python
```

```python
import nltk
nltk.download("punkt")
```

---

## 2 Place LLM outputs

Each benchmark folder inside `outputs/` must contain the following files:

```
model_prompt.py
```

Example:

```
gpt4o_p1.py
gpt4o_p2.py
gemini_p1.py
gemini_p2.py
```

---

## 3 Run evaluation

```
python scripts/evaluate_outputs.py
```

This will generate:

```
results/raw_metrics.csv
```

---

## 4 Aggregate results

```
python scripts/aggregate_results.py
```

This computes averages across the five benchmarks.

Output:

```
results/aggregated_metrics.csv
```

---

## 5 Generate paper tables

```
python scripts/generate_tables.py
```

Tables will be stored in:

```
analysis/tables_for_paper/
```

---

# Output Files

### Raw experiment metrics

```
results/raw_metrics.csv
```

Contains 50 rows corresponding to each model-prompt-benchmark combination.

### Aggregated results

```
results/aggregated_metrics.csv
```

Average scores across benchmarks.

### Figures

```
analysis/plots/
```

Visualizations of BLEU, ROUGE, and Pylint performance.

---

# Reproducibility

All evaluation metrics are computed using open-source tools:

* **Pylint** for static code analysis
* **NLTK BLEU implementation**
* **ROUGE-Score library**

The pipeline ensures consistent evaluation across all models and prompts.

---

# License

This repository is intended for **research and educational purposes**.
