from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

from extract_docs import extract_documentation


def compute_bleu(reference_text, candidate_text):
    """
    Compute BLEU score between reference and candidate documentation.
    """

    reference_tokens = reference_text.split()
    candidate_tokens = candidate_text.split()

    if len(candidate_tokens) == 0:
        return 0.0

    score = sentence_bleu([reference_tokens], candidate_tokens)

    return score


def compute_rouge(reference_text, candidate_text):
    """
    Compute ROUGE-L score.
    """

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    scores = scorer.score(reference_text, candidate_text)

    return scores["rougeL"].fmeasure


def compute_text_metrics(reference_file, candidate_file):
    """
    Compute both BLEU and ROUGE scores between files.
    """

    reference_text = extract_documentation(reference_file)
    candidate_text = extract_documentation(candidate_file)

    bleu = compute_bleu(reference_text, candidate_text)
    rouge = compute_rouge(reference_text, candidate_text)

    return bleu, rouge