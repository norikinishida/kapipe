import re
from collections import Counter

from ... import utils


def token_level_f1(pred_path, gold_path):
    """
    Parameters
    ----------
    pred_path : str | list[Question]
    gold_path : str | list[Question]

    Returns
    -------
    dict[str, Any]
    """
    scores = {}

    # Load
    if isinstance(pred_path, str):
        pred_questions = utils.read_json(pred_path)
    else:
        pred_questions = pred_path
    assert isinstance(pred_questions, list)

    # Load gold answers
    if isinstance(gold_path, str):
        gold_questions = utils.read_json(gold_path)
    else:
        gold_questions = gold_path
    assert isinstance(gold_questions, list)

    # Check
    assert len(pred_questions) == len(gold_questions)
    for pred_question, gold_question in zip(pred_questions, gold_questions):
        assert pred_question["question_key"] == gold_question["question_key"]

    # Evaluate predictions
    scores["token_level_f1"] = _token_level_f1(
        pred_questions=pred_questions,
        gold_questions=gold_questions,
    )

    return scores


def _token_level_f1(pred_questions, gold_questions):
    scores = {}

    # Initialize accumulators
    total_count = 0
    total_score = 0.0

    for pred_question, gold_question in zip(pred_questions, gold_questions):
        # Normalize the predicted answer in the same style as accuracy.py
        pred_ans_str = pred_question["output_answer"].lower()

        # Count one question (= one gold answer)
        total_count += 1

        # Find the best score over acceptable gold-answer synonyms
        best_score = 0.0
        for gold_ans in gold_question["answers"]:
            # Normalize the gold-answer synonym
            gold_ans_str = gold_ans["answer"].lower()

            # Compute a token-level F1 score
            score = _compute_token_level_f1(
                pred_ans_str=pred_ans_str,
                gold_ans_str=gold_ans_str
            )

            # Keep the best synonym score
            best_score = max(best_score, score)

        # Accumulate the best score
        total_score += best_score

    scores["total_count"] = total_count
    scores["total_score"] = total_score

    total_count = float(total_count)
    scores["f1"] = (
        total_score / total_count
        if total_count != 0 else 0.0
    ) * 100.0

    return scores


def _compute_token_level_f1(pred_ans_str, gold_ans_str):
    # Split answers into simple word tokens
    pred_tokens = _tokenize_answer(pred_ans_str)
    gold_tokens = _tokenize_answer(gold_ans_str)

    # Handle empty answers
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 1.0 if pred_tokens == gold_tokens else 0.0

    # Count shared tokens
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    # Return zero when there is no token overlap
    if num_same == 0:
        return 0.0

    # Compute precision
    precision = num_same / float(len(pred_tokens))

    # Compute recall
    recall = num_same / float(len(gold_tokens))

    # Compute F1
    return 2.0 * precision * recall / (precision + recall)


def _tokenize_answer(answer):
    # Remove punctuation-like characters by keeping word tokens
    tokens = re.findall(r"\w+", answer)

    # Remove English articles used by SQuAD-style normalization
    # tokens = [
    #     token
    #     for token in tokens
    #     if token not in ["a", "an", "the"]
    # ]

    return tokens