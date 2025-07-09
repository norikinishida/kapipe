from ... import utils


def accuracy(pred_path, gold_path, exact_match=False):
    """
    Parameters
    ----------
    pred_path : str | list[Question]
    gold_path : str | list[Question]
    exact_match : bool
        by default False

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
    if isinstance(gold_path, str):
        gold_questions = utils.read_json(gold_path)
    else:
        gold_questions = gold_path
    assert isinstance(gold_questions, list)

    # Check
    assert len(pred_questions) == len(gold_questions)
    for pred_question, gold_question in zip(pred_questions, gold_questions):
        assert pred_question["question_key"] == gold_question["question_key"]

    # Evaluate
    scores["accuracy"] = _accuracy(
        pred_questions=pred_questions,
        gold_questions=gold_questions,
        exact_match=exact_match
    )
    return scores

    
def _accuracy(pred_questions, gold_questions, exact_match):
    scores = {}
    
    total_count = 0
    total_count_correct = 0

    for pred_question, gold_question in zip(pred_questions, gold_questions):
        pred_ans_str = pred_question["output_answer"].lower()
        total_count += 1
        for gold_ans in gold_question["answers"]:
            gold_ans_str = gold_ans["answer"].lower()
            if exact_match:
                if pred_ans_str == gold_ans_str:
                    total_count_correct += 1
                    break
            else:
                if pred_ans_str in gold_ans_str or gold_ans_str in pred_ans_str:
                    total_count_correct += 1
                    break
        
    scores["total_count"] = total_count
    scores["total_count_correct"] = total_count_correct

    total_count = float(total_count)
    total_count_correct = float(total_count_correct)
    scores["accuracy"] = (
        total_count_correct / total_count
        if total_count != 0 else 0.0
    ) * 100.0

    return scores