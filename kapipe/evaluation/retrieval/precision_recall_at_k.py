from ... import utils


def precision_recall_at_k(
    pred_path,
    gold_path,
    passage_to_identifier=None
):
    scores = {}

    if passage_to_identifier is None:
        passage_to_identifier = lambda p: p["passage_id"]

    # Load
    if isinstance(pred_path, str):
        pred_contexts = utils.read_json(pred_path)
    else:
        pred_contexts = pred_path
    assert isinstance(pred_contexts, list)

    if isinstance(gold_path, str):
        gold_contexts = utils.read_json(gold_path)
    else:
        gold_contexts = gold_path
    assert isinstance(gold_contexts, list)

    # Check
    assert len(pred_contexts) == len(gold_contexts)
    for pred_contexts_for_doc, gold_contexts_for_doc in zip(pred_contexts, gold_contexts):
        assert pred_contexts_for_doc["question_key"] == gold_contexts_for_doc["question_key"]

    # Evaluate
    scores["precision_recall_at_k"] = _precision_recall_at_k(
        pred_contexts=pred_contexts,
        gold_contexts=gold_contexts,
        passage_to_identifier=passage_to_identifier
    )
    return scores


def _precision_recall_at_k(pred_contexts, gold_contexts, passage_to_identifier):
    scores = {}

    k_list = [1, 2, 4, 5, 8, 10, 16, 20, 30, 32, 50, 64, 100, 128]

    counter = {
        k: {
            "total_count_pred": 0,
            "total_count_gold": 0,
            "total_count_correct": 0
        }
        for k in k_list
    }

    for pred_contexts_for_doc, gold_contexts_for_doc in zip(pred_contexts, gold_contexts):
        pred_passage_ids = [passage_to_identifier(p) for p in pred_contexts_for_doc["contexts"]]
        gold_passage_ids = [passage_to_identifier(p) for p in gold_contexts_for_doc["contexts"]]
        gold_passage_ids = set(gold_passage_ids)

        for k in k_list:
            topk_pred_passage_ids = set(pred_passage_ids[:k])
            counter[k]["total_count_pred"] += len(topk_pred_passage_ids)
            counter[k]["total_count_gold"] += len(gold_passage_ids)
            counter[k]["total_count_correct"] += len(topk_pred_passage_ids & gold_passage_ids)

    for k in k_list:
        total_count_pred = float(counter[k]["total_count_pred"])
        total_count_gold = float(counter[k]["total_count_gold"])
        total_count_correct = float(counter[k]["total_count_correct"])

        precision_at_k = (
            total_count_correct / total_count_pred
            if total_count_pred != 0 else 0.0
        )
        recall_at_k = (
            total_count_correct / total_count_gold
            if total_count_gold != 0 else 0.0
        )
        # if precition + recall == 0:
        #     f1 = 0.0
        # else:
        #     f1 = 2.0 * (precision * recall) / (precision + recall)

        scores[f"precision@{k}"] = precision_at_k * 100.0
        scores[f"recall@{k}"] = recall_at_k * 100.0

    return scores

