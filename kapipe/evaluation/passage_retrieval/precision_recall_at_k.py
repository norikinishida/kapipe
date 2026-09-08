from ... import utils


def precision_recall_at_k(
    pred_path: str | list[dict],
    gold_path: str | list[dict]
) -> dict[str, float]:
    scores = {}

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
        gold_contexts=gold_contexts
    )
    return scores


def _precision_recall_at_k(
    pred_contexts: list[dict],
    gold_contexts: list[dict]
) -> dict[str, float]:
    scores: dict[str, float] = {}

    # Define the list of k values for which to compute precision and recall
    k_list: list[int] = [1, 2, 4, 5, 8, 10, 16, 20, 30, 32, 50, 64, 100, 128]

    # Initialize a counter dictionary to keep track of total predicted, gold, and correct counts for each k
    counter = {
        k: {
            "total_count_pred": 0,
            "total_count_gold": 0,
            "total_count_correct": 0
        }
        for k in k_list
    }

    for pred_contexts_for_doc, gold_contexts_for_doc in zip(pred_contexts, gold_contexts):
        # Extract predicted passage keys for the current document
        pred_passage_keys = [p["passage_key"] for p in pred_contexts_for_doc["contexts"]]

        # Remove duplicate predicted passage keys while preserving order
        unique_pred_passage_keys = []
        seen_pred_passage_keys = set()
        for passage_key in pred_passage_keys:
            if passage_key in seen_pred_passage_keys:
                continue
            unique_pred_passage_keys.append(passage_key)
            seen_pred_passage_keys.add(passage_key)
        pred_passage_keys = unique_pred_passage_keys

        # Extract gold passage keys for the current document 
        # and convert to a set for fast lookup.
        gold_passage_keys = [p["passage_key"] for p in gold_contexts_for_doc["contexts"]]
        gold_passage_keys = set(gold_passage_keys)

        # Compute precision and recall at each k for the current document
        for k in k_list:
            topk_pred_passage_keys = set(pred_passage_keys[:k])
            counter[k]["total_count_pred"] += len(topk_pred_passage_keys)
            counter[k]["total_count_gold"] += len(gold_passage_keys)
            counter[k]["total_count_correct"] += len(topk_pred_passage_keys & gold_passage_keys)

    # Compute precision and recall at each k by aggregating over all documents
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
