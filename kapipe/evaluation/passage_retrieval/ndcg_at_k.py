import math

from ... import utils


def ndcg_at_k(
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
    scores["ndcg_at_k"] = _ndcg_at_k(
        pred_contexts=pred_contexts,
        gold_contexts=gold_contexts
    )
    return scores


def _ndcg_at_k(
    pred_contexts: list[dict],
    gold_contexts: list[dict]
) -> dict[str, float]:
    scores: dict[str, float] = {}

    # Define the list of k values for which to compute nDCG@k
    k_list: list[int] = [1, 2, 4, 5, 8, 10, 16, 20, 30, 32, 50, 64, 100, 128]

    # Initialize a dictionary to store nDCG values for each k
    ndcg_list: dict[int, list[float]] = {k: [] for k in k_list}

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

        # Compute relevance labels for the predicted passages based on the gold passages
        relevance_labels = [
            1 if key in gold_passage_keys else 0
            for key in pred_passage_keys
        ]

        # Compute nDCG@k for each k in k_list
        for k in k_list:
            dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance_labels[:k]))
            # ideal_relevance_labels = sorted(relevance_labels, reverse=True)
            # idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_relecance_labels[:k]))
            ideal_relevance_labels = [1] * min(k, len(gold_passage_keys))
            idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_relevance_labels))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_list[k].append(ndcg)

    # Compute the final nDCG@k scores by averaging over all documents for each k
    for k in k_list:
        scores[f"nDCG@{k}"] = (
            sum(ndcg_list[k]) / len(ndcg_list[k])
            if len(ndcg_list[k]) != 0 else 0.0
        ) * 100.0

    return scores
