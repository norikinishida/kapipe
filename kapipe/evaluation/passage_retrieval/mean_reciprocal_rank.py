from ... import utils


def mean_reciprocal_rank(
    pred_path: str | list[dict],
    gold_path: str | list[dict]
) -> dict[str, float]:
    scores: dict[str, float] = {}

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
    scores["mean_reciprocal_rank"] = _mean_reciprocal_rank(
        pred_contexts=pred_contexts,
        gold_contexts=gold_contexts
    )
    return scores


def _mean_reciprocal_rank(
    pred_contexts: list[dict],
    gold_contexts: list[dict]
) -> dict[str, float]:
    scores: dict[str, float] = {}

    reciprocal_ranks: list[float] = []

    for pred_contexts_for_doc, gold_contexts_for_doc in zip(pred_contexts, gold_contexts):
        # Extract predicted passage keys for the current document
        pred_passage_keys = [p["passage_key"] for p in pred_contexts_for_doc["contexts"]]

        # Remove duplicate predicted passage keys while preserving order
        unique_pred_passage_keys = []
        seen_pred_passage_keys = set()
        for pred_passage_key in pred_passage_keys:
            if pred_passage_key in seen_pred_passage_keys:
                continue
            unique_pred_passage_keys.append(pred_passage_key)
            seen_pred_passage_keys.add(pred_passage_key)
        pred_passage_keys = unique_pred_passage_keys

        # Extract gold passage keys for the current document
        # and convert to a set for fast lookup.
        gold_passage_keys = [p["passage_key"] for p in gold_contexts_for_doc["contexts"]]
        gold_passage_keys = set(gold_passage_keys)

        # Compute reciprocal rank for the current document
        rr = 0.0
        for rank, pid in enumerate(pred_passage_keys, 1):
            if pid in gold_passage_keys:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    # Compute mean reciprocal rank across all documents
    sum_ = sum(reciprocal_ranks)
    n = len(reciprocal_ranks)
    scores["mean_reciprocal_rank"] = (sum_ / n if n != 0 else 0.0) * 100.0

    return scores
