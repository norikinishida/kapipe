from ... import utils


def mean_reciprocal_rank(
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
    scores["mean_reciprocal_rank"] = _mean_reciprocal_rank(
        pred_contexts=pred_contexts,
        gold_contexts=gold_contexts,
        passage_to_identifier=passage_to_identifier
    )
    return scores


def _mean_reciprocal_rank(pred_contexts, gold_contexts, passage_to_identifier):
    scores = {}

    reciprocal_ranks = []

    for pred_contexts_for_doc, gold_contexts_for_doc in zip(pred_contexts, gold_contexts):
        pred_passage_ids = [passage_to_identifier(p) for p in pred_contexts_for_doc["contexts"]]
        gold_passage_ids = [passage_to_identifier(p) for p in gold_contexts_for_doc["contexts"]]
        gold_passage_ids = set(gold_passage_ids)

        rr = 0.0
        for rank, pid in enumerate(pred_passage_ids, 1):
            if pid in gold_passage_ids:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    sum_ = sum(reciprocal_ranks)
    n = len(reciprocal_ranks)
    scores["mean_reciprocal_rank"] = (sum_ / n if n != 0 else 0.0) * 100.0
    return scores