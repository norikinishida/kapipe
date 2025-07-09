import math

from ... import utils


def ndcg_at_k(
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
    scores["ndcg_at_k"] = _ndcg_at_k(
        pred_contexts=pred_contexts,
        gold_contexts=gold_contexts,
        passage_to_identifier=passage_to_identifier
    )
    return scores


def _ndcg_at_k(pred_contexts, gold_contexts, passage_to_identifier):
    scores = {}

    k_list = [1, 2, 4, 5, 8, 10, 16, 20, 30, 32, 50, 64, 100, 128]

    ndcg_list = {k: [] for k in k_list}

    for pred_contexts_for_doc, gold_contexts_for_doc in zip(pred_contexts, gold_contexts):
        pred_passage_ids = [passage_to_identifier(p) for p in pred_contexts_for_doc["contexts"]]
        gold_passage_ids = [passage_to_identifier(p) for p in gold_contexts_for_doc["contexts"]]
        gold_passage_ids = set(gold_passage_ids)

        relevance_labels = [1 if pid in gold_passage_ids else 0 for pid in pred_passage_ids]

        for k in k_list:
            dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance_labels[:k]))
            # ideal_relevance_labels = sorted(relevance_labels, reverse=True)
            # idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_relecance_labels[:k]))
            ideal_relevance_labels = [1] * min(k, len(gold_passage_ids))
            idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_relevance_labels))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_list[k].append(ndcg)

    for k in k_list:
        scores[f"nDCG@{k}"] = (
            sum(ndcg_list[k]) / len(ndcg_list[k])
            if len(ndcg_list[k]) != 0 else 0.0
        ) * 100.0

    return scores