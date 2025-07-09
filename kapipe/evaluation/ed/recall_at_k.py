import numpy as np

from ... import utils


def recall_at_k(pred_path, gold_path, inkb=True):
    """
    Parameters
    ----------
    pred_path : str | list[Document]
    gold_path : str | list[Document]
    inkb : bool, optional
        by default True

    Returns
    -------
    dict[str, Any]
    """
    scores = {}

    # Load
    if isinstance(pred_path, str):
        pred_candidate_entities = utils.read_json(pred_path)
    else:
        pred_candidate_entities = pred_path
    assert isinstance(pred_candidate_entities, list)
    if isinstance(gold_path, str):
        gold_documents = utils.read_json(gold_path)
    else:
        gold_documents = gold_path
    assert isinstance(gold_documents, list)

    # Check
    assert len(pred_candidate_entities) == len(gold_documents)
    for pred_cands_for_doc, gold_doc in zip(pred_candidate_entities, gold_documents):
        assert pred_cands_for_doc["doc_key"] == gold_doc["doc_key"]

    # Evaluate
    key = "inkb_recall_at_k" if inkb else "recall_at_k"
    scores[key] = _recall_at_k(
        pred_candidate_entities=pred_candidate_entities,
        gold_documents=gold_documents,
        inkb=inkb
    )
    return scores


def _recall_at_k(pred_candidate_entities, gold_documents, inkb):
    """
    Parameters
    ----------
    pred_candidate_entities : list[dict[str, str | list[list[CandEntKeyInfo]]]]
    gold_documents : list[Document]
    inkb : bool

    Returns
    -------
    dict[str, Any]
    """
    scores = {}

    total_count_mentions = 0
    ranks = [] # list[int]
    for pred_cands_for_doc, gold_doc in zip(pred_candidate_entities, gold_documents):
        candidate_entities_for_mentions = pred_cands_for_doc["candidate_entities"]
        gold_mentions = gold_doc["mentions"]
        assert len(gold_mentions) == len(candidate_entities_for_mentions)
        # For InKB mode, we skip mentions without valid KB entities
        if inkb:
            # NOTE: Order should be candidates then mentions
            candidate_entities_for_mentions = [
                y for y_i, y in enumerate(candidate_entities_for_mentions)
                # if gold_mentions[y_i]["entity_id"] != "NOT-IN-KB"
                if gold_mentions[y_i]["in_kb"]
            ]
            gold_mentions = [
                y for y_i, y in enumerate(gold_mentions)
                # if gold_mentions[y_i]["entity_id"] != "NOT-IN-KB"
                if gold_mentions[y_i]["in_kb"]
            ]
        for m, cs in zip(gold_mentions, candidate_entities_for_mentions):
            entity_id = m["entity_id"]
            candidate_entity_ids = [c["entity_id"] for c in cs]
            if entity_id in candidate_entity_ids:
                ranks.append(candidate_entity_ids.index(entity_id))
            total_count_mentions += 1
    ranks = np.asarray(ranks)
    scores["total_count_mentions"] = total_count_mentions

    k_list = [1, 2, 4, 5, 8, 10, 16, 20, 30, 32, 50, 64, 100, 128]

    for k in k_list:
        total_count_recall_at_k = int((ranks < k).sum())
        scores[f"total_count_recall@{k}"] = total_count_recall_at_k

    for k in k_list:
        total_count_recall_at_k = float(scores[f"total_count_recall@{k}"])
        scores[f"recall@{k}"] = (
            total_count_recall_at_k / total_count_mentions
            if total_count_mentions != 0 else 0.0
        ) * 100.0

    return scores

