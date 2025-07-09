from ... import utils


def accuracy(pred_path, gold_path, inkb=True, skip_normalization=False):
    """
    Parameters
    ----------
    pred_path : str | list[Document]
    gold_path : str | list[Document]
    inkb : bool, optional
        by default True
    skip_normalization : bool, optional
        by default False

    Returns
    -------
    dict[str, Any]
    """
    scores = {}

    # Load
    if isinstance(pred_path, str):
        pred_documents = utils.read_json(pred_path)
    else:
        pred_documents = pred_path
    assert isinstance(pred_documents, list)
    if isinstance(gold_path, str):
        gold_documents = utils.read_json(gold_path)
    else:
        gold_documents = gold_path
    assert isinstance(gold_documents, list)

    # Check
    assert len(pred_documents) == len(gold_documents)
    for pred_doc, gold_doc in zip(pred_documents, gold_documents):
        assert pred_doc["doc_key"] == gold_doc["doc_key"]

    # Evaluate
    key1 = "inkb_accuracy" if inkb else "accuracy"
    key2 = "inkb_normalized_accuracy" if inkb else "normalized_accuracy"
    scores[key1] = _accuracy(
        pred_documents=pred_documents,
        gold_documents=gold_documents,
        inkb=inkb,
        normalized=False
    )
    if not skip_normalization:
        scores[key2] = _accuracy(
            pred_documents=pred_documents,
            gold_documents=gold_documents,
            inkb=inkb,
            normalized=True
        )

    return scores


def _accuracy(pred_documents, gold_documents, inkb, normalized):
    """
    Parameters
    ----------
    pred_documents : list[Document]
    gold_documents : list[Document]
    inkb : bool
    normalized : bool

    Returns
    -------
    dict[str, Any]
    """
    scores = {}

    total_count_mentions = 0
    total_count_correct = 0
    for pred_doc, gold_doc in zip(pred_documents, gold_documents):
        pred_mentions = pred_doc["mentions"]
        gold_mentions = gold_doc["mentions"]
        assert len(pred_mentions) == len(gold_mentions)
        # For InKB mode, skip mentions without valid KB entities
        if inkb:
            # NOTE: Order should be: pred -> gold
            pred_mentions = [
                y for y_i, y in enumerate(pred_mentions)
                if gold_mentions[y_i]["in_kb"]
            ]
            gold_mentions = [
                y for y_i, y in enumerate(gold_mentions)
                if gold_mentions[y_i]["in_kb"]
            ]
        # For normalized mode,
        #   skip mentions without gold entities in the candidates.
        if normalized:
            # NOTE: Order should be: pred -> gold
            pred_mentions = [
                y for y_i, y in enumerate(pred_mentions)
                if gold_mentions[y_i]["in_cand"]
            ]
            gold_mentions = [
                y for y_i, y in enumerate(gold_mentions)
                if gold_mentions[y_i]["in_cand"]
            ]
        for pred_m, gold_m in zip(pred_mentions, gold_mentions):
            assert tuple(pred_m["span"])  == tuple(gold_m["span"])
            total_count_mentions += 1
            if pred_m["entity_id"] == gold_m["entity_id"]:
                total_count_correct += 1
    scores["total_count_mentions"] = total_count_mentions
    scores["total_count_correct"] = total_count_correct

    total_count_mentions = float(total_count_mentions)
    total_count_correct = float(total_count_correct)
    acc = (
        total_count_correct / total_count_mentions
        if total_count_mentions != 0 else 0.0
    )
    scores["accuracy"] = acc * 100.0

    return scores

