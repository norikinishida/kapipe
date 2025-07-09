from ... import utils


def fscore(
    pred_path,
    gold_path,
    inkb=True,
    skip_normalization=False,
    on_predicted_spans=False
):
    """
    Parameters
    ----------
    pred_path : str | list[Document]
    gold_path : str | list[Document]
    inkb : bool, optional
        by default True
    skip_normalization : bool, optional
        by default False
    on_predicted_spans : bool, optional
        by default False

    Returns
    -------
    _type_
        _description_
    """
    if on_predicted_spans:
        # We do not compute normalized accuracy for predicted spans
        assert skip_normalization

    scores= {}

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
    key1 = "inkb_fscore" if inkb else "fscore"
    key2 = "inkb_normalized_fscore" if inkb else "normalized_fscore"
    scores[key1] = _fscore(
        pred_documents=pred_documents,
        gold_documents=gold_documents,
        inkb=inkb,
        normalized=False,
        on_predicted_spans=on_predicted_spans
    )
    if not skip_normalization:
        scores[key2] = _fscore(
            pred_documents=pred_documents,
            gold_documents=gold_documents,
            inkb=inkb,
            normalized=True,
            on_predicted_spans=on_predicted_spans
        )
    return scores

def _fscore(
    pred_documents,
    gold_documents,
    inkb,
    normalized,
    on_predicted_spans
):
    """
    Parameters
    ----------
    pred_documents : list[Document]
    gold_documents : list[Document]
    inkb : bool
    normalized : bool
    on_predicted_spans : bool

    Returns
    -------
    dict[str, Any]
    """
    scores = {}

    total_count_mentions = 0
    total_count_mentions_with_candidates = 0
    total_count_correct = 0
    for pred_doc, gold_doc in zip(pred_documents, gold_documents):
        pred_mentions = pred_doc["mentions"]
        gold_mentions = gold_doc["mentions"]
        if not on_predicted_spans:
            assert len(pred_mentions) == len(gold_mentions)
        # For InKB mode, skip mentions without valid KB entities
        if inkb:
            # NOTE: Order should be: pred -> gold
            if not on_predicted_spans:
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
        pred_mentions = set([
            (tuple(y["span"]), y["entity_id"]) for y in pred_mentions
            if y["entity_id"] != "NO-PRED"
        ])
        gold_mentions = set([
            (tuple(y["span"]), y["entity_id"]) for y in gold_mentions
        ])
        total_count_mentions += len(gold_mentions)
        total_count_mentions_with_candidates += len(pred_mentions)
        total_count_correct += len(pred_mentions & gold_mentions)
    scores["total_count_mentions"] = total_count_mentions
    scores["total_count_mentions_with_candidates"] \
        = total_count_mentions_with_candidates
    scores["total_count_correct"] = total_count_correct

    total_count_mentions = float(total_count_mentions)
    total_count_mentions_with_candidates = float(total_count_mentions_with_candidates)
    total_count_correct = float(total_count_correct)
    precision = (
        total_count_correct / total_count_mentions_with_candidates
        if total_count_mentions_with_candidates != 0 else 0.0
    )
    recall = (
        total_count_correct / total_count_mentions
        if total_count_mentions != 0 else 0.0
    )
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * (precision * recall) / (precision + recall)
    scores["precision"] = precision * 100.0
    scores["recall"] = recall * 100.0
    scores["f1"] = f1 * 100.0

    return scores

