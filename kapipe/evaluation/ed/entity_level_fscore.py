from ... import utils


def entity_level_fscore(
    pred_path,
    gold_path,
):
    """
    Parameters
    ----------
    pred_path : str | list[Document]
    gold_path : str | list[Document]

    Returns
    -------
    dict[str,float]
    """
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
    scores["entity_level_fscore"] = _entity_level_fscore(
        pred_documents=pred_documents,
        gold_documents=gold_documents
    )
    return scores


def _entity_level_fscore(
    pred_documents,
    gold_documents
):
    scores = {}

    total_count_gold_entities = 0
    total_count_pred_entities = 0
    total_count_correct = 0
    for pred_doc, gold_doc in zip(pred_documents, gold_documents):
        pred_entities = pred_doc["entities"]
        gold_entities = gold_doc["entities"]
        pred_entities = set([
            y["entity_id"] for y in pred_entities
            if y["entity_id"] != "NO-PRED"
        ])
        gold_entities = set([
            y["entity_id"] for y in gold_entities
            if y["entity_id"] != "NO-PRED"
        ])
        total_count_pred_entities += len(pred_entities)
        total_count_gold_entities += len(gold_entities)
        total_count_correct += len(pred_entities & gold_entities)
    scores["total_count_pred_entities"] = total_count_pred_entities
    scores["total_count_gold_entities"] = total_count_gold_entities
    scores["total_count_correct"] = total_count_correct

    total_count_pred_entities = float(total_count_pred_entities)
    total_count_gold_entities = float(total_count_gold_entities)
    total_count_correct = float(total_count_correct)
    precision = total_count_correct / total_count_pred_entities
    recall = total_count_correct / total_count_gold_entities
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * (precision * recall) / (precision + recall)
    scores["precision"] = precision * 100.0
    scores["recall"] = recall * 100.0
    scores["f1"] = f1 * 100.0

    return scores
