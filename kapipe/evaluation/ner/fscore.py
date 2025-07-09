from ... import utils


def fscore(pred_path, gold_path):
    """
    Parameters
    ----------
    pred_path : str | list[Document]
    gold_path : str | list[Document]

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
    # Span (exact match) & Type (exact match)
    scores["span_and_type"] = _fscore(
        pred_documents=pred_documents,
        gold_documents=gold_documents,
        span_overlap=False,
        ignore_type=False
    )
    # Span (overlap allowed) & Type (exact match)
    scores["span_overlap_and_type"] = _fscore(
        pred_documents=pred_documents,
        gold_documents=gold_documents,
        span_overlap=True,
        ignore_type=False
    )
    # Span (exact match)
    scores["span_only"] = _fscore(
        pred_documents=pred_documents,
        gold_documents=gold_documents,
        span_overlap=False,
        ignore_type=True
    )
    return scores


def _fscore(pred_documents, gold_documents, span_overlap, ignore_type):
    """
    Parameters
    ----------
    pred_documents : list[Document]
    gold_documents : list[Document]
    span_overlap : bool
    ignore_type : bool

    Returns
    -------
    dict[str, Any] 
    """
    scores = {}

    total_count_pred = 0
    total_count_gold = 0
    if not span_overlap:
        total_count_correct = 0
    else:
        total_count_correct_pred = 0 # used in span-overlap mode
        total_count_correct_gold = 0 # used in span-overlap mode

    for pred_doc, gold_doc in zip(pred_documents, gold_documents):
        pred_mentions = pred_doc["mentions"]
        gold_mentions = gold_doc["mentions"]
        # NOTE: In the `ignore_type=True` setting,
        #   we mask the gold and predicted entity types to a special symbol "*"
        pred_mentions = set([
            (
                tuple(y["span"]),
                y["entity_type"] if not ignore_type else "*"
            )
            for y in pred_mentions
        ])
        gold_mentions = set([
            (
                tuple(y["span"]),
                y["entity_type"] if not ignore_type else "*"
            )
            for y in gold_mentions
        ])
        total_count_pred += len(pred_mentions)
        total_count_gold += len(gold_mentions)
        if not span_overlap:
            total_count_correct += len(pred_mentions & gold_mentions)
        else:
            # e.g.,
            #   - pred_spans: [(1,2),(3,4)]
            #   - gold_spans: [(2,3),(5,6)] 
            #   - precision: 2/2
            #   - recall: 1/2
            for pred_span, pred_type in pred_mentions:
                # Check whether there is a gold mention that
                #   (1) overlaps and
                #   (2) has the same type
                #   with the predicted mention
                found_overlap = False
                for gold_span, gold_type in gold_mentions:
                    if (
                        find_overlap(pred_span, gold_span)
                        and
                        pred_type == gold_type
                    ):
                        found_overlap = True
                        break
                if found_overlap:
                    total_count_correct_pred += 1
            for gold_span, gold_type in gold_mentions:
                # Check whether there is a predicted mention that
                #   (1) overlaps and
                #   (2) has the same type
                #   with the gold mention
                found_overlap = False
                for pred_span, pred_type in pred_mentions:
                    if (
                        find_overlap(pred_span, gold_span)
                        and
                        pred_type == gold_type
                    ):
                        found_overlap = True
                        break
                if found_overlap:
                    total_count_correct_gold += 1

    scores["total_count_pred"] = total_count_pred
    scores["total_count_gold"] = total_count_gold
    if not span_overlap:
        scores["total_count_correct"] = total_count_correct
    else:
        scores["total_count_correct_pred"] = total_count_correct_pred
        scores["total_count_correct_gold"] = total_count_correct_gold

    total_count_pred = float(total_count_pred)
    total_count_gold = float(total_count_gold)
    if not span_overlap:
        total_count_correct = float(total_count_correct)
        precision = (
            total_count_correct / total_count_pred
            if total_count_pred != 0 else 0.0
        )
        recall = (
            total_count_correct / total_count_gold
            if total_count_gold != 0 else 0.0
        )
    else:
        total_count_correct_pred = float(total_count_correct_pred)
        total_count_correct_gold = float(total_count_correct_gold)
        precision = (
            total_count_correct_pred / total_count_pred
            if total_count_pred != 0 else 0.0
        )
        recall = (
            total_count_correct_gold / total_count_gold
            if total_count_gold != 0 else 0.0
        )
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * (precision * recall) / (precision + recall)
    scores["precision"] = precision * 100.0
    scores["recall"] = recall * 100.0
    scores["f1"] = f1 * 100.0

    return scores


def find_overlap(pred_span, gold_span):
    """
    Parameters
    ----------
    pred_span : tuple[int]
    gold_span : tuple[int]

    Returns
    -------
    bool
    """
    pred_begin_i, pred_end_i = pred_span
    gold_begin_i, gold_end_i = gold_span
    pred_set = set(range(pred_begin_i, pred_end_i + 1))
    gold_set = set(range(gold_begin_i, gold_end_i + 1))
    return len(pred_set & gold_set) > 0
