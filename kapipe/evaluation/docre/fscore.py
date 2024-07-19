from ... import utils


def fscore(
    pred_path,
    gold_path,
    skip_intra_inter=False,
    skip_ign=False,
    gold_train_triples_path=None
):
    """
    Parameters
    ----------
    pred_path : str | list[Document]
    gold_path : str | list[Document]
    skip_intra_inter : bool, optional
        by default False
    skip_ign : bool, optional
        by default False
    gold_train_triples_path : str | None, optional
        by default None

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
    assert len(pred_documents) == len(gold_documents), (len(pred_documents), len(gold_documents))
    for pred_doc, gold_doc in zip(pred_documents, gold_documents):
        assert pred_doc["doc_key"] == gold_doc["doc_key"]

    # Evaluate
    scores["standard"] = _fscore(
        pred_documents=pred_documents,
        gold_documents=gold_documents,
        all_or_intra_or_inter="all"
    )
    if not skip_intra_inter:
        scores["intra"] = _fscore(
            pred_documents=pred_documents,
            gold_documents=gold_documents,
            all_or_intra_or_inter="intra"
        )
        scores["inter"] = _fscore(
            pred_documents=pred_documents,
            gold_documents=gold_documents,
            all_or_intra_or_inter="inter"
        )
    if not skip_ign:
        assert gold_train_triples_path is not None
        gold_train_triples = utils.read_json(gold_train_triples_path)
        gold_train_triples = set([
            (h,r,t) for (h,r,t) in gold_train_triples["root"]
        ])
        scores["ign"] = _fscore_ign(
            pred_documents=pred_documents,
            gold_documents=gold_documents,
            gold_train_triples=gold_train_triples
        )

    return scores


def _fscore(
    pred_documents,
    gold_documents,
    all_or_intra_or_inter="all"
):
    """
    Parameters
    ----------
    pred_documents : list[Document]
    gold_documents : list[Document]
    all_or_intra_or_inter : str, optional
        by default "all"

    Returns
    -------
    dict[str, Any]
    """
    scores = {}

    total_count_pred = 0
    total_count_gold = 0
    total_count_correct = 0
    for pred_doc, gold_doc in zip(pred_documents, gold_documents):
        pred_triples = pred_doc["relations"]
        gold_triples = gold_doc["relations"]

        pred_triples = [
            (y["arg1"], y["relation"], y["arg2"]) for y in pred_triples
        ]
        gold_triples = [
            (y["arg1"], y["relation"], y["arg2"]) for y in gold_triples
        ]

        if all_or_intra_or_inter in ["intra", "inter"]:
            # Filter triples
            intra_inter_map = gold_doc["intra_inter_map"]
            pred_triples = [
                (arg1, rel, arg2) for arg1, rel, arg2 in pred_triples
                if intra_inter_map[f"{arg1}-{arg2}"] == all_or_intra_or_inter
            ]
            gold_triples = [
                (arg1, rel, arg2) for arg1, rel, arg2 in gold_triples
                if intra_inter_map[f"{arg1}-{arg2}"] == all_or_intra_or_inter
            ]
        else:
            # No filtering
            assert all_or_intra_or_inter == "all"

        pred_entities = pred_doc["entities"]
        gold_entities = gold_doc["entities"]
        pred_triples = [
            (pred_entities[h]["entity_id"], r, pred_entities[t]["entity_id"])
            for h, r, t in pred_triples
        ]
        gold_triples = [
            (gold_entities[h]["entity_id"], r, gold_entities[t]["entity_id"])
            for h, r, t in gold_triples
        ]

        pred_triples = set(pred_triples)
        gold_triples = set(gold_triples)
        total_count_pred += len(pred_triples)
        total_count_gold += len(gold_triples)
        total_count_correct += len(pred_triples & gold_triples)
    scores["total_count_pred"] = total_count_pred
    scores["total_count_gold"] = total_count_gold
    scores["total_count_correct"] = total_count_correct

    total_count_pred = float(total_count_pred)
    total_count_gold = float(total_count_gold)
    total_count_correct = float(total_count_correct)
    precision = safe_div(total_count_correct, total_count_pred)
    recall = safe_div(total_count_correct, total_count_gold)
    f1 = safe_div(2.0 * (precision * recall), precision + recall)
    scores["precision"] = precision * 100.0
    scores["recall"] = recall * 100.0
    scores["f1"] = f1 * 100.0

    return scores


def _fscore_ign(
    pred_documents,
    gold_documents,
    gold_train_triples
):
    """
    Parameters
    ----------
    pred_documents : list[Document]
    gold_documents : list[Document]
    gold_train_triples : set[tuple[str, str, str]]

    Returns
    -------
    dict[str, Any]
    """
    scores = {}

    total_count_pred = 0
    total_count_gold = 0
    total_count_correct = 0
    total_count_correct_in_train = 0
    for pred_doc, gold_doc in zip(pred_documents, gold_documents):
        pred_triples = pred_doc["relations"]
        gold_triples = gold_doc["relations"]

        pred_triples = [
            (y["arg1"], y["relation"], y["arg2"]) for y in pred_triples
        ]
        gold_triples = [
            (y["arg1"], y["relation"], y["arg2"]) for y in gold_triples
        ]

        pred_entities = pred_doc["entities"]
        gold_entities = gold_doc["entities"]
        pred_triples = [
            (pred_entities[h]["entity_id"], r, pred_entities[t]["entity_id"])
            for h, r, t in pred_triples
        ]
        gold_triples = [
            (gold_entities[h]["entity_id"], r, gold_entities[t]["entity_id"])
            for h, r, t in gold_triples
        ]

        pred_triples = set(pred_triples)
        gold_triples = set(gold_triples)
        total_count_pred += len(pred_triples)
        total_count_gold += len(gold_triples)
        total_count_correct += len(pred_triples & gold_triples)

        correct_relations = pred_triples & gold_triples
        mentions = gold_doc["mentions"]
        entities = gold_doc["entities"]
        entity_id_to_mention_names = {
            e["entity_id"]: [
                mentions[m_i]["name"] for m_i in e["mention_indices"]
            ]
            for e in entities
        }
        for arg1, rel, arg2 in correct_relations:
            in_train = False
            arg1_mention_names = entity_id_to_mention_names[arg1]
            arg2_mention_names = entity_id_to_mention_names[arg2]
            for arg1_mention_name in arg1_mention_names:
                for arg2_mention_name in arg2_mention_names:
                    if (arg1_mention_name, rel, arg2_mention_name) in (
                        gold_train_triples
                    ):
                        in_train = True
                        break
                if in_train:
                    break
            if in_train:
                total_count_correct_in_train += 1

    scores["total_count_pred"] = total_count_pred
    scores["total_count_gold"] = total_count_gold
    scores["total_count_correct"] = total_count_correct
    scores["total_count_correct_in_train"] = total_count_correct_in_train

    total_count_pred = float(total_count_pred)
    total_count_gold = float(total_count_gold)
    total_count_correct = float(total_count_correct)
    total_count_correct_in_train = float(total_count_correct_in_train)
    # c.f., https://github.com/wzhouad/ATLOP/evaluation.py
    precision = safe_div(
        total_count_correct - total_count_correct_in_train,
        total_count_pred - total_count_correct_in_train
    )
    recall = safe_div(total_count_correct, total_count_gold)
    f1 = safe_div(2.0 * (precision * recall), precision + recall)
    scores["precision"] = precision * 100.0
    scores["recall"] = recall * 100.0
    scores["f1"] = f1 * 100.0

    return scores


def safe_div(x, y):
    if y == 0:
        return 0.0
    else:
        return x / y
