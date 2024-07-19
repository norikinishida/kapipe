import json
import logging
import os

from ... import utils


logger = logging.getLogger(__name__)


def to_official(path_input, path_output):
    """
        Modified from the code: https://github.com/wzhouad/ATLOP
    """
    preds = utils.read_json(path_input)
    triples = []
    for pred in preds:
        doc_key = pred["doc_key"]
        triples_local = pred["relations"]
        for y in triples_local:
            dct = {
                "title": doc_key,
                "h_idx": y["arg1"],
                "t_idx": y["arg2"],
                "r": y["relation"],
            }
            triples.append(dct)
    utils.write_json(path_output, triples)
    return triples


# def to_official_by_doc(path_input, path_output):
#     """
#         Modified from the code at https://github.com/tonytan48/Re-DocRED
#     """
#     preds = utils.read_json(path_input)
#     triples = []
#     for doc_key in preds.keys():
#         triples_local = preds[doc_key]["relations"]
#         new_triples_local = []
#         for arg1, rel, arg2 in triples_local:
#             dct = {
#                 "title": doc_key,
#                 "h_idx": arg1,
#                 "t_idx": arg2,
#                 "r": rel,
#             }
#             new_triples_local.append(dct)
#         triples.append(new_triples_local)
#     utils.write_json(path_output, triples)
#     return triples


def gen_train_facts(data_file_name, truth_dir):
    fact_file_name = data_file_name[data_file_name.find("train_"):]
    fact_file_name = os.path.join(
        truth_dir,
        fact_file_name.replace(".json", ".fact")
    )
    logger.info(f"fact_file_name: {fact_file_name}")

    if os.path.exists(fact_file_name):
        logger.info(f"Found {fact_file_name}")
        fact_in_train = set([])
        triples = json.load(open(fact_file_name))
        for x in triples:
            fact_in_train.add(tuple(x))
        return fact_in_train

    logger.info(f"Creating {fact_file_name} ...")
    fact_in_train = set([])
    ori_data = json.load(open(data_file_name))
    for data in ori_data:
        vertexSet = data['vertexSet']
        for label in data['labels']:
            rel = label['r']
            for n1 in vertexSet[label['h']]:
                for n2 in vertexSet[label['t']]:
                    fact_in_train.add((n1['name'], n2['name'], rel))
    json.dump(list(fact_in_train), open(fact_file_name, "w"))
    logger.info(f"Created {fact_file_name} ...")

    return fact_in_train


def official_evaluate(
    triples,
    original_data_dir,
    train_file_name,
    dev_file_name
):
    '''
        Adapted from the official evaluation code
    '''
    # e.g., /path/to/data/docred/original/ref
    truth_dir = os.path.join(original_data_dir, 'ref')

    if not os.path.exists(truth_dir):
        os.makedirs(truth_dir)

    # e.g., /path/to/data/docred/original/train_annotated.json
    #   -> /path/to/data/docred/original/ref/train_annotated.fact
    fact_in_train_annotated = gen_train_facts(
        os.path.join(original_data_dir, train_file_name),
        truth_dir
    )
    # e.g., /path/to/data/docred/original/train_distant.json
    #   -> /path/to/data/docred/original/ref/train_distant.fact
    fact_in_train_distant = gen_train_facts(
        os.path.join(original_data_dir, "train_distant.json"),
        truth_dir
    )

    truth = json.load(open(os.path.join(original_data_dir, dev_file_name)))

    std = {}
    tot_evidences = 0
    titleset = set([])

    title2vectexSet = {}

    for x in truth:
        title = x['title']
        titleset.add(title)

        vertexSet = x['vertexSet']
        title2vectexSet[title] = vertexSet

        for label in x['labels']:
            r = label['r']
            h_idx = label['h']
            t_idx = label['t']
            std[(title, r, h_idx, t_idx)] = set(label['evidence'])
            tot_evidences += len(label['evidence'])

    tot_relations = len(std)
    triples.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
    submission_answer = [triples[0]]
    for i in range(1, len(triples)):
        x = triples[i]
        y = triples[i - 1]
        if (
            (x['title'], x['h_idx'], x['t_idx'], x['r'])
            != (y['title'], y['h_idx'], y['t_idx'], y['r'])
        ):
            submission_answer.append(triples[i])

    correct_re = 0
    correct_evidence = 0
    pred_evi = 0

    correct_in_train_annotated = 0
    correct_in_train_distant = 0
    titleset2 = set([])
    for x in submission_answer:
        title = x['title']
        h_idx = x['h_idx']
        t_idx = x['t_idx']
        r = x['r']
        titleset2.add(title)
        if title not in title2vectexSet:
            continue
        vertexSet = title2vectexSet[title]

        if 'evidence' in x:
            evi = set(x['evidence'])
        else:
            evi = set([])
        pred_evi += len(evi)

        if (title, r, h_idx, t_idx) in std:
            correct_re += 1
            stdevi = std[(title, r, h_idx, t_idx)]
            correct_evidence += len(stdevi & evi)
            in_train_annotated = in_train_distant = False
            for n1 in vertexSet[h_idx]:
                for n2 in vertexSet[t_idx]:
                    if (n1['name'], n2['name'], r) in fact_in_train_annotated:
                        in_train_annotated = True
                    if (n1['name'], n2['name'], r) in fact_in_train_distant:
                        in_train_distant = True

            if in_train_annotated:
                correct_in_train_annotated += 1
            if in_train_distant:
                correct_in_train_distant += 1

    # Relation precision/recall/f1
    re_p = 1.0 * correct_re / len(submission_answer)
    re_r = 1.0 * correct_re / tot_relations
    if re_p + re_r == 0:
        re_f1 = 0
    else:
        re_f1 = 2.0 * re_p * re_r / (re_p + re_r)

    # Evidence precision/recall/f1
    evi_p = 1.0 * correct_evidence / pred_evi if pred_evi > 0 else 0
    evi_r = 1.0 * correct_evidence / tot_evidences
    if evi_p + evi_r == 0:
        evi_f1 = 0
    else:
        evi_f1 = 2.0 * evi_p * evi_r / (evi_p + evi_r)

    # Ign Relation precision/recall/f1
    re_p_ignore_train_annotated = (
        1.0 * (correct_re - correct_in_train_annotated)
        / (len(submission_answer) - correct_in_train_annotated + 1e-5)
    )
    re_p_ignore_train = (
        1.0 * (correct_re - correct_in_train_distant)
        / (len(submission_answer) - correct_in_train_distant + 1e-5)
    )
    if re_p_ignore_train_annotated + re_r == 0:
        re_f1_ignore_train_annotated = 0
    else:
        re_f1_ignore_train_annotated = (
            2.0 * re_p_ignore_train_annotated * re_r
            / (re_p_ignore_train_annotated + re_r)
        )
    if re_p_ignore_train + re_r == 0:
        re_f1_ignore_train = 0
    else:
        re_f1_ignore_train = (
            2.0 * re_p_ignore_train * re_r / (re_p_ignore_train + re_r)
        )

    # return re_f1, evi_f1, re_f1_ignore_train_annotated, re_f1_ignore_train
    scores = {}
    scores["standard"] = {
        "total_count_pred": len(submission_answer),
        "total_count_gold": tot_relations,
        "total_count_correct": correct_re,
        "precision": re_p * 100.0,
        "recall": re_r * 100.0,
        "f1": re_f1 * 100.0,
    }
    scores["ign"] = {
        "total_count_pred": len(submission_answer),
        "total_count_gold": tot_relations,
        "total_count_correct": correct_re,
        "total_count_correct_in_train": correct_in_train_annotated,
        "precision": re_p_ignore_train_annotated * 100.0,
        "recall": re_r * 100.0,
        "f1": re_f1_ignore_train_annotated * 100.0,
    }
    return scores


# def official_evaluate_benchmark(triples, original_data_dir, train_file_name, dev_file_name, vocab_rel_path):
#     '''
#         Adapted from the official evaluation code
#     '''
#     freq_keys = set(['P17', 'P131', 'P27', 'P150', 'P175', 'P577', 'P463', 'P527', 'P495', 'P361'])
#     rel2id = utils.read_vocab(vocab_rel_path)
#     long_tail_keys = set(rel2id.keys()) - freq_keys
#     truth_dir = os.path.join(original_data_dir, 'ref')

#     if not os.path.exists(truth_dir):
#         os.makedirs(truth_dir)

#     fact_in_train_annotated = gen_train_facts(os.path.join(original_data_dir, train_file_name), truth_dir)
#     fact_in_train_distant = gen_train_facts(os.path.join(original_data_dir, "train_distant.json"), truth_dir)

#     truth = json.load(open(os.path.join(original_data_dir, dev_file_name)))

#     std = {}
#     std_freq = {}
#     std_long_tail = {}
#     tot_evidences = 1
#     titleset = set([])

#     title2vectexSet = {}
#     std_intra = {}
#     std_inter = {}
#     std_inter_long = {}
#     for x in truth:
#         title = x['title']
#         titleset.add(title)

#         vertexSet = x['vertexSet']
#         title2vectexSet[title] = vertexSet

#         for label in x['labels']:
#             r = label['r']
#             h_idx = label['h']
#             t_idx = label['t']
#             h_sent_set = [x['sent_id'] for x in vertexSet[h_idx]]
#             t_sent_set = [x['sent_id'] for x in vertexSet[t_idx]]

#             std[(title, r, h_idx, t_idx)] = set(label['evidence'])
#             tot_evidences += len(label['evidence'])
#             if findSmallestDifference(h_sent_set, t_sent_set, len(h_sent_set),len(t_sent_set) )==0:
#                 std_intra[(title, r, h_idx, t_idx)] = set(label['evidence'])
#             if 1 <= findSmallestDifference(h_sent_set, t_sent_set, len(h_sent_set),len(t_sent_set) )  :
#                 std_inter[(title, r, h_idx, t_idx)] = set(label['evidence'])
#             if 5 < findSmallestDifference(h_sent_set, t_sent_set, len(h_sent_set),len(t_sent_set) )  :
#                 std_inter_long[(title, r, h_idx, t_idx)] = set(label['evidence'])
#             if r in freq_keys:
#                 std_freq[(title, r, h_idx, t_idx)] = set(label['evidence'])
#             if r in long_tail_keys:
#                 std_long_tail[(title, r, h_idx, t_idx)] = set(label['evidence'])

#     tot_relations = len(std)
#     tot_relations_freq = len(std_freq)
#     tot_relations_long_tail = len(std_long_tail)
#     tot_relations_intra = len(std_intra)
#     tot_relations_inter = len(std_inter)
#     tot_relations_inter_long = len(std_inter_long)

#     triples.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
#     if len(triples) > 1:
#         submission_answer = [triples[0]]
#         for i in range(1, len(triples)):
#             x = triples[i]
#             y = triples[i - 1]
#             if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
#                 submission_answer.append(triples[i])
#     else:
#         submission_answer = []
#     submission_answer_freq = []
#     submission_answer_long_tail =[]

#     submission_answer_freq = [x for x in submission_answer if x['r'] in freq_keys]
#     submission_answer_long_tail = [x for x in submission_answer if x['r'] in long_tail_keys]
#     submission_answer_intra = []
#     submission_answer_inter = []
#     submission_answer_inter_long = []
#     for i in range(len(submission_answer)):
#         vertexSet = title2vectexSet[submission_answer[i]['title']]
#         if title not in title2vectexSet:
#             print(title)
#             continue
#         h_sent_set = [x['sent_id'] for x in vertexSet[submission_answer[i]['h_idx']]]
#         t_sent_set = [x['sent_id'] for x in vertexSet[submission_answer[i]['t_idx']]]
#         if findSmallestDifference(h_sent_set, t_sent_set, len(h_sent_set),len(t_sent_set) )==0:
#             submission_answer_intra.append(submission_answer[i])
#         if 1<= findSmallestDifference(h_sent_set, t_sent_set, len(h_sent_set),len(t_sent_set))  :
#             submission_answer_inter.append(submission_answer[i])
#         if 5 < findSmallestDifference(h_sent_set, t_sent_set, len(h_sent_set),len(t_sent_set)) :
#             submission_answer_inter_long.append(submission_answer[i])

#     correct_re = 0
#     correct_re_freq = 0
#     correct_re_long_tail = 0
#     correct_re_intra = 0
#     correct_re_inter = 0
#     correct_re_inter_long = 0
#     correct_evidence = 0
#     pred_evi = 0

#     correct_in_train_annotated = 0
#     correct_in_train_distant = 0
#     titleset2 = set([])
#     for x in submission_answer:
#         title = x['title']
#         h_idx = x['h_idx']
#         t_idx = x['t_idx']
#         r = x['r']
#         titleset2.add(title)
#         if title not in title2vectexSet:
#             continue
#         vertexSet = title2vectexSet[title]

#         if 'evidence' in x:
#             evi = set(x['evidence'])
#         else:
#             evi = set([])
#         pred_evi += len(evi)

#         if (title, r, h_idx, t_idx) in std:
#             correct_re += 1
#             stdevi = std[(title, r, h_idx, t_idx)]
#             correct_evidence += len(stdevi & evi)
#             in_train_annotated = in_train_distant = False
#             for n1 in vertexSet[h_idx]:
#                 for n2 in vertexSet[t_idx]:
#                     if (n1['name'], n2['name'], r) in fact_in_train_annotated:
#                         in_train_annotated = True
#                     if (n1['name'], n2['name'], r) in fact_in_train_distant:
#                         in_train_distant = True

#             if in_train_annotated:
#                 correct_in_train_annotated += 1
#             if in_train_distant:
#                 correct_in_train_distant += 1
#     for x in submission_answer_freq:
#         title = x['title']
#         h_idx = x['h_idx']
#         t_idx = x['t_idx']
#         r = x['r']
#         titleset2.add(title)
#         if title not in title2vectexSet:
#             continue
#         vertexSet = title2vectexSet[title]

#         if (title, r, h_idx, t_idx) in std_freq:
#             correct_re_freq += 1
#     for x in submission_answer_long_tail:
#         title = x['title']
#         h_idx = x['h_idx']
#         t_idx = x['t_idx']
#         r = x['r']
#         titleset2.add(title)
#         if title not in title2vectexSet:
#             continue
#         vertexSet = title2vectexSet[title]

#         if (title, r, h_idx, t_idx) in std_long_tail:
#             correct_re_long_tail += 1

#     for x in submission_answer_intra:
#         title = x['title']
#         h_idx = x['h_idx']
#         t_idx = x['t_idx']
#         r = x['r']
#         titleset2.add(title)
#         if title not in title2vectexSet:
#             continue
#         vertexSet = title2vectexSet[title]

#         if (title, r, h_idx, t_idx) in std_intra:
#             correct_re_intra += 1
#     for x in submission_answer_inter:
#         title = x['title']
#         h_idx = x['h_idx']
#         t_idx = x['t_idx']
#         r = x['r']
#         titleset2.add(title)
#         if title not in title2vectexSet:
#             continue
#         vertexSet = title2vectexSet[title]

#         if (title, r, h_idx, t_idx) in std_inter:
#             correct_re_inter += 1

#     for x in submission_answer_inter_long:
#         title = x['title']
#         h_idx = x['h_idx']
#         t_idx = x['t_idx']
#         r = x['r']
#         titleset2.add(title)
#         if title not in title2vectexSet:
#             continue
#         vertexSet = title2vectexSet[title]

#         if (title, r, h_idx, t_idx) in std_inter_long:
#             correct_re_inter_long += 1

#     # Precision/Recall/F1 for relations
#     if len(submission_answer) > 0:
#         re_p = 1.0 * correct_re / len(submission_answer)
#     else:
#         re_p = 0
#     re_r = 1.0 * correct_re / tot_relations
#     if re_p + re_r == 0:
#         re_f1 = 0
#     else:
#         re_f1 = 2.0 * re_p * re_r / (re_p + re_r)

#     # Precision/Recall/F1 for frequent relations
#     if len(submission_answer_freq)>0:
#         re_p_freq = 1.0 * correct_re_freq / len(submission_answer_freq)
#     else:
#         re_p_freq = 0
#     re_r_freq = 1.0 * correct_re_freq / tot_relations_freq
#     if re_p_freq + re_r_freq == 0:
#         re_f1_freq = 0
#     else:
#         re_f1_freq = 2.0 * re_p_freq * re_r_freq / (re_p_freq + re_r_freq)

#     # Precision/Recall/F1 for long-tail relations
#     if len(submission_answer_long_tail) > 0:
#         re_p_long_tail = 1.0 * correct_re_long_tail / len(submission_answer_long_tail)
#     else:
#         re_p_long_tail = 0
#     re_r_long_tail = 1.0 * correct_re_long_tail / tot_relations_long_tail
#     if re_p_long_tail + re_r_long_tail == 0:
#         re_f1_long_tail = 0
#     else:
#         re_f1_long_tail = 2.0 * re_p_long_tail * re_r_long_tail / (re_p_long_tail + re_r_long_tail)

#     # Precision/Recall/F1 for intra-sentence relations
#     if len(submission_answer_intra)>0:
#         re_p_intra = 1.0 * correct_re_intra / len(submission_answer_intra)
#     else:
#         re_p_intra = 0
#     re_r_intra = 1.0 * correct_re_intra / tot_relations_intra
#     if re_p_intra + re_r_intra == 0:
#         re_f1_intra = 0
#     else:
#         re_f1_intra = 2.0 * re_p_intra * re_r_intra / (re_p_intra + re_r_intra)

#     # Precision/Recall/F1 for inter-sentence relations
#     if len(submission_answer_inter)>0:
#         re_p_inter = 1.0 * correct_re_inter / len(submission_answer_inter)
#     else:
#         re_p_inter = 0
#     re_r_inter = 1.0 * correct_re_inter / tot_relations_inter
#     if re_p_inter + re_r_inter == 0:
#         re_f1_inter = 0
#     else:
#         re_f1_inter = 2.0 * re_p_inter * re_r_inter / (re_p_inter + re_r_inter)

#     # Precision/Recall/F1 for evidences
#     evi_p = 1.0 * correct_evidence / pred_evi if pred_evi > 0 else 0
#     evi_r = 1.0 * correct_evidence / tot_evidences
#     if evi_p + evi_r == 0:
#         evi_f1 = 0
#     else:
#         evi_f1 = 2.0 * evi_p * evi_r / (evi_p + evi_r)

#     # Ign Precision/Recall/F1 for relations
#     re_p_ignore_train_annotated = 1.0 * (correct_re - correct_in_train_annotated) / (len(submission_answer) - correct_in_train_annotated + 1e-5)
#     re_p_ignore_train = 1.0 * (correct_re - correct_in_train_distant) / (len(submission_answer) - correct_in_train_distant + 1e-5)
#     if re_p_ignore_train_annotated + re_r == 0:
#         re_f1_ignore_train_annotated = 0
#     else:
#         re_f1_ignore_train_annotated = 2.0 * re_p_ignore_train_annotated * re_r / (re_p_ignore_train_annotated + re_r)
#     if re_p_ignore_train + re_r == 0:
#         re_f1_ignore_train = 0
#     else:
#         re_f1_ignore_train = 2.0 * re_p_ignore_train * re_r / (re_p_ignore_train + re_r)

#     # return re_f1, evi_f1, re_f1_ignore_train_annotated, re_f1_ignore_train, re_p, re_r, re_f1_freq, re_f1_long_tail, re_f1_intra, re_f1_inter, re_p_freq, re_r_freq, re_p_long_tail, re_r_long_tail
#     scores = {}
#     scores["relations"] = {
#         "total_count_pred": len(submission_answer),
#         "total_count_gold": tot_relations,
#         "total_count_correct": correct_re,
#         "precision": re_p * 100.0,
#         "recall": re_r * 100.0,
#         "f1": re_f1 * 100.0,
#     }
#     scores["relations_intra"] = {
#         "total_count_pred": len(submission_answer_intra),
#         "total_count_gold": tot_relations_intra,
#         "total_count_correct": correct_re_intra,
#         "precision": re_p_intra * 100.0,
#         "recall": re_r_intra * 100.0,
#         "f1": re_f1_intra * 100.0,
#     }
#     scores["relations_inter"] = {
#         "total_count_pred": len(submission_answer_inter),
#         "total_count_gold": tot_relations_inter,
#         "total_count_correct": correct_re_inter,
#         "precision": re_p_inter * 100.0,
#         "recall": re_r_inter * 100.0,
#         "f1": re_f1_inter * 100.0,
#     }
#     scores["relations_freq"] = {
#         "total_count_pred": len(submission_answer_freq),
#         "total_count_gold": tot_relations_freq,
#         "total_count_correct": correct_re_freq,
#         "precision": re_p_freq * 100.0,
#         "recall": re_r_freq * 100.0,
#         "f1": re_f1_freq * 100.0,
#     }
#     scores["relations_long_tail"] = {
#         "total_count_pred": len(submission_answer_long_tail),
#         "total_count_gold": tot_relations_long_tail,
#         "total_count_correct": correct_re_long_tail,
#         "precision": re_p_long_tail * 100.0,
#         "recall": re_r_long_tail * 100.0,
#         "f1": re_f1_long_tail * 100.0,
#     }
#     scores["relations_ign"] = {
#         "total_count_pred": len(submission_answer),
#         "total_count_gold": tot_relations,
#         "total_count_correct": correct_re,
#         "total_count_correct_in_train": correct_in_train_annotated,
#         "precision": re_p_ignore_train_annotated * 100.0,
#         "recall": re_r * 100.0,
#         "f1": re_f1_ignore_train_annotated * 100.0,
#     }
#     return scores


# def findSmallestDifference(A, B, m, n):

#     # Sort both arrays
#     # using sort function
#     A.sort()
#     B.sort()

#     a = 0
#     b = 0

#     # Initialize result as max value
#     result = sys.maxsize

#     # Scan Both Arrays upto
#     # sizeof of the Arrays
#     while (a < m and b < n):

#         if (abs(A[a] - B[b]) < result):
#             result = abs(A[a] - B[b])

#         # Move Smaller Value
#         if (A[a] < B[b]):
#             a += 1

#         else:
#             b += 1
#     # return final sma result
#     return result





