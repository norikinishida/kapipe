import argparse
from collections import defaultdict
import os

import pandas as pd
from tqdm import tqdm

import sys
sys.path.insert(0, "../../..")
from kapipe import utils


def main(args):
    path_input_dir = args.input_dir
    path_triples = args.triples
    path_output_dir = args.output_dir
    # min_threshold = args.min_threshold
    # top_k = args.top_k

    utils.mkdir(path_output_dir)

    print(f"Loading triples from {path_triples}")
    triples = utils.read_json(path_triples)
    print(f"Loaded {len(triples)} triples")

    # Convert triples {(h, r, t)} to a pair-to-relations mapping {(h,t)->[r]}
    print("Converting the triples into the form of {(head, tail) -> [relation]}")
    pair2rels = defaultdict(list)
    for triple in triples:
        head = triple["head"]
        tail = triple["tail"]
        rel = triple["relation"]
        pair2rels[(head, tail)].append(rel)
    for pair, rels in pair2rels.items():
        pair2rels[pair] = list(set(rels))
    print("Converted")

    # relation_set = None
    for split in ["train", "dev", "test"]:
        path_input_file = os.path.join(path_input_dir, f"{split}.json")
        # path_output_file = os.path.join(path_output_dir, f"{split}.min{min_threshold}_top{top_k}.json")
        path_output_file = os.path.join(path_output_dir, f"{split}.json")

        print(f"Loading documents from {path_input_file}")
        documents = utils.read_json(path_input_file)
        print(f"Loaded {len(documents)} documents")

        print("Annotating distantly supervised triples")
        documents, n_added_triples = annotate(
            documents=documents,
            pair2rels=pair2rels
        )
        print(f"Annotated {n_added_triples} distantly supervised triples ({float(n_added_triples) / len(documents)} per doc)")

        # if split == "train":
        #     relation_set = get_target_relation_set(
        #         documents=documents,
        #         min_threshold=min_threshold,
        #         top_k=top_k
        #     )

        # print(f"Removing triples with rare relations (min_threshold: {min_threshold}; top-k: {top_k})")
        # documetns, n_removed_triples = remove_triples_with_rare_relations(
        #     documents=documents,
        #     relation_set=relation_set
        # )
        # print(f"Removed {n_removed_triples} rare-relation triples")

        utils.write_json(path_output_file, documents)
        print(f"Saved documents with the distantly supervised triples to {path_output_file}")

    print("Completed") 


def annotate(documents, pair2rels):
    n_added_triples = 0
    for doc_i, doc in tqdm(enumerate(documents), total=len(documents)):
        entities = doc["entities"]
        entity_index_to_sentence_indices = get_entity_index_to_sentence_indices(document=doc)
        triples_for_doc = [] # list[dict]
        for head_i, head in enumerate(entities):
            for tail_i, tail in enumerate(entities):
                if head_i == tail_i:
                    continue
                head_id = head["entity_id"]
                tail_id = tail["entity_id"]
                if len(entity_index_to_sentence_indices[head_i] & entity_index_to_sentence_indices[tail_i]) != 0:
                    if (head_id, tail_id) in pair2rels:
                        rels = pair2rels[(head_id, tail_id)]
                        for rel in rels:
                            triples_for_doc.append({
                                "arg1": head_i,
                                "relation": rel,
                                "arg2": tail_i
                            })
        documents[doc_i]["relations"] = triples_for_doc
        n_added_triples += len(triples_for_doc)
    return documents, n_added_triples

    
def get_entity_index_to_sentence_indices(document):
    token_index_to_sentence_index = []
    for s_i, sent in enumerate(document["sentences"]):
        n = len(sent.split())
        token_index_to_sentence_index.extend([s_i] * n)

    entity_index_to_sentence_indices = []
    mentions = document["mentions"]
    for e_i, ent in enumerate(document["entities"]):
        sentence_indices = set()
        for m_i in ent["mention_indices"]:
            begin_token_i, end_token_i = mentions[m_i]["span"]
            sentence_indices.add(token_index_to_sentence_index[begin_token_i])
            sentence_indices.add(token_index_to_sentence_index[end_token_i])
        entity_index_to_sentence_indices.append(sentence_indices)

    return entity_index_to_sentence_indices


# def get_target_relation_set(documents, min_threshold, top_k):
#     counter = defaultdict(int)
#     for doc in documents:
#         for triple in doc["relations"]:
#             counter[triple["relation"]] += 1
#     relation_set = [(k, v) for k, v in counter.items() if v >= min_threshold]
#     relation_set = sorted(relation_set, key=lambda x: -x[1])[:top_k]
#     relation_set = [k for k, v in relation_set]
#     return set(relation_set)


# def remove_triples_with_rare_relations(documents, relation_set
#     n_removed_triples = 0
#     for doc_i, doc in enumerate(documents):
#         n_prev = len(doc["relations"])
#         doc["relations"] = [
#             triple for triple in doc["relations"]
#             if triple["relation"] in relation_set
#         ]
#         documents[doc_i] = doc
#         n_new = len(doc["relations"])
#         n_removed_triples += (n_prev - n_new)
#     return documents, n_removed_triples
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--triples", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    # parser.add_argument("--min_threshold", type=int, default=0)
    # parser.add_argument("--top_k", type=int, default=-1)
    args = parser.parse_args()
    main(args)
