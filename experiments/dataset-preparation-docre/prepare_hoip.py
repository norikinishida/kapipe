import argparse
from collections import defaultdict
import os

from tqdm import tqdm

import sys
sys.path.insert(0, "../../..")
from kapipe import utils



def main(args):
    path_input_file = args.input_file
    path_output_file = args.output_file
    utils.mkdir(os.path.dirname(path_output_file))

    documents = utils.read_json(path_input_file)
    documents = process(documents)
    utils.write_json(path_output_file, documents)


def process(documents):
    """Transform documents to my own common format.

    Parameters
    ----------
    documents : list[Doc]

    Returns
    -------
    list[Doc]
    """
    # Transform to the common format
    results = []

    for document in tqdm(documents):
        result = {}

        result["doc_key"] = document["doc_key"]
        result["pubmed_id"] = document["pubmed_id"]
        result["sentences"] = document["sentences"]

        # First, collect entities
        entity_ids = []
        entity_id_to_names = defaultdict(list)
        entity_id_to_types = defaultdict(list)
        for triple in document["triples"]:
            # head
            entity_id = triple["head_entity"]["id"]
            entity_name  = triple["head_entity"]["name"]
            entity_type = triple["head_entity"]["type"]
            entity_ids.append(entity_id)
            entity_id_to_names[entity_id].append(entity_name)
            entity_id_to_types[entity_id].append(entity_type)
            # tail
            entity_id = triple["tail_entity"]["id"]
            entity_name  = triple["tail_entity"]["name"]
            entity_type = triple["tail_entity"]["type"]
            entity_ids.append(entity_id)
            entity_id_to_names[entity_id].append(entity_name)
            entity_id_to_types[entity_id].append(entity_type)
        entity_ids = sorted(list(set(entity_ids)))
        for e_id in entity_ids:
            names = entity_id_to_names[e_id]
            names = sorted(list(set(names)))
            entity_id_to_names[e_id] = names
            assert len(set(entity_id_to_names[e_id])) == 1
            assert len(set(entity_id_to_types[e_id])) == 1

        # Set mentions
        mentions = []
        # entity_id_to_mention_indices = defaultdict(list)
        # m_i = 0
        # for e_id in entity_ids:
        #     for e_name in entity_id_to_names[e_id]:
        #         mention = {"span": None,
        #                    "name": e_name,
        #                    "entity_type": entity_id_to_types[e_id][0],
        #                    "entity_id": e_id}
        #         mentions.append(mention)
        #         entity_id_to_mention_indices[e_id].append(m_i)
        #         m_i += 1
        entity_id_to_mention_index = {}
        m_i = 0
        for e_id in entity_ids:
            mention = {
                "span": None,
                "name": entity_id_to_names[e_id][0],
                "entity_type": entity_id_to_types[e_id][0],
                "entity_id": e_id
            }
            mentions.append(mention)
            entity_id_to_mention_index[e_id] = m_i
            m_i += 1
        result["mentions"] = mentions

        # Set entities
        entities = []
        entity_id_to_index = {}
        for e_i, e_id in enumerate(entity_ids):
            entity = {
                # "mention_indices": entity_id_to_mention_indices[e_id],
                "mention_indices": [entity_id_to_mention_index[e_id]],
                "entity_type": entity_id_to_types[e_id][0],
                "entity_id": e_id
            }
            entities.append(entity)
            entity_id_to_index[e_id] = e_i
        result["entities"] = entities

        # Set relations
        relations = []
        for triple in document["triples"]:
            head_entity_id = triple["head_entity"]["id"]
            head_entity_idx = entity_id_to_index[head_entity_id]
            tail_entity_id = triple["tail_entity"]["id"]
            tail_entity_idx = entity_id_to_index[tail_entity_id]
            relation = triple["relation"]["name"]
            relations.append((head_entity_idx, relation, tail_entity_idx))
        # NOTE: We need to filter out redundant triples when using coarse-grained triples
        relations = sorted(list(set(relations)), key=lambda x: (x[0],x[2],x[1]))
        relations = [
            {
                "arg1": x[0],
                "relation": x[1],
                "arg2": x[2]
            }
            for x in relations
        ]
        result["relations"] = relations
        result["not_include_pairs"] = []

        results.append(result)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    main(args=args)
