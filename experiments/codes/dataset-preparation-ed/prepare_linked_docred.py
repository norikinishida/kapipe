import argparse
import json
import os

from tqdm import tqdm

import sys
sys.path.insert(0, "../../..")
from kapipe import utils


def main(args):
    path_input_file = args.input_file
    path_output_file = args.output_file
    utils.mkdir(os.path.dirname(path_output_file))

    doc_key_list = []
    n_sentences = 0
    n_mentions = 0
    n_entities = 0

    records = []
    with open(path_input_file, "r") as f:
        dataset = json.load(f)
        for data in tqdm(dataset):
            doc_key = data["title"]

            assert not doc_key in doc_key_list, doc_key
            doc_key_list.append(doc_key)

            data, local_shift_map = process_spaces(data=data)
            # print(utils.pretty_format_dict(data))
            # sys.exit()
            local_to_global_map = get_local_to_global_map(data=data)

            record = {}
            record["doc_key"] = doc_key
            record["sentences"] = [" ".join(s) for s in data["sents_without_spaces"]]
            record["mentions"] = get_mentions(
                data=data,
                local_shift_map=local_shift_map,
                local_to_global_map=local_to_global_map
            )
            record["entities"] = get_entities(
                data=data,
                local_shift_map=local_shift_map,
                local_to_global_map=local_to_global_map,
                mentions=record["mentions"]
            )

            n_sentences += len(record["sentences"])
            n_mentions += len(record["mentions"])
            n_entities += len(record["entities"])
                 
            records.append(record)
    utils.write_json(path_output_file, records)

    print(f"Processed {len(doc_key_list)} documents.")
    print(f"Average number of sentences per doc: {n_sentences}/{len(doc_key_list)} = {float(n_sentences) / len(doc_key_list)}")
    print(f"Average number of mentions per doc: {n_mentions}/{len(doc_key_list)} = {float(n_mentions) / len(doc_key_list)}")
    print(f"Average number of entities per doc: {n_entities}/{len(doc_key_list)} = {float(n_entities) / len(doc_key_list)}")


def process_spaces(data):
    sents_without_spaces = []
    local_shift_map = {}
    for sent_i, sent in enumerate(data["sents"]):
        sent_without_spaces = []
        for token_i, token in enumerate(sent):
            lst = token.split()
            assert len(lst) < 2
            if len(lst) == 1:
                sent_without_spaces.append(token)
            if len(sent_without_spaces) == 0:
                # To avoid assigning -1
                local_shift_map[(sent_i, token_i)] = 0
            else:
                local_shift_map[(sent_i, token_i)] = len(sent_without_spaces) - 1
        sents_without_spaces.append(sent_without_spaces)

    for sent in sents_without_spaces:
        assert len(sent) == len(" ".join(sent).split())

    data["sents_without_spaces"] = sents_without_spaces
    return data, local_shift_map


def get_local_to_global_map(data):
    local_to_global_map = {} # (sent_index, local token index) -> global token index
    global_token_i = 0
    for s_i, sent in enumerate(data["sents_without_spaces"]):
        for t_i, token in enumerate(sent):
            local_to_global_map[(s_i, t_i)] = global_token_i
            global_token_i += 1
    return local_to_global_map


def get_mentions(data, local_shift_map, local_to_global_map):
    mentions = []
    n_ignored = 0
    for entity_i, entity in enumerate(data["entities"]):
        # Entity type
        entity_type = entity["type"]
        # Entity ID
        wikipedia_id = entity["entity_linking"]["wikipedia_resource"]
        wikidata_id = entity["entity_linking"]["wikidata_resource"]
        entity_id = wikipedia_id
        if entity_id == "#ignored#":
            entity_id = f"{entity_id}-{n_ignored}"
            n_ignored += 1
        wikipedia_not_resource = entity["entity_linking"]["wikipedia_not_resource"]
        method = entity["entity_linking"]["method"]
        confidence = entity["entity_linking"]["confidence"]
        for mention in entity["mentions"]:
            # Sentence index
            sent_index = mention["sent_id"]
            # Span
            # original, local
            mention_begin_token_index, mention_end_token_index = mention["pos"]
            mention_end_token_index -= 1
            # after space removal, local
            mention_begin_token_index = local_shift_map[(sent_index, mention_begin_token_index)]
            mention_end_token_index = local_shift_map[(sent_index, mention_end_token_index)]
            # global
            mention_begin_token_index = local_to_global_map[(sent_index, mention_begin_token_index)]
            mention_end_token_index = local_to_global_map[(sent_index, mention_end_token_index)]
            # Name
            mention_name = mention["name"]
            dct = {
                "span": (mention_begin_token_index, mention_end_token_index),
                "name": mention_name,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "entity_linking_detail": {
                    "wikidata_id": wikidata_id,
                    "wikipedia_not_resource": wikipedia_not_resource,
                    "method": method,
                    "confidence": confidence,
                }
            }
            mentions.append(dct)
    mentions = remove_duplicated_mentions(mentions=mentions)
    mentions = sorted(mentions, key=lambda x: x["span"])
    return mentions


def remove_duplicated_mentions(mentions):
    new_mentions = []
    keys = []
    for mention in mentions:
        if mention["span"] in keys:
            continue
        new_mentions.append(mention)
        keys.append(mention["span"])
    return new_mentions


def get_entities(data, local_shift_map, local_to_global_map, mentions):
    # Make a map from mention span (b, e) to mention index
    span2index = create_span_to_mention_index_dict(mentions=mentions)
    # Aggregate entities
    entities = {}
    n_ignored = 0
    for entity_i, entity in enumerate(data["entities"]):
        # Entity ID
        wikipedia_id = entity["entity_linking"]["wikipedia_resource"]
        wikidata_id = entity["entity_linking"]["wikidata_resource"]
        entity_id = wikipedia_id
        if entity_id == "#ignored#":
            entity_id = f"{entity_id}-{n_ignored}"
            n_ignored += 1
        # Entity Type
        entity_type = entity["type"]
        # Mention indices
        spans = []
        for mention in entity["mentions"]:
            sent_index = mention["sent_id"]
            # original, local
            mention_begin_token_index, mention_end_token_index = mention["pos"]
            mention_end_token_index -= 1
            # after space removal, local
            mention_begin_token_index = local_shift_map[(sent_index, mention_begin_token_index)]
            mention_end_token_index = local_shift_map[(sent_index, mention_end_token_index)]
            # global
            mention_begin_token_index = local_to_global_map[(sent_index, mention_begin_token_index)]
            mention_end_token_index = local_to_global_map[(sent_index, mention_end_token_index)]
            spans.append((mention_begin_token_index, mention_end_token_index))
        mention_indices = [span2index[(b,e)] for b, e in spans]
        mention_indices = sorted(mention_indices)
        # Mention names
        mention_names = [mentions[m_i]["name"] for m_i in mention_indices]
        if not entity_id in entities:
            entities[entity_id] = {
                "entity_type": entity_type,
                "mention_indices": mention_indices,
                "mention_names": mention_names,
            }
        else:
            assert entities[entity_id]["entity_type"] == entity_type, (entities, "@@@@@@", entity)
            assert tuple(entities[entity_id]["mention_indices"]) == tuple(mention_indices)
            assert tuple(entities[entity_id]["mention_names"]) == tuple(mention_names)

    entities = transform_entities(dct=entities)
    # entities = sorted(entities, key=lambda x: x["mention_indices"][0])
    return entities


def create_span_to_mention_index_dict(mentions):
    dct = {}
    for m_i, mention in enumerate(mentions):
        dct[tuple(mention["span"])] = m_i
    assert len(dct) == len(mentions)
    return dct


def transform_entities(dct):
    entities = []
    for entity_id in dct:
        entity_type = dct[entity_id]["entity_type"]
        mention_indices = dct[entity_id]["mention_indices"]
        mention_names = dct[entity_id]["mention_names"]
        entity = {
            "mention_indices": mention_indices,
            "mention_names": mention_names,
            "entity_type": entity_type,
            "entity_id": entity_id,
        }
        entities.append(entity)
    return entities


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    main(args=args)
 