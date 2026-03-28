import argparse
import os

import sys
sys.path.insert(0, "../../..")
from kapipe import utils


def main(args):
    assert args.input_file.endswith("_filter.data")

    path_input_file = args.input_file
    path_output_file = args.output_file
    utils.mkdir(os.path.dirname(path_output_file))

    doc_key_list = []
    n_sentences = 0
    n_mentions = 0
    # n_chemical_mentions = 0
    # n_disease_mentions = 0
    # n_chemical_entities = 0
    # n_disease_entities = 0
    # n_positive_pairs = 0
    # n_negative_pairs = 0

    # records = {}
    records = []
    with open(path_input_file, "r") as f:
        for line in f:
            line = line.strip().split("\t")
            doc_key = line[0]
            text = line[1]
            sentences = [s.split() for s in text.split("|")]
            chunks = get_chunks(line[2:])

            assert not doc_key in doc_key_list, doc_key
            doc_key_list.append(doc_key)

            record = {}
            record["doc_key"] = doc_key
            record["sentences"] = [" ".join(s) for s in sentences]
            record["mentions"] = get_mentions(chunks=chunks)
            # record["entities"] = get_entities(chunks=chunks, mentions=record["mentions"])
            # entity_id_to_index = {e["entity_id"]: e_i for e_i, e in enumerate(record["entities"])}
            # record["relations"] = get_relations(chunks=chunks, entity_id_to_index=entity_id_to_index)
            # negative_pairs = get_negative_pairs(chunks=chunks, entity_id_to_index=entity_id_to_index)
            # record["not_include_pairs"] = get_not_include_pairs(chunks=chunks, entity_id_to_index=entity_id_to_index)

            n_sentences += len(record["sentences"])
            n_mentions += len(record["mentions"])
            # n_chemical_mentions += sum([len(e["mention_indices"]) for e in record["entities"] if e["entity_type"] == "Chemical"])
            # n_disease_mentions += sum([len(e["mention_indices"]) for e in record["entities"] if e["entity_type"] == "Disease"])
            # n_chemical_entities += len([e for e in record["entities"] if e["entity_type"] == "Chemical"])
            # n_disease_entities += len([e for e in record["entities"] if e["entity_type"] == "Disease"])
            # n_positive_pairs += len(record["relations"])
            # n_negative_pairs += len(negative_pairs)

            # path_output_file = os.path.join(path_output_dir, doc_key + ".json")
            # utils.write_json(path_output_file, record)
            # records[doc_key] = record
            records.append(record)
    utils.write_json(path_output_file, records)

    print(f"Processed {len(doc_key_list)} documents.")
    print(f"Average number of sentences per doc: {n_sentences}/{len(doc_key_list)} = {float(n_sentences) / len(doc_key_list)}")
    print(f"Average number of mentions per doc: {n_mentions}/{len(doc_key_list)} = {float(n_mentions) / len(doc_key_list)}")
    # print(f"Average number of chemical mentions per doc: {n_chemical_mentions}/{len(doc_key_list)} = {float(n_chemical_mentions) / len(doc_key_list)}")
    # print(f"Average number of disease mentions per doc: {n_disease_mentions}/{len(doc_key_list)} = {float(n_disease_mentions) / len(doc_key_list)}")
    # print(f"Average number of chemical entities per doc: {n_chemical_entities}/{len(doc_key_list)} = {float(n_chemical_entities) / len(doc_key_list)}")
    # print(f"Average number of disease entities per doc: {n_disease_entities}/{len(doc_key_list)} = {float(n_disease_entities) / len(doc_key_list)}")
    # print(f"Average number of positive entity pairs per doc: {n_positive_pairs}/{len(doc_key_list)} = {float(n_positive_pairs) / len(doc_key_list)}")
    # print(f"Average number of negative entity pairs per doc: {n_negative_pairs}/{len(doc_key_list)} = {float(n_negative_pairs) / len(doc_key_list)}")


def get_chunks(lst):
    chunk_length = 17
    chunks = []
    for i in range(0, len(lst), chunk_length):
        chunk = lst[i:i+chunk_length]
        assert len(chunk) == chunk_length
        dct = {
            "relation": chunk[0], # 1:CID:2, 1:NR:2, not_include
            "direction": chunk[1], # L2R or R2L
            "intra_or_itner": chunk[2], # NON-CROSS or CROSS
            "_1": chunk[3],
            "_2": chunk[4],
            #
            "arg1_entity_id": chunk[5],
            "arg1_entity_type": chunk[7],
            "arg1_mentions": chunk[6].split("|"),
            "arg1_mention_begin_token_indices": [int(x) for x in chunk[8].split(":")],
            "arg1_mention_end_token_indices": [int(x) for x in chunk[9].split(":")],
            "arg1_mention_index_to_sentence_index": [int(x) for x in chunk[10].split(":")],
            #
            "arg2_entity_id": chunk[11],
            "arg2_entity_type": chunk[13],
            "arg2_mentions": chunk[12].split("|"),
            "arg2_mention_begin_token_indices": [int(x) for x in chunk[14].split(":")],
            "arg2_mention_end_token_indices": [int(x) for x in chunk[15].split(":")],
            "arg2_mention_index_to_sentence_index": [int(x) for x in chunk[16].split(":")],
        }
        assert dct["relation"] in ["1:CID:2", "1:NR:2", "not_include"], dct["relation"]
        if dct["relation"] in ["1:CID:2", "1:NR:2"]:
            dct["relation"] = dct["relation"].split(":")[1]
        assert dct["arg1_entity_type"] == "Chemical"
        assert dct["arg2_entity_type"] == "Disease"
        assert len(dct["arg1_mentions"]) == len(dct["arg1_mention_begin_token_indices"]) == len(dct["arg1_mention_end_token_indices"]) == len(dct["arg1_mention_index_to_sentence_index"])
        assert len(dct["arg2_mentions"]) == len(dct["arg2_mention_begin_token_indices"]) == len(dct["arg2_mention_end_token_indices"]) == len(dct["arg2_mention_index_to_sentence_index"])
        chunks += [dct]
    return chunks


def get_mentions(chunks):
    def get_mention_infos(chunk, arg):
        mentions = []
        entity_type = chunk[f"{arg}_entity_type"]
        # entity_id = chunk[f"{arg}_entity_id"]
        for mention_name, mention_begin_token_index, mention_end_token_index in zip(chunk[f"{arg}_mentions"], chunk[f"{arg}_mention_begin_token_indices"], chunk[f"{arg}_mention_end_token_indices"]):
            mention = {
                "span": (mention_begin_token_index, mention_end_token_index - 1),
                "name": mention_name,
                "entity_type": entity_type,
                # "entity_id": entity_id,
            }
            mentions.append(mention)
        return mentions

    mentions = []
    for chunk in chunks:
        ms = get_mention_infos(chunk=chunk, arg="arg1")
        mentions.extend(ms)
        ms = get_mention_infos(chunk=chunk, arg="arg2")
        mentions.extend(ms)
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


# def get_entities(chunks, mentions):
#     span2index = create_span_to_mention_index_dict(mentions=mentions)
#     entities = {}
#     for chunk in chunks:
#         # Arg1
#         entity_id = chunk["arg1_entity_id"]
#         entity_type = chunk["arg1_entity_type"]
#         mention_indices = [span2index[(b,e-1)] for b, e in zip(chunk["arg1_mention_begin_token_indices"], chunk["arg1_mention_end_token_indices"])]
#         mention_indices = sorted(mention_indices)
#         if not entity_id in entities:
#             entities[entity_id] = {
#                 "entity_type": entity_type,
#                 "mention_indices": mention_indices,
#             }
#         else:
#             assert entities[entity_id]["entity_type"] == entity_type
#             assert tuple(entities[entity_id]["mention_indices"]) == tuple(mention_indices)

#         # Arg2
#         entity_id = chunk["arg2_entity_id"]
#         entity_type = chunk["arg2_entity_type"]
#         mention_indices = [span2index[(b,e-1)] for b, e in zip(chunk["arg2_mention_begin_token_indices"], chunk["arg2_mention_end_token_indices"])]
#         mention_indices = sorted(mention_indices)
#         if not entity_id in entities:
#             entities[entity_id] = {
#                 "entity_type": entity_type,
#                 "mention_indices": mention_indices,
#             }
#         else:
#             assert entities[entity_id]["entity_type"] == entity_type
#             assert tuple(entities[entity_id]["mention_indices"]) == tuple(mention_indices)

#     entities = transform_entities(dct=entities)
#     entities = sorted(entities, key=lambda x: x["mention_indices"][0])
#     return entities


# def create_span_to_mention_index_dict(mentions):
#     dct = {}
#     for m_i, mention in enumerate(mentions):
#         dct[tuple(mention["span"])] = m_i
#     assert len(dct) == len(mentions)
#     return dct


# def transform_entities(dct):
#     entities = []
#     for entity_id in dct.keys():
#         entity_type = dct[entity_id]["entity_type"]
#         mention_indices = dct[entity_id]["mention_indices"]
#         entity = {
#             "mention_indices": mention_indices,
#             "entity_type": entity_type,
#             "entity_id": entity_id,
#         }
#         entities.append(entity)
#     return entities


# def get_relations(chunks, entity_id_to_index):
#     relations = []
#     for chunk in chunks:
#         if chunk["relation"] == "CID":
#             assert chunk["arg1_entity_type"] == "Chemical"
#             assert chunk["arg2_entity_type"] == "Disease"
#             # if chunk["direction"] == "L2R":
#             #     relation = (chunk["arg1_entity_id"], chunk["relation"], chunk["arg2_entity_id"])
#             # else:
#             #     assert chunk["direction"] == "R2L"
#             #     relation = (chunk["arg2_entity_id"], chunk["relation"], chunk["arg1_entity_id"])
#             relation = (chunk["arg1_entity_id"], chunk["relation"], chunk["arg2_entity_id"])
#             relations.append(relation)
#     assert len(relations) == len(set(relations))
#     relations = [{"arg1": entity_id_to_index[h], "relation": r, "arg2": entity_id_to_index[t]} for (h, r, t) in relations]
#     return relations


# def get_negative_pairs(chunks, entity_id_to_index):
#     epairs = []
#     for chunk in chunks:
#         if chunk["relation"] == "NR":
#             assert chunk["arg1_entity_type"] == "Chemical"
#             assert chunk["arg2_entity_type"] == "Disease"
#             # if chunk["direction"] == "L2R":
#             #     epair = (chunk["arg1_entity_id"], chunk["arg2_entity_id"])
#             # else:
#             #     assert chunk["direction"] == "R2L"
#             #     epair = (chunk["arg2_entity_id"], chunk["arg1_entity_id"])
#             epair = (chunk["arg1_entity_id"], chunk["arg2_entity_id"])
#             epairs.append(epair)
#     assert len(epairs) == len(set(epairs))
#     epairs = [{"arg1": entity_id_to_index[h], "arg2": entity_id_to_index[t]} for (h, t) in epairs]
#     return epairs


# def get_not_include_pairs(chunks, entity_id_to_index):
#     epairs = []
#     for chunk in chunks:
#         if chunk["relation"] == "not_include":
#             epair = (chunk["arg1_entity_id"], chunk["arg2_entity_id"])
#             epairs.append(epair)
#     assert len(epairs) == len(set(epairs))
#     epairs = [{"arg1": entity_id_to_index[h], "arg2": entity_id_to_index[t]} for (h, t) in epairs]
#     return epairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    main(args=args)
