import argparse
import os

# from datasets import load_dataset
from tqdm import tqdm

import sys
sys.path.insert(0, "../../..")
from kapipe import utils


BIO2_NER_TAGS = [
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-MISC",
    "I-MISC",
]

def main(args):
    path_input_file = args.input_file
    path_output_file = args.output_file
    utils.mkdir(os.path.dirname(path_output_file))

    n_docs = 0
    n_sentences = 0
    n_mentions = 0
    with open(path_input_file, encoding="utf-8") as f:
        # documents = {} # Dict[str, Document]
        documents = [] # list[Document]
        doc_i = 0
        ss_tokens = [] # List[List[str]]; tokens for sentences
        ss_ner_tags = [] # List[List[str]]; NER tags for sentences
        s_tokens = [] # List[str]; tokens for a sentence
        s_ner_tags = [] # List[str]; NER tags for a sentence
        for line in tqdm(f):
            line = line.strip()
            # document beginning
            if line.startswith("-DOCSTART-"):
                if len(ss_tokens) != 0:
                    # add document
                    assert len(ss_ner_tags) != 0
                    if len(s_tokens) != 0:
                        # add sentence
                        assert len(s_ner_tags) != 0
                        ss_tokens.append(s_tokens)
                        ss_ner_tags.append(s_ner_tags)
                        s_tokens = []
                        s_ner_tags = []
                    mentions = get_mentions(ss_tokens=ss_tokens, ss_ner_tags=ss_ner_tags)
                    document = get_document(ss_tokens=ss_tokens, mentions=mentions)
                    # documents[f"ID{doc_i}"] = document
                    documents.append({"doc_key" : f"ID{doc_i}"} | document)
                    doc_i += 1
                    n_docs += 1
                    n_sentences += len(document["sentences"])
                    n_mentions += len(document["mentions"])
                # init
                ss_tokens = []
                ss_ner_tags = []
            # sentence ending
            elif line == "":
                if len(s_tokens) != 0:
                    assert len(s_ner_tags) != 0
                    ss_tokens.append(s_tokens)
                    ss_ner_tags.append(s_ner_tags)
                s_tokens = []
                s_ner_tags = []
            # token
            else:
                items = line.split(" ")
                assert len(items) == 4, items
                token = items[0]
                # pos_tag = items[1]
                # chunk_tag = items[2]
                ner_tag = items[3]
                s_tokens.append(token)
                s_ner_tags.append(ner_tag)
        # add document
        if len(ss_tokens) != 0:
            # add document
            assert len(ss_ner_tags) != 0
            if len(s_tokens) != 0:
                # add sentence
                assert len(s_ner_tags) != 0
                ss_tokens.append(s_tokens)
                ss_ner_tags.append(s_ner_tags)
                s_tokens = []
                s_ner_tags = []
            mentions = get_mentions(ss_tokens=ss_tokens, ss_ner_tags=ss_ner_tags)
            document = get_document(ss_tokens=ss_tokens, mentions=mentions)
            # documents[f"ID{doc_i}"] = document
            documents.append({"doc_key" : f"ID{doc_i}"} | document)
            doc_i += 1
            n_docs += 1
            n_sentences += len(document["sentences"])
            n_mentions += len(document["mentions"])
    utils.write_json(path_output_file, documents)
    print(f"Processed {n_docs} documents.")
    print(f"Average number of sentences per doc: {n_sentences}/{n_docs} = {float(n_sentences)/n_docs}")
    print(f"Average number of mentions per doc: {n_mentions}/{n_docs} = {float(n_mentions)/n_docs}")

    entity_type_to_id = {}
    for document in documents:
        for mention in document["mentions"]:
            etype = mention["entity_type"]
            if etype in entity_type_to_id:
                continue
            entity_type_to_id[etype] = len(entity_type_to_id)
    utils.mkdir(os.path.join(os.path.dirname(path_output_file), "meta"))
    utils.write_json(os.path.join(os.path.dirname(path_output_file), "meta", "entity_type_to_id.json"), entity_type_to_id)

    # datasets = load_dataset("conll2003") # Huggingface

    # for split in ["train", "validation", "test"]:
    #     dataset = datasets[split]
    #     records = {}
    #     n_mentions = 0
    #     n_docs = 0
    #     for data_i in tqdm(range(len(dataset))):
    #         data = dataset[data_i]
    #         doc_key = data["id"] # str
    #         tokens = data["tokens"] # List[str]
    #         sentences = [" ".join(tokens)]
    #         # pos_tags = data["pos_tags"] # List[int]
    #         # chunk_tags = data["chunk_tags"] # List[int]
    #         ner_tag_labels = data["ner_tags"] # List[int]
    #         ner_tags = [BIO2_NER_TAGS[l] for l in ner_tag_labels] # List[str]

    #         record = {}
    #         record["sentences"] = sentences
    #         record["BIO2_ner_tags"] = ner_tags
    #         record["mentions"] = get_mentions(tokens=tokens, ner_tags=ner_tags)
    #         n_mentions += len(record["mentions"])

    #         records[doc_key] = record
    #         n_docs += 1
    #     utils.write_json(os.path.join(path_output_dir, split + ".json"), records)

    #     print(f"Processed {len(records)} documents.")
    #     print(f"Average number of mentions per doc: {n_mentions}/{n_docs} = {float(n_mentions)/n_docs}")

    # utils.mkdir(os.path.join(path_output_dir, "meta"))
    # bio2_ner_tag_to_id = {y:i for i, y in enumerate(BIO2_NER_TAGS)}
    # entity_type_to_id = {}
    # for ner_tag in BIO2_NER_TAGS:
    #     if ner_tag == "O":
    #         continue
    #     ner_tag = ner_tag.split("-")
    #     bio_tag = ner_tag[0]
    #     type_tag = "-".join(ner_tag[1:])
    #     if type_tag in entity_type_to_id:
    #         continue
    #     entity_type_to_id[type_tag] = len(entity_type_to_id)
    # utils.write_json(os.path.join(path_output_dir, "meta", "bio2_ner_tag_to_id.json"), bio2_ner_tag_to_id)
    # utils.write_json(os.path.join(path_output_dir, "meta", "entity_type_to_id.json"), entity_type_to_id)


def get_mentions(ss_tokens, ss_ner_tags):
    mentions = [] # List[{"span": (int,int), "name": str, "type": str}]
    assert len(ss_tokens) == len(ss_ner_tags) # number of sentences
    offset = 0
    for s_tokens, s_ner_tags in zip(ss_tokens, ss_ner_tags):
        s_mentions = _get_mentions(tokens=s_tokens, ner_tags=s_ner_tags, offset=offset)
        mentions.extend(s_mentions)
        offset += len(s_tokens)
    return mentions


def _get_mentions(tokens, ner_tags, offset):
    mentions = []
    begin_token_i = None
    end_token_i = None
    bio_tag_prev = "O"
    type_tag_prev = None
    for token_i, (token, ner_tag) in enumerate(zip(tokens, ner_tags)):
        # Split the NER tag to BIO tag and Type tag
        if ner_tag == "O":
            bio_tag = "O"
            type_tag = None
        else:
            ner_tag = ner_tag.split("-")
            bio_tag = ner_tag[0]
            type_tag = "-".join(ner_tag[1:])

        # Decide an action based on the BIO tag transition pattern
        if bio_tag == "O":
            # [O, O] -> do nothing
            # [B, O] -> add (span-length=1)
            # [I, O] -> add (span-length>1)
            if bio_tag_prev in ["B", "I"]:
                mentions.append({
                    "span": (offset + begin_token_i, offset + end_token_i),
                    "name": " ".join(tokens[begin_token_i: end_token_i + 1]),
                    "entity_type": type_tag_prev,
                })
            begin_token_i = None
            end_token_i = None
            bio_tag_prev = "O"
            type_tag_prev = None
        elif bio_tag == "B":
            # [O, B] -> span begin
            # [B, B] -> add (span-length=1)
            # [I, B] -> add (span-length>1)
            if bio_tag_prev in ["B", "I"]:
                mentions.append({
                    "span": (offset + begin_token_i, offset + end_token_i),
                    "name": " ".join(tokens[begin_token_i: end_token_i + 1]),
                    "entity_type": type_tag_prev,
                })
            begin_token_i = token_i
            end_token_i = token_i
            bio_tag_prev = bio_tag
            type_tag_prev = type_tag
        elif bio_tag == "I":
            # [O, I] -> span begin (valid for BIO1, while invalid for BIO2)
            # [B, I] -> continue
            # [I-A, I-A] -> continue
            # [I-A, I-B] -> add (only for BIO1)
            # if bio_tag_prev == "O":
            #     if token_i == 0:
            #         raise Exception(f"Invalid BIO2 NER tag appears at the first token position: {ner_tags[token_i]}")
            #     else:
            #         raise Exception(f"Invalid BIO2 NER tag transition: {ner_tags[token_i-1]} -> {ner_tags[token_i]}")
            if bio_tag_prev == "O":
                begin_token_i = token_i
                end_token_i = token_i
                bio_tag_prev = bio_tag
                type_tag_prev = type_tag
            elif bio_tag_prev == "B":
                end_token_i = token_i
                assert type_tag_prev == type_tag, (tokens, ner_tags)
            else:
                if type_tag_prev == type_tag:
                    end_token_i = token_i
                else:
                    mentions.append({
                        "span": (offset + begin_token_i, offset + end_token_i),
                        "name": " ".join(tokens[begin_token_i: end_token_i + 1]),
                        "entity_type": type_tag_prev,
                    })
                    begin_token_i = token_i
                    end_token_i = token_i
                    bio_tag_prev = bio_tag
                    type_tag_prev = type_tag
        else:
            raise Exception(f"Invalid BIO2 NER tag: {ner_tags[token_i]}")

    if bio_tag_prev != "O":
        mentions.append({
            "span": (offset + begin_token_i, offset + end_token_i),
            "name": " ".join(tokens[begin_token_i: end_token_i + 1]),
            "entity_type": type_tag_prev,
        })

    return mentions


def get_document(ss_tokens, mentions):
    document = {}
    document["sentences"] = [" ".join(s) for s in ss_tokens]
    document["mentions"] = mentions
    return document


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    main(args=args)
