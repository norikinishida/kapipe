import argparse
import os

from spacy_alignments import tokenizations
from tqdm import tqdm

import sys
sys.path.insert(0, "../../..")
from kapipe import utils
from kapipe.chunking import Chunker


MM_ST21PV_SEMANTIC_TYPES = [
    ("Health Care Activity", "T058", 4),
    ("Research Activity", "T062", 4),
    ("Injury or Poisoning", "T037", 3),
    ("Biologic Function", "T038", 4),
    ("Virus", "T005", 4),
    ("Bacterium", "T007", 4),
    ("Eukaryote", "T204", 4),
    ("Anatomical Structure", "T017", 3),
    ("Medical Device", "T074", 4),
    ("Body Substance", "T031", 4),
    ("Chemical", "T103", 4),
    ("Food", "T168", 4),
    ("Clinical Attribute", "T201", 4),
    ("Finding", "T033", 3),
    ("Spatial Concept", "T082", 4),
    ("Body System", "T022", 5),
    ("Biomedical Occupation or Discipline", "T091", 4),
    ("Organization", "T092", 3),
    ("Professional or Occupational Group", "T097", 4),
    ("Population Group", "T098", 4),
    ("Intellectual Product", "T170", 3)
]


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir

    utils.mkdir(output_dir)

    semantic_types = [
        {
            "id": tui,
            "name": name,
            "level": level
        }
        for (name, tui, level) in MM_ST21PV_SEMANTIC_TYPES
    ]
    utils.mkdir(os.path.join(output_dir, "meta"))
    utils.write_json(os.path.join(output_dir, "meta", "st21pv_semantic_types.json"), semantic_types)

    # Read documents
    documents = read_documents_from_pubtator_corpus(
        path=os.path.join(input_dir, "st21pv", "data", "corpus_pubtator.txt"),
        semantic_types=semantic_types
    )

    # Tokenize the title and abstract into sentences
    documents = tokenize_title_and_abstract(documents)

    # Remove duplicate mentions
    documents = remove_duplicate_mentions(documents)

    # Adjust spans
    documents = adjust_spans(documents)

    # Remove Entity ID annotations
    documents = remove_entity_id_annotations(documents)

    # Split the documents into splits 
    train_pmids = set(utils.read_lines(os.path.join(input_dir, "full", "data", "corpus_pubtator_pmids_trng.txt")))
    dev_pmids = set(utils.read_lines(os.path.join(input_dir, "full", "data", "corpus_pubtator_pmids_dev.txt")))
    test_pmids = set(utils.read_lines(os.path.join(input_dir, "full", "data", "corpus_pubtator_pmids_test.txt")))
    train_docs = [d for d in documents if d["doc_key"] in train_pmids]
    dev_docs = [d for d in documents if d["doc_key"] in dev_pmids]
    test_docs = [d for d in documents if d["doc_key"] in test_pmids]

    # Save the documents
    utils.write_json(os.path.join(output_dir, "train.json"), train_docs)
    utils.write_json(os.path.join(output_dir, "dev.json"), dev_docs)
    utils.write_json(os.path.join(output_dir, "test.json"), test_docs)
    print(f"Saved {len(train_docs)} training documents")
    print(f"Saved {len(dev_docs)} development documents")
    print(f"Saved {len(test_docs)} test documents")

    # Show the stats
    show_stats(documents, title="Total")
    show_stats(train_docs, title="Training")
    show_stats(dev_docs, title="Development")
    show_stats(test_docs, title="Test")

   
def read_documents_from_pubtator_corpus(path, semantic_types):
    # semantic_types = set([x["id"] for x in semantic_types])
    semantic_type_map = {d["id"]: d["name"] for d in semantic_types}
    documents = []
    document = {}
    n_lines = len(open(path).readlines())
    with open(path, encoding="utf-8") as f:
        for line in tqdm(f, total=n_lines):
            line = line.strip()
            items = line.split("|")

            if line == "":
                if "doc_key" in document:
                    documents.append(document)
                    document = {}

            elif len(items) > 1 and items[1] == "t":
                # Process title
                doc_key = items[0]
                title = " ".join(items[2:])
                assert not "doc_key" in document
                assert not "title" in document
                document["doc_key"] = doc_key
                document["title"] = title

            elif len(items) > 1 and items[1] == "a":
                # Process abstract and sentences (title+abstract)
                doc_key = items[0]
                abstract = " ".join(items[2:])
                assert document["doc_key"] == doc_key
                assert not "text" in document
                document["text"] = abstract
                document["sentences"] = None

            else:
                # Process mention 
                items = line.split("\t")
                assert len(items) == 6, items
                doc_key = items[0]
                begin_char_index = int(items[1])
                end_char_index = int(items[2])
                name = items[3]
                entity_type = items[4]
                entity_id = items[5]
                assert document["doc_key"] == doc_key
                assert len(entity_type.split("|")) == 1
                assert len(entity_type.split(",")) == 1
                assert len(entity_id.split("|")) == 1
                assert len(entity_id.split(",")) == 1

                text = document["title"] + " " + document["text"]
                assert text[begin_char_index: end_char_index] == name

                # assert entity_type in semantic_types
                assert entity_type in semantic_type_map

                assert entity_id.startswith("UMLS:")
                entity_id = entity_id.replace("UMLS:", "")

                if not "mentions" in document:
                    document["mentions"] = []

                document["mentions"].append({
                    "span": (begin_char_index, end_char_index - 1),
                    "name": name,
                    "entity_type": semantic_type_map[entity_type],
                    "entity_id": entity_id
                })

    if "doc_key" in document:
        documents.append(document)

    return documents


def tokenize_title_and_abstract(documents):
    chunker = Chunker()
    for doc_i, doc in enumerate(documents):
        # Tokenize the title
        sent0 = " ".join(chunker.split_to_tokens(insert_spaces(
            text=doc["title"],
            mentions=doc["mentions"],
            offset=0
        ))) # str
        # Tokenize the abstract
        sents = chunker.split_to_tokenized_sentences(insert_spaces(
            text=doc["text"],
            mentions=doc["mentions"],
            offset=len(doc["title"]) + 1
        )) # list[list[str]]
        sents = [" ".join(s) for s in sents] # list[str]
        # Combine the tokenized title and abstract
        sents = [sent0] + sents # list[str]
        doc["sentences"] = sents
        documents[doc_i] = doc
    return documents
    
    
def insert_spaces(text, mentions, offset):
    chars = list(text)
    n_chars = len(chars)
    for mention in mentions:
        begin_char_i, end_char_i = mention["span"]
        begin_char_i -= offset
        end_char_i -= offset
        if begin_char_i < 0 or end_char_i < 0:
            continue
        if begin_char_i >= n_chars or end_char_i >= n_chars:
            continue
        chars[begin_char_i] = " " + chars[begin_char_i]
        chars[end_char_i] = chars[end_char_i] + " "
    text = "".join(chars)
    text = text.replace("(", " ( ").replace(")", " ) ")
    text = text.replace("{", " { ").replace("}", " } ")
    text = text.replace("[", " [ ").replace("]", " ] ")
    text = " ".join(text.split())
    return text


def postprocess_sentences(sentences, mentions):
    # sentences = [s.split() for s in sentences]
    while True:
        need_change = False
        token_index_to_sent_index = [] # list[int]
        for sent_i, sent in enumerate(sentences):
            token_index_to_sent_index.extend([sent_i for _ in range(len(sent))])

        for mention in mentions:
            (begin_token_index, end_token_index) = mention["span"]
            begin_sent_index = token_index_to_sent_index[begin_token_index]
            end_sent_index = token_index_to_sent_index[end_token_index]
            if begin_sent_index != end_sent_index:
                assert begin_sent_index + 1 == end_sent_index
                s0 = sentences[:begin_sent_index]
                s1 = sentences[begin_sent_index]
                s2 = sentences[begin_sent_index + 1]
                s3 = sentences[begin_sent_index + 1:]
                sentences = s0 + [s1 + s2] + s3
                need_change = True
                break
        if not need_change:
            break
    # sentences = [" ".join(s) for s in sentences]
    return sentences


def remove_duplicate_mentions(documents):
    for doc_i, doc in enumerate(documents):
        tuples = [(tuple(m["span"]), m["name"], m["entity_type"], m["entity_id"]) for m in doc["mentions"]]
        n_original = len(tuples)
        tuples = sorted(list(set(tuples)), key=lambda x: x[0])
        n_reduced = len(tuples)
        if n_original != n_reduced:
            print(f"Removed {n_original - n_reduced} duplicate mentions {{(span, name, type, concept)}} in {doc['doc_key']}")
        doc["mentions"] = [
            {
                "span": span,
                "name": name,
                "entity_type": etype,
                "entity_id": eid
            }
            for (span, name, etype, eid) in tuples
        ]
        documents[doc_i] = doc
    return documents


def adjust_spans(documents):
    for doc_i, doc in enumerate(documents):
        chars = list(doc["title"] + " " + doc["text"]) # list[str]
        tokens = " ".join(doc["sentences"]).split(" ") # list[str]
        char_to_tok, _ = tokenizations.get_alignments(chars, tokens)
        for m_i, mention in enumerate(doc["mentions"]):
            begin_char_i, end_char_i = mention["span"]

            begin_tok_i = char_to_tok[begin_char_i]
            assert len(begin_tok_i) == 1
            begin_tok_i = begin_tok_i[0]

            end_tok_i = char_to_tok[end_char_i]
            assert len(end_tok_i) == 1
            end_tok_i = end_tok_i[0]

            mention["span"] = (begin_tok_i, end_tok_i)

            # print(tokens[begin_tok_i: end_tok_i + 1], mention["name"])
            assert " ".join(tokens[begin_tok_i: end_tok_i + 1]).replace(" ", "") == mention["name"].replace(" ", "")

            doc["mentions"][m_i] = mention

        doc["sentences"] = adjust_sentence_splitting(
            doc["sentences"], doc["mentions"]
        )
            
        documents[doc_i] = doc
    return documents

    
def adjust_sentence_splitting(sentences, mentions):
    sentences = [s.split() for s in sentences]
    while True:
        need_change = False
        token_index_to_sent_index = [] # list[int]
        for sent_i, sent in enumerate(sentences):
            token_index_to_sent_index.extend([sent_i for _ in range(len(sent))])
        for mention in mentions:
            (begin_token_index, end_token_index) = mention["span"]
            begin_sent_index = token_index_to_sent_index[begin_token_index]
            end_sent_index = token_index_to_sent_index[end_token_index]
            if begin_sent_index != end_sent_index:
                left = sentences[:begin_sent_index]
                merged = [w for s in sentences[begin_sent_index: end_sent_index + 1] for w in s]
                right = sentences[end_sent_index + 1:]
                sentences = left + [merged] + right
                print(f"Merged sentence[{begin_sent_index}](for token[{begin_token_index}]) - sentence[{end_sent_index}](for token[{end_token_index}])")
                need_change = True
                break
        if not need_change:
            break
    sentences = [" ".join(s) for s in sentences]
    return sentences


def remove_entity_id_annotations(documents):
    for doc_i, doc in enumerate(documents):
        new_mentions = []
        for mention in doc["mentions"]:
            new_mentions.append({
                "span": mention["span"],
                "name": mention["name"],
                "entity_type": mention["entity_type"]
            })
        doc["mentions"] = new_mentions
        documents[doc_i] = doc
    return documents


def show_stats(documents, title):
    n_docs = len(documents)
    n_mentions = 0
    entities = set()
    for doc in documents:
        n_mentions += len(doc["mentions"])
        # ents = [m["entity_id"] for m in doc["mentions"]]
        # entities.update(ents)
    print(f"{title}:")
    print(f"Number of documents: {n_docs}")
    print(f"Number of mentions: {n_mentions}")
    # print(f"Number of entities: {len(entities)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    main(args=args)
 