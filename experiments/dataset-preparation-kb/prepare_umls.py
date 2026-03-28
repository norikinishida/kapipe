import argparse
from collections import defaultdict
import os

import pandas as pd
from tqdm import tqdm

import sys
sys.path.insert(0, "../../..")
from kapipe import utils


MRCONSO_HEADERS = [
    "CUI", # *Unique identifier for concept
    "LAT", # *Language of term
    "TS", # *Term status
    "LUI", # Unique identifier for term
    "STT", # *String type
    "SUI", # Unique identifier for string
    "ISPREF", # *Atom status - preferred (Y) or not (N) for this tring within this concept
    "AUI", # Unique identifier for atom
    "SAUI", # Source asserted atom identifier [optional]
    "SCUI", # Source asserted concept identifier [optional]
    "SDUI", # Source asserted descriptor identifier [optional]
    "SAB", # Abbreviated source name (SAB)
    "TTY", # Abbreviation for term type in source vucabulary
    "CODE", # Most useful source assrerted identifier, or a Metathesaurus-generated source entry identifier
    "STR", # *String
    "SRL", # Source restriction level
    "SUPPRESS", # Suppressible flag. Values = O, E, Y, or N
    "CVF" # Content View Flag.
]

MRSTY_HEADERS = [
    "CUI", # *Unique identifier for concept
    "TUI", # *Unique identifier of Semantic Type
    "STN", # *Semantic Type tree number
    "STY", # *Semantic Type
    "ATUI", # Unique identifier for attribute
    "CVF", # Content View Flag
]

MRDEF_HEADERS = [
    "CUI", # *Unique identifier for concept
    "AUI", # Unique identifier for atom
    "ATUI", # Unique identifier for attribute
    "SATUI", # Source asserted attribute identifier [optional]
    "SAB", # Abbreviated source name of the source of the definition
    "DEF", # *Definition
    "SUPPRESS", # Suppressible flag
    "CVF" # Content View Flag
]

MRREL_HEADERS = [
    "CUI1", # *Unique identifier of first concept
    "AUI1", # Unique identifier of first atom
    "STYPE1", # The name of the column in MRCONSO.RRF that contains the identifier used for the first element in the relationship
    #
    "REL", # Relationship of second concept or atom to first concept or atom
    #
    "CUI2", # *Unique identifier of second concept
    "AUI2", # Unique identifier of second atom
    "STYPE2", # The name of the column in MRCONSO.REF that contains the identifier used for the second element in the relationship
    #
    "RELA", # *Additional (more specific) relationship label [optional]
    #
    "RUI", # Unique identifier of relationship
    "SRUI", # Source asserted relationship identifier, if present
    #
    "SAB", # Abbreviated source name of the source of relationship
    "SL", # Source of relationship labels
    #
    "RG", # Relationship group
    "DIR", # Source asserted directionality flag
    "SUPPRESS", # Suppressible flag
    "CVF" # Content View Flag
]

SRDEF_HEADERS = [
    "RT", # *Record Type (STY = Semantic Type or RL = Relation)
    "UI", # Unique identifier of the Semantic Type or Relation
    "STY/RL", # *Name of the Semantic Type or Relation
    "STN/RTN", # *Tree number of the Semantic Type or Relation
    "DEF", # *Definition of the Semantic Type or Relation
    "EX", # Examples of Metathesaurus concepts with this Semantic Type (STY records only)
    "UN", # Usage note for Semantic Type assignment (STY records only)
    "NH", # The Semantic Type and its descendents allow the non-human flag (STY records only)
    "ABR", # Abbreviation of the Relation Name or Semantic Type
    "RIN" # Inverse of the Relation (RL records only)
]

SRSTRE2_HEADERS = [
    "STY1",
    "RL",
    "STY2"
]


def main(args):
    path_input_dir = args.input_dir 
    path_output_file = args.output_file

    utils.mkdir(os.path.dirname(path_output_file))

    # Extract concepts from the UMLS RRF (Rich Release Format) files: MRCONSO, MRSTY, MRDEF
    entity_dict = construct_entity_dict_from_mrconso(
        path=os.path.join(path_input_dir, "META/MRCONSO.RRF")
    )
    entity_dict = update_entity_dict_by_mrsty(
        path=os.path.join(path_input_dir, "META/MRSTY.RRF"),
        entity_dict=entity_dict
    )
    entity_dict = update_entity_dict_by_mrdef(
        path=os.path.join(path_input_dir, "META/MRDEF.RRF"),
        entity_dict=entity_dict
    )
    entity_dict = sorted(list(entity_dict.values()), key=lambda e: e["entity_id"])
    utils.write_json(path_output_file, entity_dict)

    # Extract triples from MRREL
    triples = construct_triples_from_mrrel(
        path=os.path.join(path_input_dir, "META/MRREL.RRF")
    )
    utils.write_json(
        path_output_file.replace(".entity_dict.json", ".triples.json"),
        triples
    )

    # Extract Semantic Network information from SRDEF
    semantic_type_dict, semantic_relation_dict = construct_semantic_network_dicts_from_srdef(
        path_srdef=os.path.join(path_input_dir, "NET/SRDEF"),
        path_srstre2=os.path.join(path_input_dir, "NET/SRSTRE2")
    )
    utils.write_json(
        path_output_file.replace(".entity_dict.json", ".semantic_types.json"),
        semantic_type_dict
    )
    utils.write_json(
        path_output_file.replace(".entity_dict.json", ".semantic_relations.json"),
        semantic_relation_dict 
    )

    # Extract MeSH-to-UMLS mapping from MRCONSO
    mesh_to_umls_map = construct_mesh_to_umls_map_from_mrconso(
        path=os.path.join(path_input_dir, "META/MRCONSO.RRF")
    )
    utils.write_json(
        path_output_file.replace(".entity_dict.json", ".mesh_to_umls_map.json"),
        mesh_to_umls_map
    )

    print("Completed.")
    

def construct_entity_dict_from_mrconso(path):
    df = pd.read_csv(path, sep="|", header=None, names=MRCONSO_HEADERS + ["extra"], dtype={k: "str" for k in MRCONSO_HEADERS})
    df = df.iloc[:, :-1]
    df = df[df["LAT"] == "ENG"]
    print(f"Loaded {len(df)} records (terms)")
    print(f"Found {len(set(df['CUI']))} unique concepts (entities)")

    n_rows = len(df)
    entity_dict = {}
    for _, row in tqdm(df.iterrows(), total=n_rows):
        entity_id = str(row["CUI"])
        name = str(row["STR"])
        if not entity_id in entity_dict:
            entity_dict[entity_id] = {
                "entity_id": entity_id, # from MRCONSO
                "entity_index": len(entity_dict),
                "entity_types": [], # from MRSTY
                "entity_type_names": [], # from MRSTY
                "canonical_name": "", # from MRCONSO
                "synonyms": set(), # from MRCONSO
                "description": [], # from MRDEF
                "tree_numbers": [], # from MRSTY
            }
        # NOTE:
        #   TS: P (Preferred LUI of the CUI), S (Non-Preferred LUI of the CUI)
        #   SST: PF (Preferred form of term), VCW (Case and word-order variant of the preferred form), VC (Case variant of the preferred form),
        #        VO (Variant of the preferred form), VW (Word-order variant of the preferred form)
        if row["TS"] == "P" and row["ISPREF"] == "Y" and row["STT"] == "PF":
            assert entity_dict[entity_id]["canonical_name"] == ""
            entity_dict[entity_id]["canonical_name"] = name
        else:
            entity_dict[entity_id]["synonyms"].add(name)
        
        # if len(entity_dict) > 10000:
        #     break

    # Remove redundant synonyms
    for key, value in entity_dict.items():
        assert value["canonical_name"] != ""
        synonyms = sorted(list(value["synonyms"] - {value["canonical_name"]}))
        entity_dict[key]["synonyms"] = synonyms

    print(f"Processed {len(entity_dict)} entities")
    return entity_dict
 
    
def update_entity_dict_by_mrsty(path, entity_dict):
    df = pd.read_csv(path, sep="|", header=None, names=MRSTY_HEADERS + ["extra"], dtype={k: "str" for k in MRSTY_HEADERS})
    df = df.iloc[:, :-1]
    print(f"Loaded {len(df)} records (entity type - concept links)")

    n_rows = len(df)
    for _, row in tqdm(df.iterrows(), total=n_rows):
        entity_id = str(row["CUI"])
        entity_type = str(row["TUI"])
        entity_type_name = str(row["STY"])
        tree_number = str(row["STN"])
        if not entity_id in entity_dict:
            continue
        entity_dict[entity_id]["entity_types"].append(entity_type)
        entity_dict[entity_id]["entity_type_names"].append(entity_type_name)
        entity_dict[entity_id]["tree_numbers"].append(tree_number)

    for key, value in entity_dict.items():
        tree_numbers = sorted(value["tree_numbers"])
        entity_dict[key]["tree_numbers"] = tree_numbers

    print(f"Processed {len(df)} entity type - concept links")
    return entity_dict


def update_entity_dict_by_mrdef(path, entity_dict):
    df = pd.read_csv(path, sep="|", header=None, names=MRDEF_HEADERS + ["extra"], dtype={k: "str" for k in MRDEF_HEADERS})
    df = df.iloc[:, :-1]
    print(f"Loaded {len(df)} records (concept definitions)")

    n_rows = len(df)
    for _, row in tqdm(df.iterrows(), total=n_rows):
        entity_id = str(row["CUI"])
        definition = str(row["DEF"])
        if not entity_id in entity_dict:
            continue
        entity_dict[entity_id]["description"].append(definition)

    for key, value in entity_dict.items():
        desc = " | ".join(value["description"])
        entity_dict[key]["description"] = desc

    print(f"Processed {len(df)} concept definitions")
    return entity_dict


def construct_triples_from_mrrel(path):
    df = pd.read_csv(path, sep="|", header=None, names=MRREL_HEADERS + ["extra"])
    df = df.iloc[:, :-1]
    print(f"Loaded {len(df)} records (triples with or without RELA)")

    triples = []
    n_rows = len(df)
    for _, row in tqdm(df.iterrows(), total=n_rows):
        cui1 = str(row["CUI1"])
        cui2 = str(row["CUI2"])
        relation_class= str(row["REL"])
        relation = str(row["RELA"])
        source = str(row["SAB"])
        if relation.strip() == "nan":
            continue
        if source == "nan":
            source = ""
        triples.append({
            "head": cui2,
            "tail": cui1,
            "relation": relation,
            "relation_class": relation_class,
            "source": source
        })

    print(f"Processed {len(triples)} triples")
    return triples


def construct_semantic_network_dicts_from_srdef(path_srdef, path_srstre2):
    df = pd.read_csv(path_srdef, sep="|", header=None, names=SRDEF_HEADERS + ["extra"])
    df = df.iloc[:, :-1]
    print(f"Loaded {len(df)} records (semantic types + semantic relations)")

    dict_list = df.to_dict(orient="records") # list[dict]
    semantic_type_dict = [
        {
            "semantic_type_id": d["UI"],
            "semantic_type_name": d["STY/RL"],
            "definition": d["DEF"],
            "tree_number": d["STN/RTN"]
        }
        for d in dict_list if d["RT"] == "STY"
    ]
    semantic_relation_dict = [
        {
            "semantic_relation_id": d["UI"],
            "semantic_relation_name": d["STY/RL"],
            "definition": d["DEF"],
            "tree_number": d["STN/RTN"]
        }
        for d in dict_list if d["RT"] == "RL"
    ]

    print(f"Processed {len(semantic_type_dict)} semantic types")
    print(f"Processed {len(semantic_relation_dict)} semantic relations")

    df = pd.read_csv(path_srstre2, sep="|", header=None, names=SRSTRE2_HEADERS + ["extra"])
    df = df.iloc[:, :-1]
    print(f"Loaded {len(df)} records (semantic type-relation-type triples)")

    rel2pairs = defaultdict(list)
    dict_list = df.to_dict(orient="records") # list[dict]
    for d in dict_list:
        type1 = d["STY1"]
        type2 = d["STY2"]
        rel = d["RL"]
        if type1[0].isupper() and type2[0].isupper():
            rel2pairs[rel].append((type1, type2))
    for rel, pairs in rel2pairs.items():
        pairs = sorted(list(set(pairs)))
        rel2pairs[rel] = pairs

    for d_i, d in enumerate(semantic_relation_dict):
        rel = d["semantic_relation_name"]
        pairs = rel2pairs[rel]
        semantic_relation_dict[d_i] = {
            "semantic_relation_id": d["semantic_relation_id"],
            "semantic_relation_name": d["semantic_relation_name"],
            "definition": d["definition"],
            "semantic_type_pairs": pairs,
            "tree_number": d["tree_number"],
        }
 
    return semantic_type_dict, semantic_relation_dict


def construct_mesh_to_umls_map_from_mrconso(path):
    df = pd.read_csv(path, sep="|", header=None, names=MRCONSO_HEADERS + ["extra"], dtype={k: "str" for k in MRCONSO_HEADERS})
    df = df.iloc[:, :-1]
    df = df[df["LAT"] == "ENG"]
    print(f"Loaded {len(df)} records (terms)")
    print(f"Found {len(set(df['CUI']))} unique concepts (entities)")

    n_rows = len(df)
    mesh_to_umls_map = {}
    for _, row in tqdm(df.iterrows(), total=n_rows):
        umls_id = str(row["CUI"])
        mesh_id = str(row["CODE"])
        source = str(row["SAB"])
        if source == "MSH" and mesh_id != "nan":
            if not mesh_id in mesh_to_umls_map:
                mesh_to_umls_map[mesh_id] = set()
            mesh_to_umls_map[mesh_id].add(umls_id)

    for mesh_id, umls_ids in mesh_to_umls_map.items():
        mesh_to_umls_map[mesh_id] = list(umls_ids)
    print(f"Processed {len(mesh_to_umls_map)} MeSH-to-UMLS mappings")
    return mesh_to_umls_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    main(args=args)

