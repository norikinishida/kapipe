import argparse
import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


def main(args):
    paths_input_file = args.input_files
    path_output_file = args.output_file
    if not os.path.exists(os.path.dirname(path_output_file)):
        os.makedirs(os.path.dirname(path_output_file))

    # Get entity dictionary (class ID -> EntityPage) from the original CSV file
    print("Building an (integrated) entity dictionary ...")
    entity_dict = None
    kb_names = []
    for path in paths_input_file:
        kb_name = os.path.splitext(os.path.basename(path))[0]
        entity_dict = read_entity_dict_from_csv(path=path, kb_name=kb_name, entity_dict=entity_dict)
        kb_names.append(kb_name)
    entity_dict = remove_redundant_synonyms_and_parents(entity_dict=entity_dict)

    # Assign Tree Numbers
    print("Assignning Tree Numbers to each entity ...")
    entity_dict = assign_tree_numbers(entity_dict=entity_dict, kb_names=kb_names)
    entity_dict = remove_redundant_tree_numbers(entity_dict=entity_dict)

    # entity_dict = [{"entity_id": key} | val for key,val in entity_dict.items()]
    entity_dict = list(entity_dict.values())

    # Save
    print("Saving the entity dictionary to {path_outupt_file} ...")
    write_json(path_output_file, entity_dict)

    print("Done.")


# ---------------------------


def read_entity_dict_from_csv(path, kb_name, entity_dict=None):
    if entity_dict is None:
        entity_dict = {}

    # Read CSV
    # NOTE: The number of lines of the CSV file does not represent the number of records.
    #       A single record can be separated into multiple lines according to the multi-line definitions.
    df = read_csv(path, delimiter=",", with_head=True, with_id=False)

    # Extract columns of interest
    # df = df.loc[:, ["Class ID", "Preferred Label", "Synonyms", "Definitions", "Semantic Types", "Parents"]]
    df = df.loc[:, ["Class ID", "Preferred Label", "Synonyms", "Definitions", "Parents"]]

    # Replace NaN with None
    df = df.replace([np.nan], [None])

    # Transform the DataFrame to info Dictionary
    records = df.to_dict(orient="records")

    # Get Entity Dict (Class ID -> info)
    for record in tqdm(records):
        # str
        entity_id = record["Class ID"]

        # int
        entity_index = len(entity_dict)

        # List[str]
        canonical_names = [record["Preferred Label"]]

        # List[str]
        synonyms = record["Synonyms"]
        synonyms = synonyms.split("|") if synonyms is not None else []

        # str
        # If there are multiple definitions, we simply concatenate the texts with spaces.
        description = record["Definitions"]
        description = description if description is not None else ""
        description = description.replace("\n", " ").replace("|", " ")

        # str or None
        # entity_type = record["Semantic Types"]

        # List[str]
        kb_names = [kb_name]

        # List[str]
        parents = record["Parents"]
        parents = parents.split("|") if parents is not None else []

        if entity_id in entity_dict:
            # This scenario can appear when integrating multiple KBs,
            # because different KBs annotate different canonical names, synonyms, definitions, and parents.
            assert entity_dict[entity_id]["Class ID"] == entity_id
            # assert entity_dict[entity_id]["Semantic Types"] == entity_type

            # # We simple combine the canonical names and definitions of each KB with commas and spaces, respectively
            # if entity_dict[entity_id]["Preferred Label"] != canonical_name:
            #     canonical_name = ", ".join([entity_dict[entity_id]["Preferred Label"], canonical_name])

            # We concatenate definitions of each KB with commas and spaces, respectively
            description = " ".join([entity_dict[entity_id]["Definitions"], description]).strip()

            # Redundant canonical names, synonyms, parents, and KB names are filtered out later.
            canonical_names = entity_dict[entity_id]["Preferred Labels"] + canonical_names
            synonyms = entity_dict[entity_id]["Synonyms"] + synonyms
            parents = entity_dict[entity_id]["Parents"] + parents
            kb_names = entity_dict[entity_id]["KB Names"] + kb_names
            # Entity Index should be the same with the old record
            entity_index = entity_dict[entity_id]["Index"]

        entity_dict[entity_id] = {
            "Class ID": entity_id,
            "Index": entity_index,
            "Preferred Labels": canonical_names,
            "Synonyms": synonyms,
            "Definitions": description,
            # "Semantic Types": entity_type,
            "KB Names": kb_names,
            "Parents": parents,
        }
    return entity_dict


def remove_redundant_synonyms_and_parents(entity_dict):
    for key in entity_dict.keys():
        # Preferred Labels
        entity_dict[key]["Preferred Labels"] = sorted(list(set(entity_dict[key]["Preferred Labels"])))
        # Synonyms
        entity_dict[key]["Synonyms"] = sorted(list(set(entity_dict[key]["Synonyms"])))
        # Parents
        parents = [pid for pid in entity_dict[key]["Parents"] if pid != key]
        if len(parents) != len(entity_dict[key]["Parents"]):
            print(f"Removed a cyclic parent {key} for a child {key}")
            entity_dict[key]["Parents"] = parents
        # KB Names
        entity_dict[key]["KB Names"] = sorted(list(set(entity_dict[key]["KB Names"])))
    return entity_dict


# ---------------------------


def assign_tree_numbers(entity_dict, kb_names):
    # We add ROOT entities
    # We also set tree numbers of the ROOT entities
    # GO (ROOT) -> THING -> ...
    # HOIP (ROOT) -> THING -> ...
    ROOTS = kb_names
    for ROOT in ROOTS:
        entity_dict[ROOT] = {
            "Class ID": ROOT,
            "Index": ROOT,
            "KB Names": [ROOT],
            "Parents": [],
            "TreeNumbers": [ROOT],
        }
    THING = "http://www.w3.org/2002/07/owl#Thing"
    entity_dict[THING] = {
        "Class ID": THING,
        "Index": "THING",
        "KB Names": kb_names,
        "Parents": ROOTS,
    }

    # We also care about out-of-KB (OOK) entities
    # OOK entities can appear as the parents of in-KB entities
    ook_ids = []
    for cid in entity_dict.keys():
        for pid in entity_dict[cid]["Parents"]:
            if not pid in entity_dict:
                ook_ids.append(pid)
    print(f"{len(ook_ids)} entities are out of KB:")
    print(ook_ids)
    for i, ook_id in enumerate(ook_ids):
        entity_dict[ook_id] = {
            "Class ID": ook_id,
            "Index": -1,
            "KB Name": "OutOfKB",
            "Parents": [],
            "TreeNumbers": [
                f"OutOfKB{ook_id}"
            ]
        }

    # Step 0. Set tree numbers to the nodes without parents except for the ROOT and OOK entities
    for eid in entity_dict.keys():
        if eid in ROOTS + ook_ids:
            continue
        if len(entity_dict[eid]["Parents"]) == 0:
            assert len(entity_dict[eid]["KB Names"]) == 1
            entity_dict[eid]["TreeNumbers"] = [
                f"{entity_dict[eid]['KB Names'][0]}.{entity_dict[eid]['Index']}"
            ]

    # Step 1. Get parent->children edges
    edges = get_edges(entity_dict=entity_dict)
    print(f"# remaining nodes: {len(edges)}")

    # Step 2. Collect nodes without parents
    nodes_without_parents = get_terminal_nodes(edges=edges)
    print(f"# nodes without parents: {len(nodes_without_parents)}")
    while len(nodes_without_parents) != 0:
        # Step 3. For each node without parents, set the tree numbers of its children nodes
        for pid in nodes_without_parents:
            for cid in edges[pid]:
                entity_dict[cid] = set_tree_numbers(parent_entity=entity_dict[pid],
                                                    child_entity=entity_dict[cid])
        # Step 4. Remove the nodes without parents in `edges`
        for pid in nodes_without_parents:
            edges.pop(pid)
        print(f"# remaining nodes: {len(edges)}")
        # Step 2 (re). Re-collect nodes without parents
        nodes_without_parents = get_terminal_nodes(edges=edges)
        print(f"# nodes without parents: {len(nodes_without_parents)}")

    # Remove the ROOT and OOK entities
    for ROOT in ROOTS:
        entity_dict.pop(ROOT)
    entity_dict.pop(THING)
    for ook_id in ook_ids:
        entity_dict.pop(ook_id)

    return entity_dict


def get_edges(entity_dict):
    parent2children = {} # parent2children
    for eid in entity_dict.keys():
        parent2children[eid] = []
    for cid in entity_dict.keys():
        parents = entity_dict[cid]["Parents"]
        for pid in parents:
            parent2children[pid].append(cid)
    return parent2children


def get_terminal_nodes(edges):
    """Find terminal nodes (i.e., nodes without any in-coming edges) and nonterminal nodes (with in-coming edges)

    Parameters
    ----------
    edges : Dict[str, List[str]]

    Returns
    -------
    List[str], List[str]
    """
    # terminal_nodes = []
    # nonterminal_nodes = []
    # for key in tqdm(edges.keys()):
    #     pointed_by_others = False
    #     # Check whether `key` is pointed by any other node.
    #     # If `key` is NOT a dst key of any other node,
    #     # we treat the `key` as the terminal node
    #     for src_key in edges.keys():
    #         if key == src_key:
    #             continue
    #         dst_keys = edges[src_key]
    #         if key in dst_keys:
    #             pointed_by_others = True
    #             break
    #     if not pointed_by_others:
    #         terminal_nodes.append(key)
    #     else:
    #         nonterminal_nodes.append(key)
    # return terminal_nodes, nonterminal_nodes

    child2parents = {}
    # init
    for pid in edges.keys():
        child2parents[pid] = []
    # collect child -> parent edges
    for pid in edges.keys():
        for cid in edges[pid]:
            child2parents[cid].append(pid)
    nodes_without_parents = []
    for cid in child2parents.keys():
        if len(child2parents[cid]) == 0:
            nodes_without_parents.append(cid)
    return nodes_without_parents


def set_tree_numbers(parent_entity, child_entity):
    if not "TreeNumbers" in child_entity:
        child_entity["TreeNumbers"] = []
    child_kbs = child_entity["KB Names"]
    child_index = child_entity["Index"]
    for parent_tree_number in parent_entity["TreeNumbers"]:
        kb_name_of_this_path = parent_tree_number.split(".")[0]
        if kb_name_of_this_path in child_kbs:
            child_tree_number = f"{parent_tree_number}.{child_index}"
            child_entity["TreeNumbers"].append(child_tree_number)
    return child_entity


# def set_tree_numbers(entity_dict, nodes_without_parents):
#     for cid in nodes_without_parents:
#         entity_index = entity_dict[cid]["Index"]
#         # Collect parents' tree numbers
#         tree_numbers = []
#         parents = entity_dict[cid]["Parents"]
#         for pid in parents:
#             nums = entity_dict[pid]["TreeNumbers"]
#             tree_numbers.extend(nums)
#         tree_numbers = set(tree_numbers)
#         tree_numbers = sorted(list(tree_numbers))
#         # Append the index to the tree numbers and set them as `TreeNumbers`
#         if len(tree_numbers) == 0:
#             tree_numbers = [f"{entity_index}"]
#         else:
#             tree_numbers = [f"{num}.{entity_index}" for num in tree_numbers]
#         entity_dict[cid]["TreeNumbers"] = tree_numbers
#     return entity_dict


def remove_redundant_tree_numbers(entity_dict):
    for key in entity_dict.keys():
        assert len(entity_dict[key]["TreeNumbers"]) > 0
        entity_dict[key]["TreeNumbers"] = sorted(list(set(entity_dict[key]["TreeNumbers"])))
    return entity_dict


# ---------------------------


def read_csv(path, delimiter, with_head, with_id, encoding="utf-8"):
    header = 0 if with_head else None
    index_col = 0 if with_id else None
    data = pd.read_csv(path, encoding=encoding, delimiter=delimiter, header=header, index_col=index_col)
    return data


def write_json(path, dct):
    with open(path, "w") as f:
        # json.dump(dct, f, indent=4)
        json.dump(dct, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", nargs="+", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    main(args)

