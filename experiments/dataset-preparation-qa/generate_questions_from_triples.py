import argparse
# import json
import os
import random

import numpy as np
import networkx as nx
import pandas as pd

import sys
sys.path.insert(0, "../../..")
from kapipe import utils


# RELATION_MAP = {
#     "CID": "induces"
# }


# PROMPT_TEMPLATE_NEIGHBORHOOD = """
# Generate a single question that asks about what {node_name} {relation}.
# Return output as a well-formed JSON-formatted string with the following format:
#     {{
#         "question": <question>,
#     }}

# Entity: {node_name}
# Relation: {relation}
# Answer: {neighbor_names}

# Output:"""


def main(args):
    question_type = args.question_type
    path_input_documents_list = args.input_documents
    # path_triples = args.triples
    path_entity_dict = args.entity_dict
    path_output_questions = args.output_questions
    # config_path = args.config_path
    # config_name = args.config_name
    n_questions = args.n_questions

    utils.mkdir(os.path.dirname(path_output_questions))

    # まずはEntity Dictionaryの読み込み
    entity_dict = utils.read_json(path_entity_dict)
    entity_dict = {e["entity_id"]: e for e in entity_dict}
    print(f"Loaded {len(entity_dict)} entity pages")

    # KB triplesを読み込む
    # kb_triples = utils.read_json(path_triples)

    # triplesを読み込む
    # ついでに、Entity ID -> namesの辞書を作成しておく
    triples = []
    entity_id_to_names = {}
    n_ignored = 0
    for path_input_documents in path_input_documents_list:
        documents = utils.read_json(path_input_documents)
        for doc in documents:
            entities = doc["entities"]
            mentions = doc["mentions"]
            for triple in doc["relations"]:
                head_idx = triple["arg1"]
                tail_idx = triple["arg2"]
                relation = triple["relation"]

                head_id = entities[head_idx]["entity_id"]
                tail_id = entities[tail_idx]["entity_id"]

                if head_id.startswith("#ignored#"):
                    head_id = f"#ignored#-{n_ignored}"
                    n_ignored += 1
                if tail_id.startswith("#ignored#"):
                    tail_id = f"#ignored#-{n_ignored}"
                    n_ignored += 1

                head_names = get_entity_names(entity=entities[head_idx], entity_dict=entity_dict, mentions=mentions)
                tail_names = get_entity_names(entity=entities[tail_idx], entity_dict=entity_dict, mentions=mentions)
                # relation = RELATION_MAP[relation]

                triples.append({
                    "head_id": head_id,
                    "head_names": head_names,
                    "relation": relation,
                    "tail_id": tail_id,
                    "tail_names": tail_names
                })

                if not head_id in entity_id_to_names:
                    entity_id_to_names[head_id] = []
                if not tail_id in entity_id_to_names:
                    entity_id_to_names[tail_id] = []
                entity_id_to_names[head_id].extend(head_names)
                entity_id_to_names[tail_id].extend(tail_names)

    for entity_id, names in entity_id_to_names.items():
        entity_id_to_names[entity_id] = list(set(names))
        assert len(list(set(names))) > 0

    print(f"Loaded {len(triples)} triples")

    graph = build_graph(triples=triples)

    # # Prepare LLM
    # config = utils.get_hocon_config(
    #     config_path=config_path,
    #     config_name=config_name
    # )
    # model = OpenAILLM(
    #     # Model
    #     openai_model_name=config["openai_model_name"],
    #     # Generation
    #     max_new_tokens=config["max_new_tokens"]
    # )

    # templatesの読み込み
    df = pd.read_csv("../dataset-meta-information/question_templates_v2.csv")
    question_type_to_relation_to_template = df.groupby("Question Type").apply(lambda g: dict(zip(g["Relation"], g["Template"]))).to_dict()
    relation_to_template = question_type_to_relation_to_template[question_type]

    # questionsの生成、統合
    if question_type == "neighborhood":
        questions = generate_neighborhood_questions(relation_to_template=relation_to_template, graph=graph, entity_dict=entity_dict, n_questions=n_questions, entity_id_to_names=entity_id_to_names)
    elif question_type == "intersection":
        questions = generate_intersection_questions(relation_to_template=relation_to_template, graph=graph, entity_dict=entity_dict, n_questions=n_questions, entity_id_to_names=entity_id_to_names)
    elif question_type == "twohop":
        questions = generate_twohop_questions(relation_to_template=relation_to_template, graph=graph, entity_dict=entity_dict, n_questions=n_questions, entity_id_to_names=entity_id_to_names)
    else:
        raise Exception(f"Invalid question type provided: {question_type}")
    print(f"Generated {len(questions)} questions")

    # 保存
    utils.write_json(path_output_questions, questions)
    print(f"Saved the generated questions in {path_output_questions}")

    print("Done.")


def get_entity_names(entity, entity_dict, mentions=None):
    epage = entity_dict.get(entity["entity_id"], None)
    if epage is not None:
        names = [epage["canonical_name"]] + epage["synonyms"]
    else:
        names = []
    if mentions is not None:
        for m_i in entity["mention_indices"]:
            names.append(mentions[m_i]["name"])
    return list(set(names))

    
def build_graph(triples):
    graph = nx.DiGraph()
    for triple in triples:
        head_id = triple["head_id"]
        tail_id = triple["tail_id"]
        relation = triple["relation"]
        graph.add_edge(head_id, tail_id, relation=relation)
    return graph


####################


def get_neighbors(graph, entity_id):
    relation_to_neighbors = {}
    if entity_id in graph:
        outgoing_nodes = list(graph.neighbors(entity_id))
        outgoing_relations = [graph[entity_id][n]["relation"] for n in outgoing_nodes]
        incoming_nodes = list(graph.predecessors(entity_id))
        # incoming_relations = [graph[n][entity_id]["relation"].replace("induces", "induced_by") for n in incoming_nodes]
        incoming_relations = [graph[n][entity_id]["relation"] + "#inverse" for n in incoming_nodes]
        nodes = outgoing_nodes + incoming_nodes
        relations = outgoing_relations + incoming_relations
        for node, rel in zip(nodes, relations):
            if node == entity_id:
                continue
            if not rel in relation_to_neighbors:
                relation_to_neighbors[rel] = []
            relation_to_neighbors[rel].append(node)
    return relation_to_neighbors


def get_2hop_neighbors(graph, entity_id):
    results = set()
    one_hop_neighbors = get_neighbors(graph, entity_id)
    for _, nodes in one_hop_neighbors.items():
        for node in nodes:
            two_hop_neighbors = get_neighbors(graph, node)
            for _, nodes2 in two_hop_neighbors.items():
                for node2 in nodes2:
                    if node2 == entity_id:
                        continue
                    results.add(node2)
    return list(results)


def generate_neighborhood_questions(relation_to_template, graph, entity_dict, n_questions, entity_id_to_names):
    questions = []

    nodes = list(graph.nodes)
    node_indices = np.random.permutation(len(nodes))

    for node_i in node_indices:
        # 対象ノードをサンプリング
        target_node = nodes[node_i]

        # 対象ノードの隣接ノードを関係ごとにリストアップ
        relation_to_neighbors = get_neighbors(graph=graph, entity_id=target_node)

        # 関係と隣接ノードを決定
        candidate_relations = [r for r, ns in relation_to_neighbors.items() if len(ns) >= 2]
        if len(candidate_relations) == 0:
            continue
        target_relation = random.sample(candidate_relations, 1)[0]
        target_neighbors = relation_to_neighbors[target_relation]

        # 質問の作成
        if target_node in entity_dict:
            target_node_name = random.sample(get_entity_names(entity={"entity_id": target_node}, entity_dict=entity_dict), 1)[0]
        else:
            target_node_name = random.sample(entity_id_to_names[target_node], 1)[0]
        question_str = relation_to_template[target_relation].format(
            target=target_node_name
        )

        # 解答の作成
        answers = []
        for list_i, target_neighbor in enumerate(target_neighbors):
            if target_neighbor in entity_dict:
                for target_neighbor_name in get_entity_names(entity={"entity_id": target_neighbor}, entity_dict=entity_dict):
                    answers.append({
                        "answer": target_neighbor_name,
                        "answer_type": "list",
                        "list_index": list_i
                    })
            else:
                for target_neighbor_name in entity_id_to_names[target_neighbor]:
                    answers.append({
                        "answer": target_neighbor_name,
                        "answer_type": "list",
                        "list_index": list_i
                    })

        # 追加
        questions.append({
            "question_key": f"neighborhood#{len(questions)+1}",
            "question": question_str,
            "answers": answers,
            "source": {
                "node": target_node,
                "relation": target_relation,
                "neighbors": target_neighbors,
            }
        })

        # 終了判定
        if len(questions) >= n_questions:
            break

    print(f"Generated {len(questions)} neighborhood questions")
    return questions

    
# def generate_question_by_llm()
#     # ノードを名前を決定
#     node_name = random.sample(get_entity_names(entity={"entity_id": node}, entity_dict=entity_dict), 1)[0]
#     neighbor_names = [random.sample(get_entity_names(entity={"entity_id": n}, entity_dict=entity_dict), 1)[0] for n in neighbor_nodes]
#     # プロンプト作成
#     prompt = PROMPT_TEMPLATE_NEIGHBORHOOD.format(
#         node_name=node_name,
#         relation=relations[0],
#         neighbor_names=neighbor_names
#     )
#     # 質問生成
#     generated_text = model.generate(prompt)
#     # 生成結果のパース
#     begin_index = generated_text.find("{")
#     end_index = generated_text.rfind("}")
#     if begin_index < 0 or end_index < 0:
#         print(f"Failed to parse the generated report into a JSON object: '{generated_text}'")
#         continue
#     json_text = generated_text[begin_index: end_index + 1]
#     try:
#         json_obj = json.loads(json_text)
#     except Exception as e:
#         print(f"Failed to parse the generated report into a JSON object: '{json_text}'")
#         print(e)
#         continue
#     if not isinstance(json_obj, dict):
#         print(f"The parsed JSON object is not a dictionary: '{json_obj}'")
#         continue
#     try:
#         question_str = json_obj["question"]
#     except Exception as e:
#         print(f"Failed to textualize a parsed JSON object: '{json_obj}'")
#         print(e)
#         continue


def generate_intersection_questions(relation_to_template, graph, entity_dict, n_questions, entity_id_to_names):
    MIN_COMMON_NEIGHBORS = 2
    # MIN_COMMON_NEIGHBORS = 1

    questions = []

    nodes = list(graph.nodes)
    node_indices = np.random.permutation(len(nodes))
    memo = set()

    # while True:
    #     # 対象ノードペアをサンプリング
    #     target_node_a, target_node_b = random.sample(nodes, 2)
    #     if target_node_a == target_node_b:
    #         continue

    for node_i in node_indices:
        # 対象ノードをサンプリング
        target_node_a = nodes[node_i]

        # 2ホップ先のノードをすべて収集
        target_nodes_b = get_2hop_neighbors(graph, target_node_a)
        if len(target_nodes_b) == 0:
            continue

        for target_node_b in np.random.permutation(target_nodes_b):
            if target_node_a == target_node_b:
                continue

            # それぞれの対象ノードの隣接ノードを関係ごとにリストアップ
            relation_to_neighbors_a = get_neighbors(graph=graph, entity_id=target_node_a)
            relation_to_neighbors_b = get_neighbors(graph=graph, entity_id=target_node_b)

            # 関係と隣接ノードを決定
            # まずは、target node A、Bそれぞれが隣接ノードと持つ関係性を列挙
            candidate_relations_a = [r for r, ns in relation_to_neighbors_a.items() if len(ns) >= MIN_COMMON_NEIGHBORS]
            candidate_relations_b = [r for r, ns in relation_to_neighbors_b.items() if len(ns) >= MIN_COMMON_NEIGHBORS]
            # 共通関係に絞る
            candidate_relations = list(set(candidate_relations_a) & set(candidate_relations_b))
            candidate_relations = [r for r in candidate_relations if not (target_node_a, target_node_b, r) in memo]
            if len(candidate_relations) == 0:
                continue
            # 各関係ごとにtarget node A、Bの隣接ノードを列挙し、共通があれば候補にする
            candidate_relations2 = []
            for rel in candidate_relations:
                neighbors_a = relation_to_neighbors_a[rel]
                neighbors_b = relation_to_neighbors_b[rel]
                common_neighbors = list(set(neighbors_a) & set(neighbors_b))
                if len(common_neighbors) >= MIN_COMMON_NEIGHBORS and len(neighbors_a) > len(common_neighbors) and len(neighbors_b) > len(common_neighbors):
                    candidate_relations2.append(rel)
            if len(candidate_relations2) == 0:
                continue
            # 候補から一つ選択
            target_relation = random.sample(candidate_relations2, 1)[0]
            target_neighbors_a = relation_to_neighbors_a[target_relation]
            target_neighbors_b = relation_to_neighbors_b[target_relation]
            common_neighbors = list(set(target_neighbors_a) & set(target_neighbors_b))
            assert len(common_neighbors) >= MIN_COMMON_NEIGHBORS

            memo.add((target_node_a, target_node_b, target_relation))

            # 質問の生成
            if target_node_a in entity_dict:
                target_node_name_a = random.sample(get_entity_names(entity={"entity_id": target_node_a}, entity_dict=entity_dict), 1)[0]
            else:
                target_node_name_a = random.sample(entity_id_to_names[target_node_a], 1)[0]
            if target_node_b in entity_dict:
                target_node_name_b = random.sample(get_entity_names(entity={"entity_id": target_node_b}, entity_dict=entity_dict), 1)[0]
            else:
                target_node_name_b = random.sample(entity_id_to_names[target_node_b], 1)[0]
            question_str = relation_to_template[target_relation].format(
                target_a=target_node_name_a,
                target_b=target_node_name_b
            )

            # 解答の作成
            answers = []
            for list_i, common_neighbor in enumerate(common_neighbors):
                if common_neighbor in entity_dict:
                    for common_neighbor_name in get_entity_names(entity={"entity_id": common_neighbor}, entity_dict=entity_dict):
                        answers.append({
                            "answer": common_neighbor_name,
                            "answer_type": "list",
                            "list_index": list_i
                        })
                else:
                    for common_neighbor_name in entity_id_to_names[common_neighbor]: 
                        answers.append({
                            "answer": common_neighbor_name,
                            "answer_type": "list",
                            "list_index": list_i
                        })

            # 追加
            questions.append({
                "question_key": f"intersection#{len(questions)+1}",
                "question": question_str,
                "answers": answers,
                "source": {
                    "node_a": target_node_a,
                    "node_b": target_node_b,
                    "relation": target_relation,
                    "common_neighbors": common_neighbors,
                }
            })

            break

        # 終了判定
        if len(questions) >= n_questions:
            break

    print(f"Generated {len(questions)} intersection questions")
    return questions


def generate_twohop_questions(relation_to_template, graph, entity_dict, n_questions, entity_id_to_names):
    questions = []

    nodes = list(graph.nodes)
    node_indices = np.random.permutation(len(nodes))

    for node_i in node_indices:
        # 対象ノードをサンプリング
        target_node = nodes[node_i]

        # (1ホップ目) 対象ノードの隣接ノードを関係ごとにリストアップ
        # target_node -> {r} -> neighbors_1
        relation_to_neighbors_1 = get_neighbors(graph=graph, entity_id=target_node)

        # (1ホップ目) 関係と隣接ノードを決定
        target_relation_1 = random.sample(list(relation_to_neighbors_1.keys()), 1)[0]
        target_neighbors_1 = relation_to_neighbors_1[target_relation_1]

        # (2ホップ目) 隣接ノードの隣接ノードを関係ごとにリストアップ
        # target_node -> {r1} -> neighbors_1 -> {r2} -> neighbors_2
        relation_to_neighbors_2 = {}
        for target_neighbor_1 in target_neighbors_1:
            temp_relation_to_neighbors = get_neighbors(graph=graph, entity_id=target_neighbor_1)
            for r, ns in temp_relation_to_neighbors.items():
                if not r in relation_to_neighbors_2:
                    relation_to_neighbors_2[r] = []
                relation_to_neighbors_2[r].extend(ns)
        for r, ns in relation_to_neighbors_2.items():
            # ns = [n for n in ns if n != target_node]
            relation_to_neighbors_2[r] = list(set(ns))
                
        # (2ホップ目) 関係と隣接ノードを決定
        candidate_relations = [r for r, ns in relation_to_neighbors_2.items() if len(ns) >= 2 and target_relation_1.replace("#inverse", "") != r.replace("#inverse", "")]
        if len(candidate_relations) == 0:
            continue
        target_relation_2 = random.sample(candidate_relations, 1)[0]
        target_neighbors_2 = relation_to_neighbors_2[target_relation_2]

        # 質問の生成        
        if target_node in entity_dict:
            target_node_name = random.sample(get_entity_names(entity={"entity_id": target_node}, entity_dict=entity_dict), 1)[0]
        else:
            target_node_name = random.sample(entity_id_to_names[target_node], 1)[0]
        question_str = relation_to_template[f"{target_relation_1}-{target_relation_2}"].format(
            target=target_node_name
        )

        # 解答の作成
        answers = []
        for list_i, target_neighbor_2 in enumerate(target_neighbors_2):
            if target_neighbor_2 in entity_dict:
                for target_neighbor_name_2 in get_entity_names(entity={"entity_id": target_neighbor_2}, entity_dict=entity_dict):
                    answers.append({
                        "answer": target_neighbor_name_2,
                        "answer_type": "list",
                        "list_index": list_i
                    })
            else: 
                for target_neighbor_name_2 in entity_id_to_names[target_neighbor_2]:
                    answers.append({
                        "answer": target_neighbor_name_2,
                        "answer_type": "list",
                        "list_index": list_i
                    })

        # 追加
        questions.append({
            "question_key": f"twohop#{len(questions)+1}",
            "question": question_str,
            "answers": answers,
            "source": {
                "node": target_node,
                "relation-hop1": target_relation_1,
                "neighbors-hop1": target_neighbors_1,
                "relation-hop2": target_relation_2,
                "neighbors-hop2":target_neighbors_2 
            }
        })

        # 終了判定
        if len(questions) >= n_questions:
            break

    print(f"Generated {len(questions)} indirect questions")
    return questions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question_type", type=str, required=True)
    parser.add_argument("--input_documents", nargs="+")
    # parser.add_argument("--triples", type=str, required=True)
    parser.add_argument("--entity_dict", type=str, required=True)
    parser.add_argument("--output_questions", type=str, required=True)
    # parser.add_argument("--config_path", type=str, required=True)
    # parser.add_argument("--config_name", type=str, required=True)
    parser.add_argument("--n_questions", type=int, default=256)
    args = parser.parse_args()
    main(args)


