import logging

from neo4j import GraphDatabase
# from neo4j import basic_auth
from tqdm import tqdm


logger = logging.getLogger(__name__)


class Neo4jConnection:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def run_query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]

    def remove_all(self):
        query = "MATCH (n) DETACH DELETE n"
        self.run_query(query=query, parameters=None)

    def get_entities(self):
        query = "MATCH (n) RETURN n"
        return self.run_query(query=query, parameters=None)
    
    def add_entity(
        self,
        entity_id,
        entity_type,
        name,
        description,
        doc_key_list,
        community_id,
        community_level
    ):
        query = f"MERGE (n:{entity_type} {{ entity_id: $entity_id, name: $name, description: $description, doc_key_list: $doc_key_list, community_id: $community_id, community_level: $community_level }}) RETURN n"
        self.run_query(
            query=query,
            parameters={
                "entity_id": entity_id,
                "name": name,
                "description": description,
                "doc_key_list": doc_key_list,
                "community_id": community_id,
                "community_level": community_level
            }
        )

    def add_relation(self, head_id, tail_id, relation, doc_key_list):
        query = f"""
        MATCH (e1 {{ entity_id: $head_id }})
        MATCH (e2 {{ entity_id: $tail_id }})
        MERGE (e1)-[r:{relation} {{ doc_key_list: $doc_key_list }}]->(e2)
        RETURN e1,e2,r
        """
        #print(query)
        self.run_query(
            query=query,
            parameters={
                "head_id": head_id,
                "tail_id": tail_id,
                "doc_key_list": doc_key_list
            }
        )


def export_nx_to_neo4j(graph, uri, user, password):
    conn = Neo4jConnection(uri=uri, user=user, password=password)

    for node, prop in tqdm(graph.nodes(data=True)):
        entity_id = prop["entity_id"]
        entity_type = prop["entity_type"]
        name = prop["name"]
        desc = prop["description"]
        doc_key_list = prop["doc_key_list"]
        community_id = prop["community_id"]
        community_level = prop["community_level"]
        conn.add_entity(
            entity_id=entity_id,
            entity_type=entity_type,
            name=name,
            description=desc,
            doc_key_list=doc_key_list,
            community_id=community_id,
            community_level=community_level
        )

    for h, t, prop in tqdm(graph.edges(data=True)):
        h_id = graph.nodes[h]["entity_id"]
        t_id = graph.nodes[t]["entity_id"]
        relation = prop["relation"]
        doc_key_list = prop["doc_key_list"]
        conn.add_relation(
            head_id=h_id,
            tail_id=t_id,
            relation=relation,
            doc_key_list=doc_key_list
        )

    conn.close()

 