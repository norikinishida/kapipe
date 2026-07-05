import logging
import os

import torch
import transformers

from kapipe import utils
from kapipe.pipelines import GraphRAGPipeline
from kapipe.llms import OpenAILLM
from kapipe.ner import LLMNER
from kapipe.ed_retrieval import BlinkBiEncoder
from kapipe.ed_reranking import LLMED
from kapipe.docre import LLMDocRE
from kapipe.entity_graph_construction import EntityGraphConstructor
from kapipe.community_clustering import NeighborhoodAggregation
from kapipe.report_generation import TemplateBasedReportGenerator
from kapipe.chunking import Chunker
from kapipe.passage_retrieval import Qwen3Embedding
from kapipe.qa import LLMQA


def main():
    torch.autograd.set_detect_anomaly(True)
    transformers.logging.set_verbosity_error()

    # Set input and output paths
    data_dir = "..//graphrag_pipeline_tacl2026/data/examples"
    index_dir = "./quickstart/indexes"

    # Instantiate the components
    llm = OpenAILLM(model_name="gpt-5.4-nano", max_new_tokens=8192)
    ner = LLMNER.from_identifier(model=llm, identifier="llm_ner_cdr")
    ed_retrieval = BlinkBiEncoder.from_identifier(identifier="blink_bi_encoder_cdr")
    ed_retrieval.make_index(use_precomputed_entity_vectors=True)
    ed_reranking = LLMED.from_identifier(model=llm, identifier="llm_ed_cdr")
    docre = LLMDocRE.from_identifier(model=llm, identifier="llm_docre_cdr")
    entity_graph_construction = EntityGraphConstructor()
    community_clustering = NeighborhoodAggregation(hop_size=1)
    report_generation = TemplateBasedReportGenerator()
    chunker = Chunker(model_name="en_core_sci_md")
    passage_retrieval = Qwen3Embedding(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        max_passage_length=8192,
        normalize=True,
        metric="inner-product",
        query_instruction="Given a question, retrieve relevant passages that answer the question."
    )
    qa = LLMQA(
        model=llm,
        prompt_template_name_or_path="qa_03_with_context",
    )

    # Instantiate the GraphRAG pipeline
    graphrag = GraphRAGPipeline(
        ner=ner,
        ed_retrieval=ed_retrieval,
        ed_reranking=ed_reranking,
        docre=docre,
        entity_graph_construction=entity_graph_construction,
        community_clustering=community_clustering,
        report_generation=report_generation,
        chunker=chunker,
        passage_retrieval=passage_retrieval,
        qa=qa,
    )

    # Step 1. Extract triples from documents
    documents = utils.read_json(os.path.join(data_dir, "documents.json"))
    graphrag.extract_triples(
        documents=documents,
        retrieval_size=10,
        index_dir=index_dir,
    )

    # Step 2. Construct an entity graph from triples
    graph = graphrag.construct_entity_graph(
        documents_path_list=[os.path.join(index_dir, "documents_with_triples.json")],
        entity_dict_path=os.path.join(data_dir, "entity_dict.json"),
        additional_triples_path=None,
        index_dir=index_dir,
    )

    # Step 3. Cluster the graph into communities (subgraphs)
    communities = graphrag.cluster_communities(
        graph=graph,
        index_dir=index_dir,
    )

    # Step 4. Generate reports for each community
    reports = graphrag.generate_community_reports(
        graph=graph,
        communities=communities,
        index_dir=index_dir,
    )

    # Step 5. Chunk community reports into chunks
    chunked_reports = graphrag.chunk_reports(
        reports=reports,
        window_size=100,
        index_dir=index_dir,
    )

    # Step 6. Build a retrieval index over the chunked reports
    graphrag.make_passage_retrieval_index(
        chunked_reports=chunked_reports,
        batch_size=64,
        index_dir=index_dir,
    )

    # Step 7. Load the retrieval index and answer questions
    questions = utils.read_json(os.path.join(data_dir, "questions.json"))
    answers = [
        graphrag.infer(question=question, top_k=5)
        for question in questions
    ]

    # Save the results
    utils.write_json("./predictions.json", answers)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO
    )
    logging.getLogger("httpx").addFilter(
        lambda r: "huggingface.co" not in r.getMessage()
    )

    main()