import argparse
import logging
import os
import sys
from typing import Any

import torch
import transformers

from kapipe import evaluation
from kapipe import utils
from kapipe.utils import StopWatch

from kapipe.pipelines import GraphRAGPipeline

from kapipe.llms import BaseLLM, HuggingFaceLLM, OpenAILLM
from kapipe.ner import (
    BaseNER,
    BiaffineNER,
    LLMNER,
)
from kapipe.ed_retrieval import (
    BaseEDRetriever,
    MentionNameEntityRetriever,
    BlinkBiEncoder,
)
from kapipe.ed_reranking import (
    BaseEDReranker,
    IdenticalEntityReranker,
    BlinkCrossEncoder,
    LLMED,
)
from kapipe.docre import (
    BaseDocRE,
    ATLOP,
    LLMDocRE,
)
from kapipe.entity_graph_construction import (
    BaseEntityGraphConstructor,
    EntityGraphConstructor,
)
from kapipe.community_clustering import (
    BaseCommunityClusterer,
    HierarchicalLeiden,
    NeighborhoodAggregation,
    TripleLevelFactorization,
)
from kapipe.report_generation import (
    LLMBasedReportGenerator,
    TemplateBasedReportGenerator,
)
from kapipe.report_generation.base import BaseReportGenerator
from kapipe.chunking import (
    BaseChunker,
    Chunker,
)
from kapipe.passage_retrieval import (
    BasePassageRetriever,
    BM25,
    Contriever,
    Qwen3Embedding,
)
from kapipe.qa import (
    BaseQA,
    LLMQA,
)


def main(args: argparse.Namespace) -> None:
    torch.autograd.set_detect_anomaly(True)
    transformers.logging.set_verbosity_error()

    sw = StopWatch()
    sw.start("main")

    ##################
    # Arguments
    ##################

    # Method
    method_name = args.method
    config_path = args.config_path
    config_name = args.config_name

    # Input Data
    input_documents_path = args.input_documents
    entity_dict_path = args.entity_dict
    additional_triples_path = args.additional_triples
    input_questions_path = args.input_questions

    # Output Path
    results_dir = args.results_dir
    prefix = args.prefix
    if prefix is None or prefix == "None":
        prefix = utils.get_current_time()
        args.prefix = prefix

    # Action
    actiontype = args.actiontype

    # Evaluation
    do_evaluation = args.do_evaluation
    gold_questions_path = args.gold

    ##################
    # Logging Setup
    ##################

    # Set base output path
    base_output_path = os.path.join(
        results_dir,
        "graphrag_pipeline",
        method_name,
        config_name,
        prefix
    )
    utils.mkdir(base_output_path)

    # Set the base filename for query processing
    base_filename: str | None = None
    if actiontype == "inference":
        if input_questions_path is None:
            raise ValueError(
                f"--input_questions is required for {actiontype}"
            )
        base_filename = os.path.splitext(
            os.path.basename(input_questions_path)
        )[0]

    # Index will be saved to `index_dir`
    index_dir = os.path.join(base_output_path, "indexes")
    utils.mkdir(index_dir)

    # Set logger
    if actiontype != "inference":
        set_logger(
            os.path.join(base_output_path, f"{actiontype}.log"),
            # overwrite=True
        )
    else:
        set_logger(
            os.path.join(base_output_path, f"{base_filename}.inference.log"),
            # overwrite=True
        )

    # Show arguments
    logging.info(utils.pretty_format_dict(vars(args)))
    logging.info(f"index dir: {index_dir}")

    ##################
    # Data
    ##################

    # Data are loaded in the corresponding actiontype section below

    ##################
    # Method Instantiation
    ##################

    # Load the experiment configuration
    config = utils.get_hocon_config(
        config_path=config_path,
        config_name=config_name
    )

    # Save the experiment configuration to the output path
    utils.write_json(os.path.join(base_output_path, "config.json"), config)

    # Initialize the loaded LLM map
    loaded_llm_map: dict[str, BaseLLM] = {}

    # Instantiate the NER component
    ner: BaseNER | None = None
    if actiontype == "triple_extraction":
        ner, loaded_llm_map = instantiate_ner_component(
            ner_config=config["ner"],
            loaded_llm_map=loaded_llm_map,
        )

    # Instantiate the ED-Retrieval component
    ed_retrieval: BaseEDRetriever | None = None
    if actiontype == "triple_extraction":
        ed_retrieval, loaded_llm_map = instantiate_ed_retrieval_component(
            ed_retrieval_config=config["ed_retrieval"],
            loaded_llm_map=loaded_llm_map,
        )

    # Instantiate the ED-Reranking component
    ed_reranking: BaseEDReranker | None = None
    if actiontype == "triple_extraction":
        ed_reranking, loaded_llm_map = instantiate_ed_reranking_component(
            ed_reranking_config=config["ed_reranking"],
            loaded_llm_map=loaded_llm_map,
        )

    # Instantiate the DocRE component
    docre: BaseDocRE | None = None
    if actiontype == "triple_extraction":
        docre, loaded_llm_map = instantiate_docre_component(
            docre_config=config["docre"],
            loaded_llm_map=loaded_llm_map,
        )

    # Instantiate the Entity Graph Construction component
    entity_graph_construction: BaseEntityGraphConstructor | None = None
    if actiontype == "entity_graph_construction":
        entity_graph_construction = instantiate_entity_graph_construction_component(
            entity_graph_construction_config=config["entity_graph_construction"],
        )

    # Instantiate the Community Clustering component
    community_clustering: BaseCommunityClusterer | None = None
    if actiontype == "community_clustering":
        community_clustering = instantiate_community_clustering_component(
            community_clustering_config=config["community_clustering"],
        )

    # Instantiate the Report Generation component
    report_generation: BaseReportGenerator | None = None
    if actiontype == "report_generation":
        report_generation, loaded_llm_map = instantiate_report_generation_component(
            report_generation_config=config["report_generation"],
            loaded_llm_map=loaded_llm_map,
        )

    # Instantiate the Chunking component
    chunker: BaseChunker | None = None
    if actiontype == "chunking":
        chunker = instantiate_chunking_component(
            chunking_config=config["chunking"],
        )

    # Instantiate the Passage Retrieval component
    passage_retrieval: BasePassageRetriever | None = None
    if actiontype in ["passage_retrieval_indexing", "inference"]:
        passage_retrieval = instantiate_passage_retrieval_component(
            passage_retrieval_config=config["passage_retrieval"],
        )

    # Instantiate the QA component
    qa: BaseQA | None = None
    if actiontype == "inference":
        qa, loaded_llm_map = instantiate_qa_component(
            qa_config=config["qa"],
            loaded_llm_map=loaded_llm_map,
        )

    # Instantiate the GraphRAG pipeline
    graphrag: GraphRAGPipeline = GraphRAGPipeline(
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

    ##################
    # Method Execution
    ##################

    if actiontype != "inference":
        # Load documents only when Triple Extraction is selected
        if actiontype == "triple_extraction":
            if input_documents_path is None:
                raise ValueError(
                    "--input_documents is required for triple_extraction"
                )
            documents: list[dict[str, Any]] | None = utils.read_json(
                input_documents_path
            )
        else:
            documents = None

        # Set component-specific arguments
        if config["passage_retrieval"]["method_name"] == "bm25":
            passage_retrieval_indexing_kwargs: dict[str, Any] = {}
        else:
            passage_retrieval_indexing_kwargs = {
                "batch_size": config["passage_retrieval"][
                    "indexing_batch_size"
                ],
            }

        # Run the selected GraphRAG indexing component
        graphrag.make_index(
            # Input
            documents=documents,
            # Output directory
            index_dir=index_dir,
            # Component-specific arguments
            retrieval_size=config["ed_retrieval"]["retrieval_size"],
            window_size=config["chunking"]["window_size"],
            entity_dict_path=entity_dict_path,
            additional_triples_path=additional_triples_path,
            node_attr_keys=tuple(
                config["report_generation"]["node_attr_keys"]
            ),
            edge_attr_keys=tuple(
                config["report_generation"]["edge_attr_keys"]
            ),
            passage_retrieval_indexing_kwargs=passage_retrieval_indexing_kwargs,
            # Target component for indexing
            target_component=actiontype,
        )

    else:
        # Validate that the input questions path is provided
        if input_questions_path is None:
            raise ValueError("--input_questions is required for inference")

        # Load questions
        questions: list[dict[str, Any]] = utils.read_json(input_questions_path)

        logging.info(
            f"Applying the GraphRAG pipeline to {len(questions)} "
            f"questions in {input_questions_path} ..."
        )

        # Load the built index
        graphrag.load_index(index_dir=index_dir)

        # Run all inference components for every question
        result_questions: list[dict[str, Any]] = graphrag.infer(
            questions=questions,
            top_k=config["passage_retrieval"]["top_k"],
        )

        # Save the results
        output_questions_path = os.path.join(
            base_output_path,
            f"{base_filename}.pred.json",
        )
        utils.write_json(output_questions_path, result_questions)
        logging.info(f"Saved the prediction results to {output_questions_path}")

        # Save the prompts, raw responses, parsed answers, and optional gold answers
        output_prompt_and_responses_path: str = os.path.join(
            base_output_path,
            f"{base_filename}.prompt_and_responses.txt",
        )
        with open(
            output_prompt_and_responses_path,
            "w",
            encoding="utf-8",
        ) as fout:
            # Write one human-readable block for each question
            for result_question in result_questions:
                fout.write("=" * 80 + "\n\n")

                fout.write("QUESTION KEY:\n")
                fout.write(result_question["question_key"] + "\n\n")

                fout.write("PROMPT:\n")
                fout.write(result_question["qa_prompt"].rstrip() + "\n\n")

                fout.write("GENERATED TEXT:\n")
                fout.write(
                    result_question["qa_generated_text"].rstrip() + "\n\n"
                )

                fout.write("PARSED ANSWER:\n")
                fout.write(result_question["output_answer"].rstrip() + "\n\n")

                # Write gold answers only when they are included in the input question
                if "answers" in result_question:
                    fout.write("GOLD ANSWERS:\n")
                    for answer in result_question["answers"]:
                        fout.write(f"- {answer['answer']}\n")
                    fout.write("\n")

        logging.info(
            "Saved the prompts and responses to "
            f"{output_prompt_and_responses_path}"
        )

        ##################
        # Evaluation
        ##################

        if do_evaluation:
            # Validate that the gold questions path is provided
            if gold_questions_path is None:
                raise ValueError("--gold is required when --do_evaluation is set")

            # Evaluate the prediction results
            qa_scores = evaluation.qa.accuracy(
                pred_path=output_questions_path,
                gold_path=gold_questions_path,
                exact_match=False,
            )
            qa_scores.update(
                evaluation.qa.token_level_f1(
                    pred_path=output_questions_path,
                    gold_path=gold_questions_path,
                )
            )
            qa_scores.update(
                evaluation.qa.recall(
                    pred_path=output_questions_path,
                    gold_path=gold_questions_path,
                    exact_match=False,
                )
            )
            scores = {
                "qa": qa_scores,
            }
            logging.info(utils.pretty_format_dict(scores))

            # Save the evaluation results
            output_evaluation_path = os.path.join(
                base_output_path,
                f"{base_filename}.eval.json",
            )
            utils.write_json(output_evaluation_path, scores)

            # Log the evaluation results
            logging.info(utils.pretty_format_dict(scores))
            logging.info(f"Saved the evaluation results to {output_evaluation_path}")

    ##################
    # Closing
    ##################

    logging.info("Done.")
    sw.stop("main")
    logging.info("Time: %f min." % sw.get_time("main", minute=True))


def set_logger(filename: str, overwrite: bool = False) -> None:
    if os.path.exists(filename) and not overwrite:
        logging.info("%s already exists." % filename)
        do_remove = input("Delete the existing log file? [y/n]: ")
        if (not do_remove.lower().startswith("y")) and (not len(do_remove) == 0):
            logging.info("Done.")
            sys.exit(0)

    root_logger = logging.getLogger()
    handler = logging.FileHandler(filename, "w")
    root_logger.addHandler(handler)


def instantiate_ner_component(
    ner_config: dict[str, Any],
    loaded_llm_map: dict[str, BaseLLM] | None,
) -> tuple[BaseNER, dict[str, BaseLLM]]:

    # Instantiate the NER component
    if ner_config["method_name"] == "biaffine_ner":
        # Load the Biaffine NER component
        ner = BiaffineNER.from_identifier(
            identifier=ner_config["identifier"]
        )
    elif ner_config["method_name"] == "llm_ner":
        # Instantiate the LLM wrapper
        llm, loaded_llm_map = instantiate_llm(
            config=ner_config,
            loaded_llm_map=loaded_llm_map
        )

        if "identifier" in ner_config:
            # Load the LLM-based NER component from the public snapshot
            ner = LLMNER.from_identifier(
                model=llm,
                identifier=ner_config["identifier"],
            )

        else:
            # Load the user-defined schema
            vocab_etype: dict[str, int] = {
                etype: etype_i
                for etype_i, etype in enumerate(ner_config["entity_types"])
            }
            etype_meta_info: dict[str, dict[str, str]] = ner_config["etype_meta_info"]

            # Instantiate the LLM-based NER component with the user-defined schema 
            ner = LLMNER(
                model=llm,
                prompt_template_name_or_path=ner_config["prompt_template_name_or_path"],
                vocab_etype=vocab_etype,
                etype_meta_info=etype_meta_info,
            )

    else:
        raise ValueError(f"Unknown NER method: {ner_config['method_name']}")

    return ner, loaded_llm_map


def instantiate_ed_retrieval_component(
    ed_retrieval_config: dict[str, Any],
    loaded_llm_map: dict[str, BaseLLM] | None,
) -> tuple[BaseEDRetriever, dict[str, BaseLLM]]:

    # Instantiate the ED-Retrieval component.
    # Also, re-build the index over entities.
    if ed_retrieval_config["method_name"] == "mention_name_entity_retriever":
        # Instantiate the ED-Retrieval component using simple mention-name assignment
        ed_retrieval = MentionNameEntityRetriever()
        ed_retrieval.make_index()
    elif ed_retrieval_config["method_name"] == "blink_bi_encoder":
        # Instantiate the BLINK Bi-Encoder ED-Retrieval component 
        ed_retrieval = BlinkBiEncoder.from_identifier(
            identifier=ed_retrieval_config["identifier"]
        )
        ed_retrieval.make_index(use_precomputed_entity_vectors=True)
    else:
        raise ValueError(
            f"Unknown ED-Retrieval method: {ed_retrieval_config['method_name']}"
        )

    return ed_retrieval, loaded_llm_map


def instantiate_ed_reranking_component(
    ed_reranking_config: dict[str, Any],
    loaded_llm_map: dict[str, BaseLLM] | None,
) -> tuple[BaseEDReranker, dict[str, BaseLLM]]:

    # Instantiate the ED-Reranking component
    if ed_reranking_config["method_name"] == "identical_entity_reranker":
        # Instantiate the ED-Reranking component using identical function
        ed_reranking = IdenticalEntityReranker()
    elif ed_reranking_config["method_name"] == "blink_cross_encoder":
        # Load the BLINK Cross-Encoder ED-Reranking component from the public snapshot
        ed_reranking = BlinkCrossEncoder.from_identifier(
            identifier=ed_reranking_config["identifier"]
        )
    elif ed_reranking_config["method_name"] == "llm_ed":
        # Instantiate the LLM wrapper
        llm, loaded_llm_map = instantiate_llm(
            config=ed_reranking_config,
            loaded_llm_map=loaded_llm_map
        )

        # Load the LLM-based ED-Reranking component from the public snapshot
        ed_reranking = LLMED.from_identifier(
            model=llm,
            identifier=ed_reranking_config["identifier"],
        )
    else:
        raise ValueError(
            f"Unknown ED-Reranking method: {ed_reranking_config['method_name']}"
        )

    return ed_reranking, loaded_llm_map


def instantiate_docre_component(
    docre_config: dict[str, Any],
    loaded_llm_map: dict[str, BaseLLM] | None,
) -> tuple[BaseDocRE, dict[str, BaseLLM]]:

    # Instantiate the DocRE component
    if docre_config["method_name"] == "atlop":
        # Load the ATLOP-based DocRE component from the public snapshot
        docre = ATLOP.from_identifier(
            identifier=docre_config["identifier"]
        )
    elif docre_config["method_name"] == "llm_docre":
        # Instantiate the LLM wrapper
        llm, loaded_llm_map = instantiate_llm(
            config=docre_config,
            loaded_llm_map=loaded_llm_map
        )

        if "identifier" in docre_config:
            # Load the LLM-based DocRE component from the public snapshot
            docre = LLMDocRE.from_identifier(
                model=llm,
                identifier=docre_config["identifier"],
            )

        else:
            # Load the user-defined schema
            possible_head_entity_types = docre_config["possible_head_entity_types"]
            possible_tail_entity_types = docre_config["possible_tail_entity_types"]
            vocab_relation: dict[str, int] = {
                rel: rel_i
                for rel_i, rel in enumerate(docre_config["relations"])
            }
            rel_meta_info: dict[str, dict[str, str]] = docre_config["rel_meta_info"]
            entity_dict_path = docre_config.get("entity_dict_path", None)

            # Instantiate the LLM-based DocRE component with the user-defined schema
            docre = LLMDocRE(
                model=llm,
                prompt_template_name_or_path=(
                    docre_config["prompt_template_name_or_path"]
                ),
                knowledge_base_name=docre_config["knowledge_base_name"],
                mention_style=docre_config["mention_style"],
                with_span_annotation=docre_config["with_span_annotation"],
                possible_head_entity_types=possible_head_entity_types,
                possible_tail_entity_types=possible_tail_entity_types,
                vocab_relation=vocab_relation,
                rel_meta_info=rel_meta_info,
                entity_dict_path=entity_dict_path,
            )
    else:
        raise ValueError(f"Unknown DocRE method: {docre_config['method_name']}")

    return docre, loaded_llm_map


def instantiate_entity_graph_construction_component(
    entity_graph_construction_config: dict[str, Any],
) -> BaseEntityGraphConstructor:

    # Instantiate the Entity Graph Construction component
    if entity_graph_construction_config["method_name"] == "entity_graph_constructor":
        entity_graph_construction = EntityGraphConstructor(
            missing_entity_policy=(
                entity_graph_construction_config["missing_entity_policy"]
            ),
            missing_entity_description=(
                entity_graph_construction_config["missing_entity_description"]
            ),
        )
    else:
        raise ValueError(
            f"Unknown entity graph construction method: {entity_graph_construction_config['method_name']}"
        )

    return entity_graph_construction


def instantiate_community_clustering_component(
    community_clustering_config: dict[str, Any],
) -> BaseCommunityClusterer:

    # Instantiate the Community Clustering component
    if community_clustering_config["method_name"] == "hierarchical_leiden":
        community_clustering = HierarchicalLeiden(
            max_cluster_size=community_clustering_config["max_cluster_size"],
            use_lcc=community_clustering_config["use_lcc"],
        )
    elif community_clustering_config["method_name"] == "neighborhood_aggregation":
        community_clustering = NeighborhoodAggregation(
            hop_size=community_clustering_config["hop_size"],
        )
    elif community_clustering_config["method_name"] == "triple_level_factorization":
        community_clustering = TripleLevelFactorization()
    else:
        raise ValueError(
            f"Unknown community clustering method: {community_clustering_config['method_name']}"
        )

    return community_clustering


def instantiate_report_generation_component(
    report_generation_config: dict[str, Any],
    loaded_llm_map: dict[str, BaseLLM] | None,
) -> tuple[BaseReportGenerator, dict[str, BaseLLM]]:

    # Instantiate the Report Generation component
    if report_generation_config["method_name"] == "template_based_report_generator":
        # Instantiate the template-based Report Generation component
        report_generation = TemplateBasedReportGenerator(
            relation_map=report_generation_config["relation_map"],
        )
    elif report_generation_config["method_name"] == "llm_based_report_generator":
        # Instantiate the LLM wrapper
        llm, loaded_llm_map = instantiate_llm(
            config=report_generation_config,
            loaded_llm_map=loaded_llm_map,
        )

        # Instantiate the LLM-based Report Generation component
        report_generation = LLMBasedReportGenerator(
            model=llm,
            prompt_template_name_or_path=(
                report_generation_config["prompt_template_name_or_path"]
            ),
            relation_map=report_generation_config["relation_map"],
        )
    else:
        raise ValueError(
            f"Unknown report generation method: {report_generation_config['method_name']}"
        )

    return report_generation, loaded_llm_map


def instantiate_chunking_component(
    chunking_config: dict[str, Any],
) -> BaseChunker:

    # Instantiate the Chunking component
    if chunking_config["method_name"] == "chunker":
        model_name = None
        if "spacy_model_name" in chunking_config:
            model_name = chunking_config["spacy_model_name"]
        chunker = Chunker(
            model_name=model_name,
        )
    else:
        raise ValueError(f"Unknown chunking method: {chunking_config['method_name']}")

    return chunker


def instantiate_passage_retrieval_component(
    passage_retrieval_config: dict[str, Any],
) -> BasePassageRetriever:

    # Instantiate the BM25-based Passage Retrieval component
    if passage_retrieval_config["method_name"] == "bm25":
        passage_retrieval = BM25(
            tokenizer=lambda text: text.lower().split(),
            k1=passage_retrieval_config["k1"],
            b=passage_retrieval_config["b"],
        )

    # Instantiate the Contriever-based Passage Retrieval component
    elif passage_retrieval_config["method_name"] == "contriever":
        passage_retrieval = Contriever(
            model_name=passage_retrieval_config["model_name"],
            max_passage_length=passage_retrieval_config["max_passage_length"],
            pooling_method=passage_retrieval_config["pooling_method"],
            normalize=passage_retrieval_config["normalize"],
            metric=passage_retrieval_config["metric"],
        )

    # Instantiate the Qwen3-Embedding-based Passage Retrieval component
    elif passage_retrieval_config["method_name"] == "qwen3_embedding":
        passage_retrieval = Qwen3Embedding(
            model_name=passage_retrieval_config["model_name"],
            max_passage_length=passage_retrieval_config["max_passage_length"],
            normalize=passage_retrieval_config["normalize"],
            metric=passage_retrieval_config["metric"],
            query_instruction=passage_retrieval_config["query_instruction"],
        )

    else:
        raise ValueError(
            f"Unknown passage retrieval method: {passage_retrieval_config['method_name']}"
        )

    return passage_retrieval


def instantiate_qa_component(
    qa_config: dict[str, Any],
    loaded_llm_map: dict[str, BaseLLM] | None,
) -> tuple[BaseQA, dict[str, BaseLLM]]:

    # Instantiate the QA component
    if qa_config["method_name"] == "llm_qa":
        # Instantiate the LLM wrapper
        llm, loaded_llm_map = instantiate_llm(
            config=qa_config,
            loaded_llm_map=loaded_llm_map,
        )

        # Instantiate the LLM-based QA component
        qa = LLMQA(
            model=llm,
            prompt_template_name_or_path=qa_config["prompt_template_name_or_path"],
            n_contexts=qa_config["n_contexts"],
        )

    else:
        raise ValueError(f"Unknown QA method: {qa_config['method_name']}")

    return qa, loaded_llm_map


def instantiate_llm(
    config: dict[str, Any],
    loaded_llm_map: dict[str, BaseLLM]
) -> tuple[BaseLLM, dict[str, BaseLLM]]:

    # Create a key for reusing the same LLM instance
    key_parts = [
        config["llm_provider"],
        config["llm_model_name"],
        str(config["llm_max_new_tokens"]),
    ]
    if config["llm_provider"] == "hf":
        key_parts.append(str(config["llm_quantization_bits"]))
    key = "___".join(key_parts)

    # If the key is found, reuse the corresponding LLM
    if key in loaded_llm_map:
        return loaded_llm_map[key], loaded_llm_map

    # Instantiate the LLM wrapper
    if config["llm_provider"] == "openai":
        llm = OpenAILLM(
            model_name=config["llm_model_name"],
            max_new_tokens=config["llm_max_new_tokens"],
        )
    elif config["llm_provider"] == "hf":
        llm = HuggingFaceLLM(
            model_name=config["llm_model_name"],
            max_new_tokens=config["llm_max_new_tokens"],
            quantization_bits=config["llm_quantization_bits"],
        )
    else:
        raise ValueError(f"Unknown LLM provider: {config['llm_provider']}")

    # Add the key-LLM record to the dict
    loaded_llm_map[key] = llm

    return llm, loaded_llm_map


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO
    )
    logging.getLogger("httpx").addFilter(
        lambda r: "huggingface.co" not in r.getMessage()
    )

    parser = argparse.ArgumentParser()

    # Method
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--config_name", type=str, required=True)

    # Input Data
    parser.add_argument("--input_documents", type=str, default=None)
    parser.add_argument("--entity_dict", type=str, default=None)
    parser.add_argument("--additional_triples", type=str, default=None)
    parser.add_argument("--input_questions", type=str, default=None)

    # Output Path
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default=None)

    # Action
    parser.add_argument(
        "--actiontype",
        type=str,
        required=True,
        choices=[
            "triple_extraction",
            "entity_graph_construction",
            "community_clustering",
            "report_generation",
            "chunking",
            "passage_retrieval_indexing",
            "inference",
        ],
    )

    # Evaluation
    parser.add_argument("--do_evaluation", action="store_true")
    parser.add_argument("--gold", type=str, default=None)

    args = parser.parse_args()

    main(args)
