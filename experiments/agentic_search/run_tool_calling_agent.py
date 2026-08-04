import argparse
from dataclasses import asdict
import json
import logging
import os
from typing import Any
import sys

import torch
from tqdm import tqdm
import transformers

from kapipe import evaluation
from kapipe import utils
from kapipe.utils import StopWatch

from kapipe.agents import AgentTrajectory, Tool, ToolCallingAgent

from kapipe.llms import BaseLLM, HuggingFaceLLM, OpenAILLM
from kapipe.passage_retrieval import (
    BM25,
    BasePassageRetriever,
    Contriever,
    Qwen3Embedding,
)


def main(args):
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
    input_questions_path = args.input_questions
    index_dir = args.index_dir

    # Output Path
    results_dir = args.results_dir
    prefix = args.prefix
    if prefix is None or prefix == "None":
        prefix = utils.get_current_time()
        args.prefix = prefix

    # Evaluation
    do_evaluation = args.do_evaluation
    gold_questions_path = args.gold_answers
    gold_contexts_path = args.gold_contexts

    ##################
    # Logging Setup
    ##################

    # Set base output path
    base_output_path = os.path.join(
        results_dir,
        "tool_calling_agent",
        method_name,
        config_name,
        prefix
    )
    utils.mkdir(base_output_path)

    base_filename = os.path.splitext(
        os.path.basename(input_questions_path)
    )[0]

    # Set logger
    set_logger(
        os.path.join(
            base_output_path,
            base_filename + ".inference.log"
        ),
        # overwrite=True
    )

    # Show arguments
    logging.info(utils.pretty_format_dict(vars(args)))

    ##################
    # Data
    ##################

    # Load questions
    logging.info(f"Loading questions from {input_questions_path} ...")
    questions = utils.read_json(input_questions_path)
    logging.info(f"Loaded {len(questions)} questions")

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

    # Instantiate the LLM
    llm = instantiate_llm(llm_config=config["llm"])

    # Instantiate the Passage Retrieval component
    passage_retrieval = instantiate_passage_retrieval_component(
        passage_retrieval_config=config["passage_retrieval"],
    )

    # Load the prebuilt Passage Retrieval index
    passage_retrieval.load_index(index_dir=index_dir)

    # Transform the Passage Retrieval component as a tool for the Tool-Calling Agent
    passage_retrieval_tool = Tool(
        name="passage_retrieval",
        description="Retrieve passages relevant to a natural-language query. Use this tool to obtain factual evidence before answering a question.",
        input_schema={
            "query": {
                "type": "str",
                "description": "Natural-language query.",
                "required": True,
            },
        },
        output_schema={
            "passages": {
                "type": "list[dict[str, Any]]",
                "description": "Retrieved passages ordered by relevance to the query.",
            },
        },
        function=lambda tool_input: {
            "passages": passage_retrieval.search(
                queries=[tool_input["query"]],
                top_k=config["passage_retrieval"]["top_k"],
            )[0],
        },
    )

    # Instantiate the tool-calling agent
    agent = ToolCallingAgent(
        llm=llm,
        tools=[
            passage_retrieval_tool
        ],
        max_steps=config["max_steps"],
    )

    ##################
    # Method Execution
    ##################

    logging.info(f"Applying the tool-calling agent to {len(questions)} questions in {input_questions_path} ...")

    # Apply the tool-calling agent to the questions
    result_questions = []

    # Initialize a human-readable trace file for this inference run
    output_trace_path: str = os.path.join(
        base_output_path,
        f"{base_filename}.trace.txt",
    )
    with open(output_trace_path, mode="w", encoding="utf-8"):
        pass

    # Apply the tool-calling agent to the question
    result_questions = []
    for question in tqdm(questions):
        # Run the agent
        trajectory: AgentTrajectory = agent.infer(
            initial_input=question["question"],
        )

        # Ensure that the agent produced a final answer
        if trajectory.final_answer is None:
            raise RuntimeError(f"The agent did not produce a final answer: {question['question_key']}")

        # Collect passages in the order in which the agent retrieved them
        retrieved_passages = []
        for agent_step in trajectory.agent_steps:
            # Skip unrelated tools
            if agent_step.tool_name != "passage_retrieval":
                continue
            
            # Require the output structure declared by the retrieval Tool
            if not isinstance(agent_step.tool_output, dict):
                raise TypeError(
                    "The passage_retrieval tool output must be a dictionary."
                )

            # Append passages while preserving tool-call and retrieval order
            retrieved_passages.extend(agent_step.tool_output["passages"])

        # Build a compatible prediction record
        result_question = {
            "question_key": question["question_key"],
            "question": question["question"],
            "output_answer": trajectory.final_answer,
            "contexts": retrieved_passages,
            "agent_trajectory": asdict(trajectory),
        }
        result_questions.append(result_question)

        # Append the completed trajectory to the human-readable trace file
        write_agent_trace(
            output_trace_path=output_trace_path,
            question=question,
            trajectory=trajectory,
        )

    # Save the results
    output_questions_path = os.path.join(
        base_output_path,
        f"{base_filename}.pred.json" ,
    )
    utils.write_json(output_questions_path, result_questions)
    logging.info(f"Saved the prediction results to {output_questions_path}")

    ##################
    # Evaluation
    ##################

    if do_evaluation:
        # Require gold answers only when evaluation is requested
        if gold_questions_path is None:
            raise ValueError("--gold_answers is required when --do_evaluation is set")
        if gold_contexts_path is None:
            raise ValueError("--gold_contexts is required when --do_evaluation is set") 

        # Evaluate the prediction results
        qa_scores = evaluation.qa.accuracy(
            pred_path=output_questions_path,
            gold_path=gold_questions_path,
            exact_match=False,
        ) | evaluation.qa.token_level_f1(
            pred_path=output_questions_path,
            gold_path=gold_questions_path
        ) | evaluation.qa.recall(
            pred_path=output_questions_path,
            gold_path=gold_questions_path,
            exact_match=False,
        )
        ret_scores = evaluation.passage_retrieval.precision_recall_at_k(
            pred_path=output_questions_path,
            gold_path=gold_contexts_path,
            passage_to_identifier=lambda p: p["text"]
        )
        ret_scores.update(
            evaluation.passage_retrieval.ndcg_at_k(
                pred_path=output_questions_path,
                gold_path=gold_contexts_path,
                passage_to_identifier=lambda p: p["text"]
            )
        )
        scores = {
            "qa": qa_scores,
            "passage_retrieval": ret_scores,
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


def instantiate_llm(
    llm_config: dict[str, Any],
) -> BaseLLM:

    # Instantiate the LLM wrapper
    if llm_config["llm_provider"] == "openai":
        llm = OpenAILLM(
            model_name=llm_config["llm_model_name"],
            max_new_tokens=llm_config["llm_max_new_tokens"],
        )
    elif llm_config["llm_provider"] == "hf":
        llm = HuggingFaceLLM(
            model_name=llm_config["llm_model_name"],
            max_new_tokens=llm_config["llm_max_new_tokens"],
            quantization_bits=llm_config["llm_quantization_bits"],
        )
    else:
        raise ValueError(f"Unknown LLM provider: {llm_config['llm_provider']}")

    return llm


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


def write_agent_trace(
    output_trace_path: str,
    question: dict[str, Any],
    trajectory: AgentTrajectory,
) -> None:
    """Append one completed agent trajectory to a human-readable text file."""

    # Require a completed trajectory
    if trajectory.final_answer is None:
        raise ValueError("The agent trajectory must contain a final answer.")

    # Add the question-level header
    trace_lines: list[str] = [
        "=" * 50,
        f"QUESTION KEY: {question['question_key']}",
        f"QUESTION: {question['question']}",
        "",
    ]

    # Add each LLM interaction in execution order
    for step_i, agent_step in enumerate(
        trajectory.agent_steps,
        start=1,
    ):
        trace_lines.extend([
            "-" * 50,
            f"STEP: {step_i}",
            "",
            "[LLM INPUT]",
            agent_step.llm_input,
            "",
            "[LLM OUTPUT]",
            agent_step.llm_output,
            "",
        ])

        # Add the Tool interaction only when the step executed a Tool
        if agent_step.action_type == "use_tool":
            trace_lines.extend([
                "[TOOL INPUT]",
                json.dumps(
                    agent_step.tool_input,
                    ensure_ascii=False,
                    indent=2,
                ),
                "",
                "[TOOL OUTPUT]",
                json.dumps(
                    agent_step.tool_output,
                    ensure_ascii=False,
                    indent=2,
                ),
                "",
            ])

    # Add the normalized final answer for quick inspection
    trace_lines.extend([
        "[FINAL ANSWER]",
        trajectory.final_answer,
        "",
    ])

    # Append the complete trace while preserving earlier questions
    with open(
        output_trace_path,
        mode="a",
        encoding="utf-8",
    ) as trace_file:
        trace_file.write("\n".join(trace_lines))
        trace_file.write("\n")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO
    )

    parser = argparse.ArgumentParser()

    # Method
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--config_name", type=str, required=True)

    # Input Data
    parser.add_argument("--input_questions", type=str, required=True)
    parser.add_argument("--index_dir", type=str, required=True)

    # Output Path
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default=None)

    # Evaluation
    parser.add_argument("--do_evaluation", action="store_true")
    parser.add_argument("--gold_answers", type=str, default=None)
    parser.add_argument("--gold_contexts", type=str, default=None)

    args = parser.parse_args()

    main(args) 
