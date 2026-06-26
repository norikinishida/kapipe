from __future__ import annotations

from collections import defaultdict
import copy
import logging
import os
import re
from typing import Any

import torch
from tqdm import tqdm

from .. import evaluation
from .. import utils
from ..datatypes import (
    Config,
    Document,
    Mention,
    Entity,
    EntityPage,
    CandidateEntitiesForDocument,
    ContextsForOneExample
)
from ..llms import HuggingFaceLLM, OpenAILLM
from ..resources import resolve_snapshot_path


logger = logging.getLogger(__name__)


N_CAND = 3
N_MENT_PER_CHUNK = 5


class LLMED:

    def __init__(
        self,
        model: HuggingFaceLLM | OpenAILLM,
        # Initialization
        config: Config | str | None = None,
        path_entity_dict: str | None = None,
        demonstration_documents: list[Document] | str | None = None,
        demonstration_candidate_entities: (
            list[CandidateEntitiesForDocument] | str | None
        ) = None,
        # Loading
        path_snapshot: str | None = None,
        identifier: str | None = None,
    ):
        logger.info("########## LLMED Initialization Starts ##########")

        # Resolve a public identifier to the corresponding local snapshot
        if identifier is not None:
            if path_snapshot is not None:
                raise ValueError(
                    "identifier and path_snapshot cannot be specified together."
                )

            path_snapshot = resolve_snapshot_path(
                component_name="ed_reranking",
                method_name="llm_ed",
                identifier=identifier,
            )

        self.model = model
        self.path_snapshot = path_snapshot
        self.identifier = identifier

        if path_snapshot is not None:
            # Explicit initialization resources must not be mixed with a
            # complete snapshot.
            if config is not None:
                raise ValueError(
                    "config cannot be specified when loading a snapshot."
                )
            if path_entity_dict is not None:
                raise ValueError(
                    "path_entity_dict cannot be specified when loading "
                    "a snapshot."
                )
            if demonstration_documents is not None:
                raise ValueError(
                    "demonstration_documents cannot be specified when loading "
                    "a snapshot."
                )
            if demonstration_candidate_entities is not None:
                raise ValueError(
                    "demonstration_candidate_entities cannot be specified when "
                    "loading a snapshot."
                ) 

            # Specify the default paths for the resources in the snapshot
            config = path_snapshot + "/config"
            path_entity_dict = path_snapshot + "/entity_dict.json"
            path_demonstration_documents = (
                path_snapshot + "/demonstration_documents.json"
            )
            path_demonstration_candidate_entities = (
                path_snapshot + "/demonstration_candidate_entities.json"
            )

            # Use no demonstrations when loading an old zero-shot snapshot
            if os.path.exists(path_demonstration_documents):
                demonstration_documents = path_demonstration_documents
            else:
                demonstration_documents = []

            # Use no demonstration candidate entities when loading an old zero-shot snapshot
            if os.path.exists(path_demonstration_candidate_entities):
                demonstration_candidate_entities = (
                    path_demonstration_candidate_entities
                )
            else:
                demonstration_candidate_entities = []

        # Load the configuration
        if isinstance(config, str):
            config_path = config
            config = utils.get_hocon_config(config_path=config_path)
            logger.info(f"Loaded configuration from {config_path}")
        self.config = config
        logger.info(utils.pretty_format_dict(self.config))

        # Load the entity dictionary
        logger.info(f"Loading entity dictionary from {path_entity_dict}")
        self.entity_dict = {
            epage["entity_id"]: epage
            for epage in utils.read_json(path_entity_dict)
        }
        logger.info(f"Completed loading of entity dictionary with {len(self.entity_dict)} entities from {path_entity_dict}")

        # Load the demonstration documents
        if isinstance(demonstration_documents, str):
            path_demonstration_documents = demonstration_documents
            demonstration_documents = utils.read_json(path_demonstration_documents)
            logger.info(
                f"Loaded {len(demonstration_documents)} demonstration documents "
                f"from {path_demonstration_documents}"
            )

        # Load demonstration candidate entities
        if isinstance(demonstration_candidate_entities, str):
            path_demonstration_candidate_entities = demonstration_candidate_entities
            demonstration_candidate_entities = utils.read_json(
                path_demonstration_candidate_entities
            )
            logger.info(
                "Loaded "
                f"{len(demonstration_candidate_entities)} "
                "demonstration candidate-entity records "
                f"from {path_demonstration_candidate_entities}"
            )

        # Represent the zero-shot setting as an empty list
        if demonstration_documents is None:
            demonstration_documents = []
        if demonstration_candidate_entities is None:
            demonstration_candidate_entities = [] 

        # Validate the lengths of demonstration documents and demonstration candidate entities
        if len(demonstration_documents) != len(demonstration_candidate_entities):
            raise ValueError(
                "demonstration_documents and demonstration_candidate_entities "
                "must have the same length."
            )

        self.demonstration_documents: list[Document] = demonstration_documents
        self.demonstration_candidate_entities: list[CandidateEntitiesForDocument] = (
            demonstration_candidate_entities
        )

        # Initialize the prompt processor, whioch generates prompts for the LLM
        self.prompt_processor = PromptProcessor(
            prompt_template_name_or_path=self.config["prompt_template_name_or_path"],
            knowledge_base_name_prompt=self.config["knowledge_base_name"],
            entity_dict=self.entity_dict,
        )

        # Check the provider and initialize the LLM model accordingly
        self.provider = self.config["provider"]
        if self.provider not in ["hf", "openai"]:
            raise ValueError(f"Invalid provider: {self.provider}")
        logger.info("LLM is provided by an argument")

        # Define regular expression for output parsing.
        # Parse generated lines of the following form:
        #
        #     - [mention text] | [entity ID]
        #
        self.re_comp = re.compile("(.+?)\s*(.+?)\s*\|\s*(.+?)$")

        logger.info("########## LLMED Initialization Ends ##########")

    def save(self, path_snapshot: str) -> None:
        """Save the configuration, entity dictionary, demonstration documents, and demonstration candidate entities to a snapshot directory."""

        path_config = path_snapshot + "/config"
        path_entity_dict = path_snapshot + "/entity_dict.json"
        path_demonstration_documents = (
            path_snapshot + "/demonstration_documents.json"
        )
        path_demonstration_candidate_entities = (
            path_snapshot + "/demonstration_candidate_entities.json"
        )

        utils.write_json(path_config, self.config)
        utils.write_json(path_entity_dict, list(self.entity_dict.values()))
        utils.write_json(
            path_demonstration_documents,
            self.demonstration_documents
        )
        utils.write_json(
            path_demonstration_candidate_entities,
            self.demonstration_candidate_entities
        )

    def rerank(
        self,
        document: Document,
        candidate_entities_for_doc: CandidateEntitiesForDocument,
        # Optional: prompt augmentation
        contexts_for_doc: ContextsForOneExample | None = None
    ) -> Document:
        """Rerank candidate entities for a single document."""

        with torch.no_grad():
            # Switch to inference mode for Hugging Face models
            if self.provider == "hf":
                self.model.llm.eval()
 
            # Split mentions into groups and perform reranking on the groups iteratively
            prompt_list: list[str] = []
            generated_text_list: list[str] = []
            target_mentions_list: list[list[Mention]] = []
            indices = list(range(0, len(document["mentions"])))

            for m_i in range(0, len(document["mentions"]), N_MENT_PER_CHUNK):
                # Get mention indices for this group
                target_mention_indices = indices[m_i: m_i + N_MENT_PER_CHUNK]

                # Generate a prompt
                prompt = self.prompt_processor.generate(
                    document=document,
                    candidate_entities_for_doc=candidate_entities_for_doc,
                    target_mention_indices=target_mention_indices,
                    demonstration_documents=self.demonstration_documents,
                    demonstration_candidate_entities=(
                        self.demonstration_candidate_entities
                    ),
                    contexts_for_doc=contexts_for_doc
                )
                prompt_list.append(prompt)

                # Generate a response
                generated_text = self.model.generate(prompt)
                generated_text_list.append(generated_text)

                # Convert the generated text into mention-level records
                target_mentions = self.structurize(
                    document=document,
                    candidate_entities_for_doc=candidate_entities_for_doc,
                    generated_text=generated_text,
                    target_mention_indices=target_mention_indices
                )
                target_mentions_list.append(target_mentions)

            # Aggregate the mention-level records
            mentions = utils.flatten_lists(target_mentions_list)
            assert len(mentions) == len(document["mentions"])
            entities: list[Entity] = utils.aggregate_mentions_to_entities(
                document=document,
                mentions=mentions
            )

            # Integrate the entities into the document
            result_document = copy.deepcopy(document)
            for m_i in range(len(result_document["mentions"])):
                result_document["mentions"][m_i].update(mentions[m_i])
            result_document["entities"] = entities
            result_document["ed_prompt"] = "\n@@@@@@@@@@\n".join(prompt_list)
            result_document["ed_generated_text"] = "\n@@@@@@@@@@\n".join(
                generated_text_list
            )

            return result_document

    def structurize(
        self,
        document: Document,
        candidate_entities_for_doc: CandidateEntitiesForDocument,
        generated_text: str,
        target_mention_indices: list[int]
    ) -> list[Mention]:
        """Structurize the generated text into mention-level entity records."""

        doc_key = document["doc_key"]

        # Get one-to-many mapping from normalized mention name to mention indices
        normalized_name_to_mention_indices = defaultdict(list)
        words = " ".join(document["sentences"]).split()
        for m_i, mention in enumerate(document["mentions"]):
            if m_i in target_mention_indices:
                b_i, e_i = mention["span"]
                name = " ".join(words[b_i: e_i+1])
                normalized_name = name.lower()
                normalized_name_to_mention_indices[normalized_name].append(m_i)
        
        # Get a list of possible entity IDs
        possible_entity_ids = []
        for cands in candidate_entities_for_doc["candidate_entities"]:
            for cand in cands:
                possible_entity_ids.append(cand["entity_id"])
        possible_entity_ids = set(possible_entity_ids)

        # Initialize the output mentions
        # NOTE: We create a list of mentions, whose length is the same with the original number of mentions
        # We will filter out the mentions later using the target_mention_indices
        mentions = []
        for _ in range(len(document["mentions"])):
            mentions.append(
                {
                    "entity_id": "NO-PRED",
                    # "corresponding_output_line": None
                }
            )

        # Parse each generated line
        names = []
        entity_ids = []
        for generated_line in generated_text.split("\n"):
            generated_line = generated_line.strip()

            # Skip the empty line
            if generated_line == "":
                continue

            # Parse the generated line
            parsed = self.re_comp.findall(generated_line)
            if not (len(parsed) == 1 and len(parsed[0]) == 3):
                logger.info(f"[{doc_key}] Skipped a generated line of invalid formatting: '{generated_line}'")
                continue
            _, name, entity_id = parsed[0]
            names.append(name)
            entity_ids.append(entity_id)

        if len(names) == len(entity_ids) == len(target_mention_indices):
            # The number of entities (i.e., len(names), len(entity_ids)) is the same with that of all mentions in the document
            if len(document["mentions"]) == 0:
                assert len(target_mention_indices) == 0
            else:
                for m_i, (name, entity_id) in enumerate(zip(names, entity_ids)):
                    # Skip checking the mention names
    
                    # Check whether the entity ID can be found in the possible list
                    if entity_id not in possible_entity_ids:
                        logger.info(f"[{doc_key}] Skipped a generated line with invalid concept ID: {entity_id}")
                        continue
    
                    # Add mention
                    mentions[target_mention_indices[m_i]]["entity_id"] = entity_id

        else:
            for name, entity_id in zip(names, entity_ids):

                # Check whether the mention can be found in the possible list
                normalized_name = name.lower()
                pattern = r"\s*".join(re.escape(c) for c in normalized_name)
                normalized_name2 = None
                for n in normalized_name_to_mention_indices.keys():
                    results = list(re.finditer(
                        "@@@" + pattern + "@@@",
                        "@@@" + n + "@@@"
                    ))
                    if len(results) == 1:
                        normalized_name2 = n
                        break
                if normalized_name2 is None:
                    logger.info(f"[{doc_key}] Skipped a generated line with invalid mention: '{normalized_name}' not in {list(normalized_name_to_mention_indices.keys())}")
                    continue
                normalized_name = normalized_name2

                # Check whether the entity ID can be found in the possible list
                if entity_id not in possible_entity_ids:
                    logger.info(f"[{doc_key}] Skipped a generated line with invalid concept ID: {entity_id}")
                    continue

                # Add mention
                mention_indices = normalized_name_to_mention_indices[normalized_name]
                for m_i in mention_indices:
                    mentions[m_i]["entity_id"] = entity_id

        # Check
        for m_i in range(len(document["mentions"])):
            if m_i not in target_mention_indices:
                assert mentions[m_i]["entity_id"] == "NO-PRED"

        return [mentions[m_i] for m_i in target_mention_indices]

    def batch_rerank(
        self,
        documents: list[Document],
        candidate_entities: list[CandidateEntitiesForDocument],
        # Optional: context augmentation
        contexts: list[ContextsForOneExample] | None = None
    ) -> list[Document]:
        """Rerank candidate entities for a batch of documents."""

        result_documents: list[Document] = []

        # Use empty contexts when no contexts are provided
        if contexts is None:
            contexts = [None] * len(documents)

        for (
            document,
            candidate_entities_for_doc,
            contexts_for_doc
        ) in tqdm(
                zip(
                    documents,
                    candidate_entities,
                    contexts
                ),
                total=len(documents),
                desc="reranking steps"
            ):
            result_document = self.rerank(
                document=document,
                candidate_entities_for_doc=candidate_entities_for_doc,
                contexts_for_doc=contexts_for_doc
            )
            result_documents.append(result_document)

        return result_documents


class PromptProcessor:

    def __init__(
        self,
        prompt_template_name_or_path: str,
        knowledge_base_name_prompt: str,
        entity_dict: dict[str, EntityPage],
    ) -> None:

        self.prompt_template_name_or_path = prompt_template_name_or_path
        self.knowledge_base_name_prompt = knowledge_base_name_prompt
        self.entity_dict = entity_dict

        # Load the prompt template
        self.prompt_template = utils.read_prompt_template(
            prompt_template_name_or_path=self.prompt_template_name_or_path,
            prompt_template_package_name="kapipe.ed_reranking.prompt_templates",
        )
 
    def generate(
        self,
        document: Document,
        candidate_entities_for_doc: CandidateEntitiesForDocument,
        target_mention_indices: list[int],
        demonstration_documents: list[Document],
        demonstration_candidate_entities: list[CandidateEntitiesForDocument],
        # optional: context augmentation
        contexts_for_doc: ContextsForOneExample | None = None
    ) -> str:
        """Generate a prompt for the LLM based on the input document, candidate entities, and optional context augmentation."""

        ##########
        # Demonstrations Prompt
        ##########

        # Create candidate entity pages for the demonstration documents
        candidate_entity_pages_for_demos: list[list[list[EntityPage]]] = []
        for candidate_entities_for_demo in demonstration_candidate_entities:
            candidate_entity_pages_for_demo: list[list[EntityPage]] = []
            for candidate_entities_for_one_mention in (
                candidate_entities_for_demo["candidate_entities"]
            ):
                candidate_entity_pages_for_one_mention: list[EntityPage] = [
                    self.entity_dict[cand_key_dict["entity_id"]]
                    for cand_key_dict in candidate_entities_for_one_mention
                ]
                candidate_entity_pages_for_demo.append(
                    candidate_entity_pages_for_one_mention
                )
            candidate_entity_pages_for_demos.append(
                candidate_entity_pages_for_demo
            )

        # Generate the prompt part for demonstrations
        demonstrations_prompt = self.generate_demonstrations_prompt(
            demonstration_documents=demonstration_documents,
            candidate_entity_pages_for_demos=candidate_entity_pages_for_demos
        )

        ##########
        # Contexts Prompt
        ##########

        if contexts_for_doc is not None:
            # Create contexts
            context_texts: list[str] = []
            for passage in contexts_for_doc["contexts"]:
                text = utils.create_text_from_passage(passage=passage, sep=" : ")
                context_texts.append(text)
            # Generate prompt part for contexts
            contexts_prompt = self.generate_contexts_prompt(
                context_texts=context_texts
            )
        else:
            contexts_prompt = ""

        ##########
        # Test Case Prompt
        ##########

        # Create candidate entities for the input document
        candidate_entity_pages_for_doc: list[list[EntityPage]] = []
        for candidate_entities_for_one_mention in (
            candidate_entities_for_doc["candidate_entities"]
        ):
            candidate_entity_pages_for_one_mention: list[EntityPage] = [
                self.entity_dict[cand_key_dict["entity_id"]]
                for cand_key_dict in candidate_entities_for_one_mention
            ]
            candidate_entity_pages_for_doc.append(
                candidate_entity_pages_for_one_mention
            )

        # Generate prompt part for test case
        test_case_prompt = self.generate_test_case_prompt(
            document=document,
            candidate_entity_pages_for_doc=candidate_entity_pages_for_doc,
            target_mention_indices=target_mention_indices
        )

        ##########
        # Final Prompt
        ##########

        # Combine the prompt parts
        prompt = self.prompt_template.format(
            knowledge_base_name_prompt=self.knowledge_base_name_prompt,
            demonstrations_prompt=demonstrations_prompt,
            contexts_prompt=contexts_prompt,
            test_case_prompt=test_case_prompt
        )

        return prompt

    def generate_demonstrations_prompt(
        self,
        demonstration_documents: list[Document],
        candidate_entity_pages_for_demos: list[list[list[EntityPage]]]
    ) -> str:
        """Generate a prompt for the demonstrations based on the demonstration documents and their corresponding candidate entity pages."""

        prompt = ""
        n_demos = len(demonstration_documents)
        for demo_i, (demo_doc, cand_ent_pages_for_demo) in enumerate(zip(
            demonstration_documents,
            candidate_entity_pages_for_demos
        )):
            prompt += f"Example {demo_i+1}:\n"

            # Generate prompt part """fro the input text
            prompt += f"Text: {self.generate_input_text_prompt(document=demo_doc)}\n"
           
            # Sample target mention indices
            target_mention_indices = [
                m_i for m_i, m in enumerate(demo_doc["mentions"])
                if m["entity_id"] in self.entity_dict
            ]

            # Generate prompt part for the mentions and their candidate concepts
            mention_candidates_pairs_prompt = (
                self.generate_input_mention_candidates_pairs_prompt(
                    document=demo_doc, 
                    candidate_entity_pages_for_doc=cand_ent_pages_for_demo,
                    target_mention_indices=target_mention_indices[:2],
                    demonstration_mode=True
                )
            )
            prompt += f"{mention_candidates_pairs_prompt}\n"

            # Generate prompt part for the output
            output_prompt = self.generate_output_prompt(
                document=demo_doc,
                candidate_entity_pages_for_doc=cand_ent_pages_for_demo,
                target_mention_indices=target_mention_indices[:2]
            )
            prompt += "Output:\n"
            prompt += f"{output_prompt}\n"

            if demo_i < n_demos - 1:
                prompt += "\n"

        return prompt.rstrip()

    def generate_contexts_prompt(self, context_texts: list[str]) -> str:
        """Generate a prompt for the contexts based on the provided context texts."""

        n_contexts = len(context_texts)
        if n_contexts == 0:
            return ""
        else:
            prompt = ""
            for context_i, content in enumerate(context_texts):
                prompt += f"[{context_i+1}] {content.strip()} \n"
                if context_i < n_contexts - 1:
                    prompt += "\n"
            return prompt.rstrip()

    def generate_test_case_prompt(
        self,
        document: Document,
        candidate_entity_pages_for_doc: list[list[EntityPage]],
        target_mention_indices: list[int]
    ) -> str:
        """Generate a prompt for the test case based on the input document, candidate entity pages, and target mention indices."""

        prompt = ""

        # Generate prompt part for the input text
        prompt += f"Text: {self.generate_input_text_prompt(document=document)}\n"

        # Generate prompt part for the mentions and their candidate concepts 
        mention_candidates_pairs_prompt = (
            self.generate_input_mention_candidates_pairs_prompt(
                document=document, 
                candidate_entity_pages_for_doc=candidate_entity_pages_for_doc,
                target_mention_indices=target_mention_indices,
                demonstration_mode=False
            )
        )
        prompt += f"{mention_candidates_pairs_prompt}\n"

        return prompt.rstrip()

    def generate_input_text_prompt(self, document: Document) -> str:
        """Generate a prompt for the input text based on the provided document."""

        prompt = " ".join(document["sentences"]) + "\n"

        return prompt.rstrip()

    def generate_input_mention_candidates_pairs_prompt(
        self,
        document: Document,
        candidate_entity_pages_for_doc: list[list[EntityPage]],
        target_mention_indices: list[int],
        demonstration_mode: bool = False
    ) -> str:
        """Generate a prompt for the mention-candidate pairs based on the provided document, candidate entity pages, and target mention indices."""

        # Aggregate mentions strings
        words = " ".join(document["sentences"]).split()
        names = []
        for m_i, mention in enumerate(document["mentions"]):
            if m_i in target_mention_indices:
                begin_i, end_i = mention["span"]
                name = " ".join(words[begin_i: end_i + 1])
                names.append(name)

        # Sample candidate concepts for each mention
        cands_list: list[list[EntityPage]] = []
        for m_i, candidate_entity_pages_for_one_mention in enumerate(
            candidate_entity_pages_for_doc
        ):
            if m_i in target_mention_indices:
                cands: list[EntityPage] = [] 
                if demonstration_mode:
                    # Place the ground-truth entity at the end of the candidates
                    gold_entity_id = document["mentions"][m_i]["entity_id"]
                    gold_entity_page = self.entity_dict[gold_entity_id]
                    for cand_page in candidate_entity_pages_for_one_mention[:N_CAND]:
                        if cand_page["entity_id"] != gold_entity_id:
                            cands.append(cand_page)
                    cands = cands[:2] + [gold_entity_page]
                else:
                    for cand_page in candidate_entity_pages_for_one_mention[:N_CAND]:
                        cands.append(cand_page)
                cands_list.append(cands)

        assert len(target_mention_indices) == len(names) == len(cands_list)

        # Texturize the mention-concepts pairs
        prompt = ""
        for m_i, (name, cands) in enumerate(zip(names, cands_list)):
            prompt += f"Mention {m_i + 1}: {name}\n"
            prompt += f"Candidate Concept IDs for Mention {m_i + 1}:\n"
            for cand_page in cands:
                entity_id = cand_page["entity_id"].replace("|", " ")
                canonical_name = cand_page["canonical_name"].replace("|", " ")
                desc = cand_page["description"].replace("|", " ").replace("\n", " ").rstrip()
                prompt += f"- ID: {entity_id} | Name: {canonical_name} | Description: {desc}\n"

        return prompt.rstrip()

    def generate_output_prompt(
        self,
        document: Document,
        candidate_entity_pages_for_doc: list[list[EntityPage]],
        target_mention_indices: list[int]
    ) -> str:
        """Generate a prompt for the output based on the provided document, candidate entity pages, and target mention indices."""

        prompt = ""

        words = " ".join(document["sentences"]).split()

        for m_i, mention in enumerate(document["mentions"]):
            if m_i in target_mention_indices:
                begin_i, end_i = mention["span"]
                name = " ".join(words[begin_i : end_i + 1])

                entity_id = mention["entity_id"]

                # If the ground-truth entity cannot be found in the candidates,
                # set the target output "NA".
                if entity_id not in {
                    epage["entity_id"]
                    for epage in candidate_entity_pages_for_doc[m_i]
                }:
                    entity_id = "NA"

                prompt += f"- {name} | {entity_id}\n"

        return prompt.rstrip()


 #####################
# Trainer (Evaluator)
#####################


class LLMEDTrainer:

    def __init__(self, base_output_path: str):
        self.base_output_path = base_output_path
        self.paths = self.get_paths()

    def get_paths(self) -> dict[str, str]:
        paths = {}

        # configurations
        paths["path_snapshot"] = self.base_output_path

        # evaluation outputs
        paths["path_dev_gold"] = self.base_output_path + "/dev.gold.json"
        paths["path_dev_pred"] = self.base_output_path + "/dev.pred.json"
        paths["path_dev_eval"] = self.base_output_path + "/dev.eval.json"
        paths["path_test_gold"] = self.base_output_path + "/test.gold.json"
        paths["path_test_pred"] = self.base_output_path + "/test.pred.json"
        paths["path_test_eval"] = self.base_output_path + "/test.eval.json"

        return paths

    def setup_dataset(
        self,
        reranker: LLMED,
        documents: list[Document],
        candidate_entities: list[CandidateEntitiesForDocument],
        split: str
    ) -> None:
        # Cache the gold annotations for evaluation
        path_gold = self.paths[f"path_{split}_gold"]
        if not os.path.exists(path_gold):
            kb_entity_ids = set(list(reranker.prompt_processor.entity_dict.keys()))
            gold_documents = []
            for document, candidate_entities_for_doc in tqdm(
                zip(documents, candidate_entities),
                desc="dataset setup"
            ):
                gold_doc = copy.deepcopy(document)

                cands_for_mentions = candidate_entities_for_doc["candidate_entities"]
                mentions = document["mentions"]
                assert len(mentions) == len(cands_for_mentions)

                for m_i, (mention, cands_for_mention) in enumerate(zip(
                    mentions,
                    cands_for_mentions
                )):
                    cand_entity_ids = [c["entity_id"] for c in cands_for_mention]
                    entity_id = mention["entity_id"]
                    in_kb = entity_id in kb_entity_ids
                    in_cand = entity_id in cand_entity_ids
                    gold_doc["mentions"][m_i]["in_kb"] = in_kb
                    gold_doc["mentions"][m_i]["in_cand"] = in_cand
                gold_documents.append(gold_doc)
            utils.write_json(path_gold, gold_documents)
            logger.info(f"Saved the gold annotations for evaluation in {path_gold}")

    def save_reranker(self, reranker: LLMED) -> None:
        reranker.save(path_snapshot=self.paths["path_snapshot"])

    def evaluate(
        self,
        reranker: LLMED,
        documents: list[Document],
        candidate_entities: list[CandidateEntitiesForDocument],
        contexts: list[ContextsForOneExample],
        split: str,
        #
        prediction_only: bool = False,
        get_scores_only: bool = False
    ) -> dict[str, Any]:
        # Apply the reranker
        result_documents = reranker.batch_rerank(
            documents=documents,
            candidate_entities=candidate_entities,
            contexts=contexts
        )

        # Save the prediction results
        utils.write_json(self.paths[f"path_{split}_pred"], result_documents)

        # Save the prompt-response pairs in plain text
        with open(
            self.paths[f"path_{split}_pred"].replace(".json", ".txt"), "w"
        ) as f:
            for result_doc in result_documents:
                doc_key = result_doc["doc_key"]
                prompt = result_doc["ed_prompt"]
                generated_text = result_doc["ed_generated_text"]
                f.write("-------------------------------------\n\n")
                f.write(f"DOC_KEY: {doc_key}\n\n")
                f.write("PROMPT:\n")
                f.write(prompt + "\n\n")
                f.write("GENERATED TEXT:\n")
                f.write(generated_text + "\n\n")
                f.flush()

        if prediction_only:
            return

        # Calculate the evaluation scores
        scores = evaluation.ed.accuracy(
            pred_path=self.paths[f"path_{split}_pred"],
            gold_path=self.paths[f"path_{split}_gold"],
            inkb=True
        )
        scores.update(evaluation.ed.fscore(
            pred_path=self.paths[f"path_{split}_pred"],
            gold_path=self.paths[f"path_{split}_gold"],
            inkb=True
        ))

        if get_scores_only:
            return scores

        # Save the evaluation scores
        utils.write_json(self.paths[f"path_{split}_eval"], scores)
        logger.info(utils.pretty_format_dict(scores))
        return scores
