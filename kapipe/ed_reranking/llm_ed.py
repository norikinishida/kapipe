from __future__ import annotations

from collections import defaultdict
import copy
import json
import logging
import os
import re
from typing import Any

import torch
from tqdm import tqdm

from .. import evaluation
from .. import utils
from ..datatypes import (
    Document,
    Mention,
    Entity,
    EntityPage,
    CandidateEntitiesForDocument,
)
from ..llms import HuggingFaceLLM, OpenAILLM
from ..resources import resolve_snapshot_path
from .base import BaseEDReranker


logger = logging.getLogger(__name__)


N_CAND = 3
N_MENT_PER_CHUNK = 5


class LLMED(BaseEDReranker):

    @classmethod
    def from_identifier(
        cls,
        model: HuggingFaceLLM | OpenAILLM,
        identifier: str,
    ) -> "LLMED":

        # Resolve the public identifier to the corresponding local snapshot
        snapshot_path = resolve_snapshot_path(
            component_name="ed_reranking",
            method_name="llm_ed",
            identifier=identifier,
        )

        # Load the reranker from the resolved snapshot
        reranker = cls.from_snapshot(
            model=model,
            snapshot_path=snapshot_path,
        )

        # Store the public identifier for later inspection
        reranker.identifier = identifier

        return reranker

    @classmethod
    def from_snapshot(
        cls,
        model: HuggingFaceLLM | OpenAILLM,
        snapshot_path: str,
    ) -> "LLMED":

        # Define the default paths for the resources in the snapshot
        component_config_path = snapshot_path + "/component_config.json"
        entity_dict_path = snapshot_path + "/entity_dict.json"
        demonstration_documents_path = (
            snapshot_path + "/demonstration_documents.json"
        )
        demonstration_candidate_entities_path = (
            snapshot_path + "/demonstration_candidate_entities.json"
        )

        # Set the demonstration documents to None if the file does not exist
        if not os.path.exists(demonstration_documents_path):
            demonstration_documents_path = None
        if not os.path.exists(demonstration_candidate_entities_path):
            demonstration_candidate_entities_path = None

        # Load the component configuration
        component_config = utils.read_json(component_config_path)
        logger.info(f"Loaded component configuration from {component_config_path}")
        logger.info(utils.pretty_format_dict(component_config))

        # Initialize the reranker from explicit snapshot resources
        reranker = cls(
            # External
            model=model,
            # Internal
            **component_config,
            entity_dict_path=entity_dict_path,
            # Optional (Internal)
            demonstration_documents=demonstration_documents_path,
            demonstration_candidate_entities=demonstration_candidate_entities_path,
        )

        # Store the snapshot path for later inspection
        reranker.snapshot_path = snapshot_path

        return reranker

    def __init__(
        self,
        # External
        model: HuggingFaceLLM | OpenAILLM,
        # Internal
        knowledge_base_name: str,
        entity_dict_path: str,
        prompt_template_name_or_path: str = "ed_12_zeroshot",
        # Optional (Internal)
        demonstration_documents: list[Document] | str | None = None,
        demonstration_candidate_entities: (
            list[CandidateEntitiesForDocument] | str | None
        ) = None,
        **unused_kwargs: object,
    ):
        logger.info("########## LLMED Initialization Starts ##########")

        self.model = model
        self.prompt_template_name_or_path = prompt_template_name_or_path
        self.knowledge_base_name = knowledge_base_name

        # Load the entity dictionary
        logger.info(f"Loading entity dictionary from {entity_dict_path}")
        self.entity_dict = {
            epage["entity_id"]: epage
            for epage in utils.read_json(entity_dict_path)
        }
        logger.info(f"Completed loading of entity dictionary with {len(self.entity_dict)} entities from {entity_dict_path}")

        # Load the demonstration documents
        if isinstance(demonstration_documents, str):
            demonstration_documents_path = demonstration_documents
            demonstration_documents = utils.read_json(demonstration_documents_path)
            logger.info(
                f"Loaded {len(demonstration_documents)} demonstration documents "
                f"from {demonstration_documents_path}"
            )
        elif demonstration_documents is None:
            # Use an empty list for zero-shot setting
            demonstration_documents = []
        self.demonstration_documents: list[Document] = demonstration_documents

        # Load demonstration candidate entities
        if isinstance(demonstration_candidate_entities, str):
            demonstration_candidate_entities_path = demonstration_candidate_entities
            demonstration_candidate_entities = utils.read_json(
                demonstration_candidate_entities_path
            )
            logger.info(
                "Loaded "
                f"{len(demonstration_candidate_entities)} "
                "demonstration candidate-entity records "
                f"from {demonstration_candidate_entities_path}"
            )
        elif demonstration_candidate_entities is None:
            # Use an empty list for zero-shot setting
            demonstration_candidate_entities = [] 
        self.demonstration_candidate_entities: list[CandidateEntitiesForDocument] = (
            demonstration_candidate_entities
        )

        # Validate the lengths of demonstration documents and demonstration
        # candidate entities.
        if (
            len(self.demonstration_documents)
            != len(self.demonstration_candidate_entities)
        ):
            raise ValueError(
                "demonstration_documents and demonstration_candidate_entities "
                "must have the same length."
            )

        # Load the prompt template
        self.prompt_template = utils.read_prompt_template(
            prompt_template_name_or_path=self.prompt_template_name_or_path,
            prompt_template_package_name="kapipe.ed_reranking.prompt_templates",
        )

        # Validate the prompt template
        if "{knowledge_base_name_prompt}" not in self.prompt_template:
            raise ValueError(
                "The prompt template must contain "
                "{knowledge_base_name_prompt}."
            )
        if "{test_case_prompt}" not in self.prompt_template:
            raise ValueError(
                "The prompt template must contain {test_case_prompt}."
            )

        # Set the prompt section for the knowledge base name
        self.knowledge_base_name_prompt = self.knowledge_base_name

        # Generate the prompt section for demonstrations
        self.demonstrations_prompt = self.generate_demonstrations_prompt()

        logger.info("########## LLMED Initialization Ends ##########")

    def save(self, snapshot_path: str) -> None:
        """Save the configuration, entity dictionary, demonstration documents, and demonstration candidate entities to a snapshot directory."""

        component_config_path = snapshot_path + "/component_config.json"
        entity_dict_path = snapshot_path + "/entity_dict.json"
        demonstration_documents_path = (
            snapshot_path + "/demonstration_documents.json"
        )
        demonstration_candidate_entities_path = (
            snapshot_path + "/demonstration_candidate_entities.json"
        )

        component_config: dict[str, Any] = {
            "prompt_template_name_or_path": self.prompt_template_name_or_path,
            "knowledge_base_name": self.knowledge_base_name,
        }

        utils.write_json(component_config_path, component_config)
        utils.write_json(entity_dict_path, list(self.entity_dict.values()))
        utils.write_json(
            demonstration_documents_path,
            self.demonstration_documents
        )
        utils.write_json(
            demonstration_candidate_entities_path,
            self.demonstration_candidate_entities
        )

    def rerank(
        self,
        document: Document,
        candidate_entities_for_doc: CandidateEntitiesForDocument,
    ) -> Document:
        """Rerank candidate entities for a single document."""

        with torch.no_grad():
            # Switch to inference mode for Hugging Face models
            if self.model.provider == "hf":
                self.model.llm.eval()
 
            # Split mentions into groups and perform reranking on the groups iteratively
            prompt_list: list[str] = []
            generated_text_list: list[str] = []
            target_mentions_list: list[list[Mention]] = []
            indices = list(range(0, len(document["mentions"])))

            for m_i in range(0, len(document["mentions"]), N_MENT_PER_CHUNK):
                # Get mention indices for this group
                target_mention_indices = indices[m_i: m_i + N_MENT_PER_CHUNK]

                # Generate the prompt
                prompt = self.generate_prompt(
                    document=document,
                    candidate_entities_for_doc=candidate_entities_for_doc,
                    target_mention_indices=target_mention_indices,
                )
                prompt_list.append(prompt)

                # Generate a response
                generated_text = self.model.generate(prompt)
                generated_text_list.append(generated_text)

                # Parse the generated text into mention-level records
                target_mentions = self.parse(
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

    def parse(
        self,
        document: Document,
        candidate_entities_for_doc: CandidateEntitiesForDocument,
        generated_text: str,
        target_mention_indices: list[int]
    ) -> list[Mention]:
        """Parse the generated text into mention-level entity records."""

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

        # Parse the generated response as a JSON array
        records = utils.safe_json_loads(
            generated_text=generated_text,
            fallback=[],
            list_type=True,
        )

        # Extract valid mention names and entity IDs
        names: list[str] = []
        entity_ids: list[str] = []
        for record in records:
            # Skip malformed array elements
            if not isinstance(record, dict):
                logger.warning(
                    "[%s] Skipped an entity assignment that is not a JSON "
                    "object: %s",
                    doc_key,
                    record,
                )
                continue

            # Skip records that do not contain all required keys
            required_keys = {"mention", "entity_id"}
            if not required_keys.issubset(record.keys()):
                logger.warning(
                    "[%s] Skipped an entity assignment with missing fields: %s",
                    doc_key,
                    record,
                )
                continue

            # Skip records with a non-string mention
            name = record["mention"]
            if not isinstance(name, str):
                logger.warning(
                    "[%s] Skipped an entity assignment with a non-string "
                    "mention: %s",
                    doc_key,
                    record,
                )
                continue
            name = name.strip()

            # Skip records with a non-string entity ID
            entity_id = record["entity_id"]
            if not isinstance(entity_id, str):
                logger.warning(
                    "[%s] Skipped an entity assignment with a non-string "
                    "entity ID: %s",
                    doc_key,
                    record,
                )
                continue
            entity_id = entity_id.strip()

            # Add the parsed fields
            names.append(name)
            entity_ids.append(entity_id)

        # Assign entity IDs by output order when every target mention is present
        if len(names) == len(entity_ids) == len(target_mention_indices):
            # The number of entities (i.e., len(names), len(entity_ids)) is the same
            # with that of all mentions in the document.
            if len(document["mentions"]) == 0:
                assert len(target_mention_indices) == 0
            else:
                for m_i, (name, entity_id) in enumerate(zip(names, entity_ids)):
                    # Skip checking the mention names
    
                    # Check whether the entity ID can be found in the possible list
                    if entity_id not in possible_entity_ids:
                        logger.info(f"[{doc_key}] Skipped a generated record with invalid concept ID: {entity_id}")
                        continue
    
                    # Add mention
                    mentions[target_mention_indices[m_i]]["entity_id"] = entity_id

        # Match records by mention name when the output length is unexpected
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
                    logger.info(f"[{doc_key}] Skipped a generated record with invalid mention: '{normalized_name}' not in {list(normalized_name_to_mention_indices.keys())}")
                    continue
                normalized_name = normalized_name2

                # Check whether the entity ID can be found in the possible list
                if entity_id not in possible_entity_ids:
                    logger.info(f"[{doc_key}] Skipped a generated record with invalid concept ID: {entity_id}")
                    continue

                # Add mention
                mention_indices = normalized_name_to_mention_indices[normalized_name]
                for m_i in mention_indices:
                    mentions[m_i]["entity_id"] = entity_id

        # Check that non-target mentions remain unchanged
        for m_i in range(len(document["mentions"])):
            if m_i not in target_mention_indices:
                assert mentions[m_i]["entity_id"] == "NO-PRED"

        return [mentions[m_i] for m_i in target_mention_indices]

    def generate_prompt(
        self,
        document: Document,
        candidate_entities_for_doc: CandidateEntitiesForDocument,
        target_mention_indices: list[int],
    ) -> str:
        """Generate the prompt for the LLM based on the input document, candidate entities."""

        # Create candidate entity pages for the input document
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

        # Generate the prompt section for the test case
        test_case_prompt = self.generate_test_case_prompt(
            document=document,
            candidate_entity_pages_for_doc=candidate_entity_pages_for_doc,
            target_mention_indices=target_mention_indices
        )

        # Combine all the prompt sections
        prompt = self.prompt_template.format(
            knowledge_base_name_prompt=self.knowledge_base_name_prompt,
            demonstrations_prompt=self.demonstrations_prompt,
            test_case_prompt=test_case_prompt
        )

        return prompt

    def generate_demonstrations_prompt(self) -> str:
        """Generate the prompt for the demonstration documents."""

        # Create candidate entity pages for the demonstration documents
        candidate_entity_pages_for_demos: list[list[list[EntityPage]]] = []
        for candidate_entities_for_demo in (
            self.demonstration_candidate_entities
        ):
            candidate_entity_pages_for_demo: list[list[EntityPage]] = []
            for candidate_entities_for_one_mention in (
                candidate_entities_for_demo["candidate_entities"]
            ):
                candidate_entity_pages_for_one_mention: list[EntityPage] = [
                    self.entity_dict[candidate["entity_id"]]
                    for candidate in candidate_entities_for_one_mention
                ]
                candidate_entity_pages_for_demo.append(
                    candidate_entity_pages_for_one_mention
                )
            candidate_entity_pages_for_demos.append(
                candidate_entity_pages_for_demo
            )

        # Generate the prompt section for demonstrations
        prompt = ""
        n_demos = len(self.demonstration_documents)
        for demo_i, (demo_doc, cand_ent_pages_for_demo) in enumerate(zip(
            self.demonstration_documents,
            candidate_entity_pages_for_demos
        )):
            # Sample target mention indices
            target_mention_indices = [
                m_i for m_i, m in enumerate(demo_doc["mentions"])
                if m["entity_id"] in self.entity_dict
            ]

            # Generate the prompt section for the mentions and their candidate concepts
            mention_candidates_pairs_prompt = (
                self.generate_input_mention_candidates_pairs_prompt(
                    document=demo_doc, 
                    candidate_entity_pages_for_doc=cand_ent_pages_for_demo,
                    target_mention_indices=target_mention_indices[:2],
                    demonstration_mode=True
                )
            )

            # Example ID
            prompt += f"Example {demo_i+1}:\n"

            # Input
            prompt += "Input Text:\n"
            prompt += f"{self.generate_input_text_prompt(document=demo_doc)}\n"
            prompt += "Mentions and Candidate Concepts:\n"
            prompt += f"{mention_candidates_pairs_prompt}\n"

            # Output
            prompt += "Output:\n"
            output_prompt = self.generate_output_prompt(
                document=demo_doc,
                candidate_entity_pages_for_doc=cand_ent_pages_for_demo,
                target_mention_indices=target_mention_indices[:2]
            )
            prompt += f"{output_prompt}\n"

            if demo_i < n_demos - 1:
                prompt += "\n"

        return prompt.rstrip()

    def generate_test_case_prompt(
        self,
        document: Document,
        candidate_entity_pages_for_doc: list[list[EntityPage]],
        target_mention_indices: list[int]
    ) -> str:
        """Generate the prompt for the test case based on the input document, candidate entity pages, and target mention indices."""

        # Generate the prompt section for the mentions and their candidate concepts 
        mention_candidates_pairs_prompt = (
            self.generate_input_mention_candidates_pairs_prompt(
                document=document, 
                candidate_entity_pages_for_doc=candidate_entity_pages_for_doc,
                target_mention_indices=target_mention_indices,
                demonstration_mode=False
            )
        )

        prompt = ""

        # Input
        prompt += "Input Text:\n"
        prompt += f"{self.generate_input_text_prompt(document=document)}\n"
        prompt += "Mentions and Candidate Concepts:\n"
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

        # Convert the gold entity assignments into JSON records
        records: list[dict[str, str]] = []
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

                records.append({
                    "mention": name,
                    "entity_id": entity_id,
                })

        # Serialize the records as a JSON array
        return json.dumps(records, ensure_ascii=False, indent=4)

    def submit_batch(
        self,
        documents: list[Document],
        candidate_entities: list[CandidateEntitiesForDocument],
    ) -> list[str]:
        """Submit ED-reranking prompts and return the OpenAI Batch IDs.

        Pass the same documents and candidate entities in the same order to
        fetch_and_process_batch(). Keep the model settings and prompt template
        unchanged between calls.
        """

        # Validate that the model is an OpenAILLM instance for Batch API usage
        if not isinstance(self.model, OpenAILLM):
            raise TypeError("Batch API requires OpenAILLM")

        # Validate that there is one candidate entity record for every document
        if len(documents) != len(candidate_entities):
            raise ValueError(
                "The number of documents does not match "
                "the number of candidate entity records"
            )

        # Generate prompts using the same method as rerank()
        prompts: list[str] = []
        for document, candidate_entities_for_doc in zip(
            documents,
            candidate_entities,
            strict=True,
        ):
            # Validate that the document and candidate entities match
            if document["doc_key"] != candidate_entities_for_doc["doc_key"]:
                raise ValueError("Document and candidate entity keys do not match")

            # Split mentions into groups
            indices = list(range(0, len(document["mentions"])))
            for m_i in range(0, len(document["mentions"]), N_MENT_PER_CHUNK):
                # Get mention indices for this group
                target_mention_indices = indices[m_i: m_i + N_MENT_PER_CHUNK]

                # Generate the prompt
                prompt = self.generate_prompt(
                    document=document,
                    candidate_entities_for_doc=candidate_entities_for_doc,
                    target_mention_indices=target_mention_indices,
                )
                prompts.append(prompt)

        # Validate that there is at least one prompt to submit
        if len(prompts) == 0:
            raise ValueError(
                "No prompts to submit because all documents have no mentions"
            )

        # Submit the prompts and get the Batch IDs
        batch_ids: list[str] = self.model.submit_batch(prompts=prompts)
        return batch_ids

    def fetch_and_process_batch(
        self,
        documents: list[Document],
        candidate_entities: list[CandidateEntitiesForDocument],
        batch_ids: list[str],
    ) -> list[Document]:
        """Fetch responses and rerank mentions in the original documents.

        Require the same documents, candidate entities, order, model settings,
        and prompt template used at submission. Raise an error if the batch
        is not complete.
        """

        # Validate that the model is an OpenAILLM instance for Batch API usage
        if not isinstance(self.model, OpenAILLM):
            raise TypeError("Batch API requires OpenAILLM")

        # Validate that there is one candidate entity record for every document
        if len(documents) != len(candidate_entities):
            raise ValueError(
                "The number of documents does not match "
                "the number of candidate entity records"
            )

        # Record the document for each submitted request
        request_document_indices: list[int] = []
        for document_i, document in enumerate(documents):
            for _ in range(0, len(document["mentions"]), N_MENT_PER_CHUNK):
                request_document_indices.append(document_i)

        # Fetch generated texts in the original request order
        generated_texts: list[str] = self.model.fetch_batch(batch_ids=batch_ids)

        # Validate that the number of generated texts matches the number of requests
        if len(generated_texts) != len(request_document_indices):
            raise ValueError(
                "The response count does not match the submitted request count"
            )

        # Group the responses by their original documents
        generated_texts_by_document: list[list[str]] = [
            [] for _ in documents
        ]
        for document_i, generated_text in zip(
            request_document_indices,
            generated_texts,
            strict=True,
        ):
            generated_texts_by_document[document_i].append(generated_text)

        # Process each Batch response using the same procedure as rerank()
        result_documents: list[Document] = []
        for document, candidate_entities_for_doc, generated_text_list in zip(
            documents,
            candidate_entities,
            generated_texts_by_document,
            strict=True,
        ):
            # Validate that the document and candidate entities match
            if document["doc_key"] != candidate_entities_for_doc["doc_key"]:
                raise ValueError("Document and candidate entity keys do not match")

            # Split mentions into groups and perform reranking on the groups iteratively
            prompt_list: list[str] = []
            target_mentions_list: list[list[Mention]] = []
            indices = list(range(0, len(document["mentions"])))

            for m_i, generated_text in zip(
                range(0, len(document["mentions"]), N_MENT_PER_CHUNK),
                generated_text_list,
                strict=True,
            ):
                # Get mention indices for this group
                target_mention_indices = indices[m_i: m_i + N_MENT_PER_CHUNK]

                # Generate the prompt
                prompt = self.generate_prompt(
                    document=document,
                    candidate_entities_for_doc=candidate_entities_for_doc,
                    target_mention_indices=target_mention_indices,
                )
                prompt_list.append(prompt)

                # Parse the generated text into mention-level records
                target_mentions = self.parse(
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
            result_documents.append(result_document)

        return result_documents


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
        paths["snapshot_path"] = self.base_output_path

        # evaluation outputs
        paths["dev_gold_path"] = self.base_output_path + "/dev.gold.json"
        paths["dev_pred_path"] = self.base_output_path + "/dev.pred.json"
        paths["dev_eval_path"] = self.base_output_path + "/dev.eval.json"
        paths["test_gold_path"] = self.base_output_path + "/test.gold.json"
        paths["test_pred_path"] = self.base_output_path + "/test.pred.json"
        paths["test_eval_path"] = self.base_output_path + "/test.eval.json"

        return paths

    def setup_dataset(
        self,
        reranker: LLMED,
        documents: list[Document],
        candidate_entities: list[CandidateEntitiesForDocument],
        split: str
    ) -> None:
        # Cache the gold annotations for evaluation
        gold_path = self.paths[f"{split}_gold_path"]
        if not os.path.exists(gold_path):
            kb_entity_ids = set(list(reranker.entity_dict.keys()))
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
            utils.write_json(gold_path, gold_documents)
            logger.info(f"Saved the gold annotations for evaluation in {gold_path}")

    def save_reranker(self, reranker: LLMED) -> None:
        reranker.save(snapshot_path=self.paths["snapshot_path"])

    def evaluate(
        self,
        reranker: LLMED,
        documents: list[Document],
        candidate_entities: list[CandidateEntitiesForDocument],
        split: str,
        #
        prediction_only: bool = False,
        get_scores_only: bool = False
    ) -> dict[str, Any]:
        # Apply the reranker
        result_documents: list[Document] = []
        for (document, candidate_entities_for_doc) in tqdm(
                zip(documents, candidate_entities),
                total=len(documents),
                desc="reranking steps"
            ):
            result_document = reranker.rerank(
                document=document,
                candidate_entities_for_doc=candidate_entities_for_doc,
            )
            result_documents.append(result_document)

        # Save the prediction results
        utils.write_json(self.paths[f"{split}_pred_path"], result_documents)

        # Save the prompt-response pairs in plain text
        with open(
            self.paths[f"{split}_pred_path"].replace(".json", ".txt"), "w"
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
            pred_path=self.paths[f"{split}_pred_path"],
            gold_path=self.paths[f"{split}_gold_path"],
            inkb=True
        )
        scores.update(evaluation.ed.fscore(
            pred_path=self.paths[f"{split}_pred_path"],
            gold_path=self.paths[f"{split}_gold_path"],
            inkb=True
        ))

        if get_scores_only:
            return scores

        # Save the evaluation scores
        utils.write_json(self.paths[f"{split}_eval_path"], scores)
        logger.info(utils.pretty_format_dict(scores))
        return scores
