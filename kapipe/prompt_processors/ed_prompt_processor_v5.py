from collections import defaultdict
import logging
import os
# import random
import re

# from .. import utils


logger = logging.getLogger(__name__)


class EDPromptProcessorV5:

    def __init__(
        self,
        prompt_template_name_or_path,
        knowledge_base_name_prompt
    ):
        """
        Parameters
        ----------
        prompt_template_name : str
        knowledge_base_name_prompt: str
        """
        self.prompt_template_name_or_path = prompt_template_name_or_path
        self.knowledge_base_name_prompt = knowledge_base_name_prompt

        # Load prompt template
        path_prompt_templates_dir = os.path.join(
            os.path.dirname(__file__),
            "prompt_templates"
        )
        prompt_template_names = os.listdir(path_prompt_templates_dir)
        prompt_template_names = [
            os.path.splitext(name)[0] for name in prompt_template_names
        ]
        if self.prompt_template_name_or_path in prompt_template_names:
            with open(os.path.join(
                path_prompt_templates_dir,
                self.prompt_template_name_or_path + ".txt"
            )) as f:
                self.prompt_template = f.read()
        else:
            with open(self.prompt_template_name_or_path) as f:
                self.prompt_template = f.read()
        assert "{knowledge_base_name_prompt}" in self.prompt_template
        assert "{demonstrations_prompt}" in self.prompt_template
        assert "{test_prompt}" in self.prompt_template

        # Output format
        # <bullet> <mention> -> <entity id>
        self.re_comp = re.compile("(.+?)\s+(.+?)\s*->\s*(.+?)$")

    #####
    # Encoding
    #####

    def encode(
        self,
        document,
        candidate_entity_dicts_for_doc,
        demonstration_documents,
        candidate_entity_dicts_for_demos
    ):
        """
        Parameters
        ----------
        document : Document
        candidate_entity_dicts_for_doc: list[list[EntityPage]]
            shape of (n_mentions, n_candidates)
        demonstration_documents : list[Document]
            shape of (n_demos,)
        candidate_entity_dicts_for_demos: list[list[list[EntityPage]]]
            shape of (n_demos, n_mentions, n_candidates)

        Returns
        -------
        str
        """
        demonstrations_prompt = self.generate_demonstrations_prompt(
            demonstration_documents=demonstration_documents,
            candidate_entity_dicts_for_demos=candidate_entity_dicts_for_demos
        )
        test_prompt = self.generate_test_prompt(
            document=document,
            candidate_entity_dicts_for_doc=candidate_entity_dicts_for_doc
        )
        prompt = self.prompt_template.format(
            knowledge_base_name_prompt=self.knowledge_base_name_prompt,
            demonstrations_prompt=demonstrations_prompt,
            test_prompt=test_prompt
        )
        return prompt

    #####
    # Subfunctions for encoding
    #####

    def generate_demonstrations_prompt(
        self,
        demonstration_documents,
        candidate_entity_dicts_for_demos
    ):
        """
        Parameters
        ----------
        demonstration_documents: list[Document]
            shape of (n_demos,)
        candidate_entity_dicts_for_demos: list[list[list[EntityPage]]]
            shape of (n_demos, n_mentions, n_candidates)

        Returns
        -------
        str
        """
        text = ""
        n_demos = len(demonstration_documents)
        for demo_i, (demo_doc, cand_ent_dicts_for_demo) in enumerate(
            zip(
                demonstration_documents,
                candidate_entity_dicts_for_demos
            )
        ):
            text += f"# Example {demo_i+1}\n"
            text += (
                "Text: "
                + self.generate_input_prompt(document=demo_doc)
                + "\n"
            )
            mentions_text, selected_mention_indices = \
                self.generate_mentions_prompt(document=demo_doc, demo=True)
            text += (
                "Mentions:\n"
                + mentions_text
                + "\n"
            )
            text += (
                "Concept IDs (candidates):\n"
                + self.generate_candidate_entities_prompt(
                    candidate_entity_dicts_for_doc=cand_ent_dicts_for_demo,
                    selected_mention_indices=selected_mention_indices
                )
                + "\n"
            )
            text += (
                "Answer:\n"
                + self.generate_output_prompt(
                    document=demo_doc,
                    selected_mention_indices=selected_mention_indices
                )
                + "\n"
            )
            if demo_i < n_demos - 1:
                text += "\n"
        return text.rstrip()

    def generate_test_prompt(
        self,
        document,
        candidate_entity_dicts_for_doc
    ):
        """
        Parameters
        ----------
        document : Document
        candidate_entity_dicts_for_doc : list[list[EntityPage]]
            shape of (n_mentions, n_candidates)

        Returns
        -------
        str
        """
        text = ""
        text += "# Test Example\n"
        text += (
            "Text: "
            + self.generate_input_prompt(document=document)
            + "\n"
        )
        mentions_text, selected_mention_indices = \
            self.generate_mentions_prompt(document=document)
        text += (
            "Mentions:\n"
            + mentions_text
            + "\n"
        )
        text += (
            "Concept IDs (candidates):\n"
            + self.generate_candidate_entities_prompt(
                candidate_entity_dicts_for_doc=candidate_entity_dicts_for_doc,
                selected_mention_indices=selected_mention_indices
            )
        )
        return text.rstrip()

    def generate_input_prompt(self, document):
        """
        Parameters
        ----------
        document : Document

        Returns
        -------
        str
        """
        text = " ".join(document["sentences"]) + "\n"
        return text.rstrip()

    def generate_mentions_prompt(self, document, demo=False):
        """
        Parameters
        ----------
        document : Document
        demo : bool
            by default False

        Returns
        -------
        tuple[str, list[int]]
        """
        text = ""
        words = " ".join(document["sentences"]).split()
        names = []
        selected_mention_indices = []
        for m_i, mention in enumerate(document["mentions"]):
            begin_i, end_i = mention["span"]
            name = " ".join(words[begin_i: end_i + 1])
            if name in names:
                continue
            names.append(name)
            selected_mention_indices.append(m_i)
            # In the demonstrations, we skip some mentions
            if demo and len(names) >= 3:
                break
        for n_i, name in enumerate(names):
            text += f"{n_i + 1}. {name}\n"
        return text.rstrip(), selected_mention_indices

    def generate_candidate_entities_prompt(
        self,
        candidate_entity_dicts_for_doc,
        selected_mention_indices=None
    ):
        """
        Parameters
        ----------
        candidate_entity_dicts_for_doc : list[list[EntityPage]]
        selected_mention_indices : list[int] | None
            by default None

        Returns
        -------
        str
        """
        N_CAND = 3
        # Aggregate candidates as a single list
        candidates = []
        memorized_ids = set()
        for m_i, candidate_entity_dicts_for_one_mention in (
            enumerate(candidate_entity_dicts_for_doc)
        ):
            if (
                (selected_mention_indices is not None)
                and
                (not m_i in selected_mention_indices)
            ):
                continue
            for cand_dict in candidate_entity_dicts_for_one_mention[:N_CAND]:
                if not cand_dict["entity_id"] in memorized_ids:
                    candidates.append(cand_dict)
                    memorized_ids.add(cand_dict["entity_id"])
        # Transform the candidate list into text
        text = ""
        for cand_dict in candidates:
            entity_id = cand_dict["entity_id"]
            canonical_name = cand_dict["canonical_name"]
            # desc = cand_dict["description"]
            text += f"* {entity_id}: {canonical_name}\n"
        return text.rstrip()

    def generate_output_prompt(self, document, selected_mention_indices=None):
        """
        Parameters
        ----------
        document : Document
        selected_mention_indices : list[int] | None
            by default NOne

        Returns
        -------
        str
        """
        text = ""
        words = " ".join(document["sentences"]).split()
        for m_i, mention in enumerate(document["mentions"]):
            if (
                (selected_mention_indices is not None)
                and
                (not m_i in selected_mention_indices)
            ):
                continue
            begin_i, end_i = mention["span"]
            name = " ".join(words[begin_i : end_i + 1])
            entity_id = mention["entity_id"]
            text += f"{m_i+1}. {name} -> {entity_id}\n"
        return text.rstrip()

    #####
    # Decoding
    #####

    def decode(self, generated_text, document):
        """
        Parameters
        ----------
        generated_text : str
        document : Document

        Returns
        -------
        list[Mention]
        """
        mention_names = [] # list[str]
        # words = utils.flatten_lists([s.split() for s in document["sentences"]])
        words = " ".join(document["sentences"]).split()
        for mention in document["mentions"]:
            b_i, e_i = mention["span"]
            name = " ".join(words[b_i: e_i+1])
            mention_names.append(name)

        n_mentions = len(mention_names)

        mentions = []
        for _ in range(n_mentions):
            mentions.append(
                {
                    "entity_id": "NO-PRED",
                    # "corresponding_output_line": None
                }
            )

        # Variables shared across generated lines
        normalized_name_to_mention_indices = defaultdict(list)
        for m_i, mention_name in enumerate(mention_names):
            normalized_name = mention_name.lower()
            normalized_name_to_mention_indices[normalized_name].append(m_i)

        # We process each generated line
        generated_lines = generated_text.split("\n")
        memory = set()
        for generated_line in generated_lines:
            generated_line = generated_line.strip()
            if generated_line == "":
                continue

            # Parse the generated line
            parsed = self.re_comp.findall(generated_line)
            if not (len(parsed) == 1 and len(parsed[0]) == 3):
                logger.info(f"Skipped a generated line of invalid formatting: '{generated_line}'")
                continue
            _, name, entity_id = parsed[0]

            # Check whether the mention can be found in the possible list
            normalized_name = name.lower()
            if not normalized_name in normalized_name_to_mention_indices:
                logger.info(f"Skipped a generated line with invalid mention name: '{generated_line}'")
                continue
            if normalized_name in memory:
                logger.info(f"Skipped a generated line because the mention is already linked: '{generated_line}'")
                continue
            memory.add(normalized_name)

            # We do not check whether the entity ID can be found in the possible list

            # Clean the entity ID prediction
            # parsed2 = self.re_comp2.findall(entity_id)
            # if len(parsed2) > 0:
            #     before_ = entity_id
            #     entity_id, _ = parsed2[0]
            #     after_ = entity_id
            #     logger.info(
            #         f"Cleaned the entity ID prediction: {before_} -> {after_}"
            #     )

            # Add mention
            mention_indices \
                = normalized_name_to_mention_indices[normalized_name]
            for m_i in mention_indices:
                mentions[m_i]["entity_id"] = entity_id
                # mentions[m_i]["corresponding_output_line"] \
                #     = generated_line

        return mentions

