import logging
import os
import re


logger = logging.getLogger(__name__)


class DocREPromptProcessorV4:

    def __init__(
        self,
        prompt_template_name_or_path,
        entity_dict,
        rel_name_to_pretty_rel_name,
        knowledge_base_name_prompt,
        mention_style,
        with_span_annotation=True,
    ):
        """
        Parameters
        ----------
        prompt_template_name : str
        entity_dict : dict[str, EntityPage]
        rel_name_to_pretty_rel_name: dict[str, str]
        knowledge_base_name_prompt: str
        mention_style: str
        with_span_annotation : bool, default True
        """
        self.prompt_template_name_or_path = prompt_template_name_or_path
        self.entity_dict = entity_dict
        self.rel_name_to_pretty_rel_name = rel_name_to_pretty_rel_name
        self.knowledge_base_name_prompt = knowledge_base_name_prompt
        self.mention_style = mention_style
        self.with_span_annotation = with_span_annotation

        assert self.mention_style in ["canonical_name", "first_mention", "all_mentions"]

        self.relations_prompt = ", ".join([
            v for k,v in rel_name_to_pretty_rel_name.items()
        ])

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
        assert "{relations_prompt}" in self.prompt_template
        assert "{demonstrations_prompt}" in self.prompt_template
        assert "{test_prompt}" in self.prompt_template

        # e.g., "chemical-induce-disease" -> "CID"
        self.normalized_to_canonical = {
            pretty_rel_name.lower(): rel
            for rel, pretty_rel_name in rel_name_to_pretty_rel_name.items()
        }

        # Output format
        # <bullet> (<head entity ID> , <relation> , <tail entity ID>)
        self.re_comp = re.compile(
            "(.+?)\s*\(\s*(.+?)\s*,\s*(.+?)\s*,\s*(.+?)\s*\)$"
        )

    #####
    # Encoding
    #####

    def encode(self, document, demonstration_documents):
        """
        Parameters
        ----------
        document : Document
        demonstration_documents : list[Document]

        Returns
        -------
        str
        """
        demonstrations_prompt = self.generate_demonstrations_prompt(
            demonstration_documents=demonstration_documents
        )
        test_prompt = self.generate_test_prompt(
            document=document,
        )
        prompt = self.prompt_template.format(
            knowledge_base_name_prompt=self.knowledge_base_name_prompt,
            relations_prompt=self.relations_prompt,
            demonstrations_prompt=demonstrations_prompt,
            test_prompt=test_prompt
        )
        return prompt

    #####
    # Subfunctions for encoding
    #####

    def generate_demonstrations_prompt(
        self,
        demonstration_documents
    ):
        """
        Parameters
        ----------
        demonstration_documents: list[Document]

        Returns
        -------
        str
        """
        text = ""
        n_demos = len(demonstration_documents)
        for demo_i, demo_doc in enumerate(demonstration_documents):
            text += f"# Example {demo_i+1}\n"
            text += (
                "Text: "
                + self.generate_input_prompt(document=demo_doc)
                + "\n"
            )
            text += (
                "Entities:\n"
                + self.generate_entities_prompt(document=demo_doc)
                + "\n"
            )
            text += (
                "Answer:\n"
                + self.generate_output_prompt(
                    document=demo_doc
                )
                + "\n"
            )
            if demo_i < n_demos - 1:
                text += "\n"
        return text.rstrip()

    def generate_test_prompt(self, document):
        """
        Parameters
        ----------
        document : Document

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
        text += (
            "Entities:\n"
            + self.generate_entities_prompt(document=document)
            + "\n"
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

    def generate_entities_prompt(self, document):
        """
        Parameters
        ----------
        document : Document

        Returns
        -------
        str
        """
        text = ""
        words = " ".join(document["sentences"]).split()
        mentions = document["mentions"]
        entities = document["entities"]
        for e_i, entity in enumerate(entities):
            entity_id = entity["entity_id"]
            if self.mention_style == "all_mentions":
                mention_indices = entity["mention_indices"]
                names = []
                for m_i in mention_indices:
                    mention = mentions[m_i]
                    if not self.with_span_annotation:
                        name = mention["name"]
                    else:
                        begin_i, end_i = mention["span"]
                        name = " ".join(words[begin_i: end_i + 1])
                    # Remove duplicated mentions
                    # (inserted after the BioNLP'24 submission)
                    if name in names:
                        continue
                    names.append(name)
                text += f"* {entity_id}: {str(names)}\n"
            elif self.mention_style == "first_mention":
                mention_indices = entity["mention_indices"]
                mention = mentions[mention_indices[0]]
                if not self.with_span_annotation:
                    name = mention["name"]
                else:
                    begin_i, end_i = mention["span"]
                    name = " ".join(words[begin_i: end_i + 1])
                text += f"* {entity_id}: {name}\n"
            elif self.mention_style == "canonical_name":
                epage = self.entity_dict[entity_id]
                name = epage["canonical_name"]
                text += f"* {entity_id}: {name}\n"
            else:
                raise Exception(f"Invalid mention_style: {self.mention_style}")
        return text.rstrip()

    def generate_output_prompt(self, document):
        """
        Parameters
        ----------
        document : Document

        Returns
        -------
        str
        """
        text = ""
        # words = " ".join(document["sentences"]).split()
        # pretty_rel_name = self.rel_name_to_pretty_rel_name[relation_type]
        entities = document["entities"]
        output_i = 0
        for triple in document["relations"]:
            head_idx = triple["arg1"]
            tail_idx = triple["arg2"]
            rel = triple["relation"]
            head_id = entities[head_idx]["entity_id"]
            tail_id = entities[tail_idx]["entity_id"]
            pretty_rel = self.rel_name_to_pretty_rel_name[rel]
            text += f"{output_i+1}. ({head_id}, {pretty_rel}, {tail_id})\n"
            output_i += 1
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
        list[Triple]
        """
        # Variables shared across generated lines
        entity_id_to_index = {}
        for e_i, e in enumerate(document["entities"]):
            e_id = e["entity_id"]
            entity_id_to_index[e_id] = e_i

        # We process each generated triple
        generated_lines = generated_text.split("\n")
        tuples = []
        memory = set()
        for generated_line in generated_lines:
            generated_line = generated_line.strip()
            if generated_line == "":
                continue

            # Parse the generated line
            parsed = self.re_comp.findall(generated_line)
            if not (len(parsed) == 1 and len(parsed[0]) == 4):
                logger.info(f"Skipped a generated line of invalid formatting: '{generated_line}'")
                continue
            _, head_id, relation, tail_id= parsed[0]

            # Check whether the head/tail IDs can be found in the possible list
            if (
                not head_id in entity_id_to_index
                or
                not tail_id in entity_id_to_index
                or
                head_id == tail_id
            ):
                logger.info(f"Skipped a generated line with invalid entity pair: '{generated_line}'")
                continue

            # Check whether the relation can be found in the possible set
            normalized_relation = relation.lower() # e.g., "Chemical-Induce-Disease" -> "chemical-induce-relation"
            if not normalized_relation in self.normalized_to_canonical:
                logger.info(f"Skipped a generated line with invalid relation: '{generated_line}'")
                continue
            canonical_relation \
                = self.normalized_to_canonical[normalized_relation] # e.g., "CID"

            # Get entity index
            head_idx = entity_id_to_index[head_id]
            tail_idx = entity_id_to_index[tail_id]

            # Add new tuples
            tuple_core = (head_idx, canonical_relation, tail_idx)
            tuple_with_line = (
               head_idx, canonical_relation, tail_idx, generated_line
            )
            if not tuple_core in memory:
                memory.add(tuple_core)
                tuples.append(tuple_with_line)

        # Convert tuples to triple dicts
        triples = [] # list[Triple]
        for tuple_with_line in tuples:
            arg1, rel, arg2, generated_line = tuple_with_line
            triples.append({
                "arg1": arg1,
                "relation": rel,
                "arg2": arg2,
                # "corresponding_output_line": generated_line,
            })

        triples = sorted(
            triples,
            key=lambda x: (x["arg1"], x["arg2"], x["relation"])
        )
        return triples
