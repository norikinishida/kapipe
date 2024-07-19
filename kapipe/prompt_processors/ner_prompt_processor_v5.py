import logging
import os
import re

# from .. import utils


logger = logging.getLogger(__name__)


class NERPromptProcessorV5:

    def __init__(
        self,
        prompt_template_name_or_path,
        possible_entity_types
    ):
        """
        Parameters
        ----------
        prompt_template_name : str
        possible_entity_types : list[str]
        """
        self.prompt_template_name_or_path = prompt_template_name_or_path
        self.possible_entity_types = possible_entity_types

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
        assert "{entity_types_prompt}" in self.prompt_template
        assert "{demonstrations_prompt}" in self.prompt_template
        assert "{test_prompt}" in self.prompt_template

        self.entity_types_prompt = ", ".join(self.possible_entity_types)

        self.normalized_to_canonical = {
            etype.lower(): etype for etype in possible_entity_types
        }

        # Output format
        # <bullet> <mention> -> <entity type>
        self.re_comp = re.compile("(.+?)\s+(.+?)\s*->\s*(.+?)$")

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
        ) # str
        test_prompt = self.generate_test_prompt(
            document=document
        ) # str
        prompt = self.prompt_template.format(
            entity_types_prompt=self.entity_types_prompt,
            demonstrations_prompt=demonstrations_prompt,
            test_prompt=test_prompt
        )
        return prompt

    #####
    # Subfunctions for encoding
    #####

    def generate_demonstrations_prompt(self, demonstration_documents):
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
                "Answer:\n"
                + self.generate_output_prompt(document=demo_doc)
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
        words = " ".join(document["sentences"]).split()
        names = []
        entity_types = []
        for m_i, mention in enumerate(document["mentions"]):
            begin_i, end_i = mention["span"]
            name = " ".join(words[begin_i: end_i + 1])
            if name in names:
                continue
            entity_type = mention["entity_type"]
            names.append(name)
            entity_types.append(entity_type)
        for n_i, (name, entity_type) in enumerate(zip(names, entity_types)):
            text += f"{n_i+1}. {name} -> {entity_type}\n"
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
        # Variables shared across generated lines
        # words = utils.flatten_lists([s.split() for s in document["sentences"]])
        words = " ".join(document["sentences"]).split()
        normalized_text = " ".join(words).lower()
        char_index_to_word_index = [] # list[int]
        for w_i, w in enumerate(words):
            n_chars = len(w)
            char_index_to_word_index.extend([w_i] * n_chars)
            char_index_to_word_index.append(None) # for space between words

        # We process each generated line
        generated_lines = generated_text.split("\n")
        tuples = [] # list[tuple[int, int, str, str]]
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
            _, name, entity_type = parsed[0]

            # Check whether the mention can be found in the input text
            # i.e., get word-level spans
            normalized_name = name.lower()
            spans = self.extract_word_level_spans(
                normalized_name=normalized_name,
                normalized_text=normalized_text,
                char_index_to_word_index=char_index_to_word_index
            )
            if len(spans) == 0:
                logger.info(f"Skipped a generated line with invalid mention: '{generated_line}'")
                continue

            # Check whether the entity type can be found in the possible list
            normalized_entity_type = entity_type.lower()
            if not normalized_entity_type in self.normalized_to_canonical:
                logger.info(f"Skipped a generated line with invalid entity type: '{generated_line}'")
                continue
            canonical_entity_type \
                = self.normalized_to_canonical[normalized_entity_type]

           # Add new tuples
            for begin_token_i, end_token_i in spans:
                tuple_core = (
                    begin_token_i,
                    end_token_i,
                    canonical_entity_type
                )
                tuple_with_line = (
                    begin_token_i,
                    end_token_i,
                    canonical_entity_type,
                    generated_line
                )
                if not tuple_core in memory:
                    memory.add(tuple_core)
                    tuples.append(tuple_with_line)

        # Convert tuples to mention dicts
        mentions = [] # list[Mention]
        for tuple_with_line in tuples:
            begin_i, end_i, etype, generated_line = tuple_with_line
            name = " ".join(words[begin_i: end_i + 1])
            mentions.append({
                "span": (begin_i, end_i),
                "name": name,
                "entity_type": etype,
                # "corresponding_output_line": generated_line,
            })
        mentions = sorted(mentions, key=lambda m: m["span"])
        return mentions

    #####
    # Subfunctions for decoding
    #####

    def extract_word_level_spans(
        self,
        normalized_name,
        normalized_text,
        char_index_to_word_index
    ):
        """
        Parameters
        ----------
        normalized_name : str
        normalized_text : str
        char_index_to_word_index : list[int]

        Returns
        -------
        list[tuple[int,int]]
        """
        spans = [] # list[tuple[int,int]]
        results = re.finditer(
            re.escape(" " + normalized_name + " "),
            " " + normalized_text + " "
        )
        for result in results:
            begin_char_i, end_char_i = result.span()
            # Remove the spaces around the span
            begin_char_i += 1
            end_char_i -= 1
            # Remove the space appended to the top of the text
            begin_char_i -= 1
            end_char_i -= 1
            begin_word_i = char_index_to_word_index[begin_char_i]
            end_word_i = char_index_to_word_index[end_char_i - 1]
            spans.append((begin_word_i, end_word_i))
        return spans

