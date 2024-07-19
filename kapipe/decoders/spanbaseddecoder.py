import numpy as np


class SpanBasedDecoder:
    """A span-based decoder for NER. This decoder applies constraints (for either Flat NER or Nested NER) to given scored spans and outputs a list of mentions.
    """

    def __init__(
        self,
        allow_nested_entities
    ):
        """
        Parameters
        ----------
        allow_nested_entities : bool
        """
        self.allow_nested_entities = allow_nested_entities

    def decode(
        self,
        spans,
        words
    ):
        """
        Parameters
        ----------
        spans : list[tuple[int, int, str, float]]
        words : list[str]

        Results
        -------
        list[Mention]
        """
        mentions = [] # list[Mention]

        # Sort the candidate spans based on their scores
        spans = sorted(spans, key=lambda x: -x[-1])

        # Select spans
        n_words = len(words)
        self.check_matrix = np.zeros((n_words, n_words)) # Used in Flat NER
        self.check_set = set() # Uased in Nested NER
        for span in spans:
            begin_token_index, end_token_index, etype, _ = span
            name = " ".join(words[begin_token_index: end_token_index + 1])
            if self.is_violation(
                begin_token_index=begin_token_index,
                end_token_index=end_token_index
            ):
                continue
            mentions.append({
                "span": (begin_token_index, end_token_index),
                "name": name,
                "entity_type": etype,
            })
            self.check_matrix[begin_token_index: end_token_index + 1] = 1
            self.check_set.add((begin_token_index, end_token_index))

        # Sort mentions based on the positions
        mentions = sorted(mentions, key=lambda m: m["span"])

        return mentions

    def is_violation(self, begin_token_index, end_token_index):
        """
        Parameters
        ----------
        begin_token_index : int
        end_token_index : int

        Returns
        -------
        bool
        """
        if not self.allow_nested_entities:
            # Flat NER
            if (
                self.check_matrix[begin_token_index: end_token_index + 1].sum()
                > 0
            ):
                return True
            else:
                return False
        else:
            # Nested NER
            for begin_token_j, end_token_j in self.check_set:
                if (
                    (begin_token_index < begin_token_j
                     <= end_token_index < end_token_j)
                    or
                    (begin_token_j < begin_token_index
                     <= end_token_j < end_token_index)
                ):
                    return True
            return False


