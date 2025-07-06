from collections import OrderedDict
from typing import NamedTuple

# import numpy as np
# import spacy_alignments as tokenizations
import torch
import torch.nn as nn
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers.modeling_outputs import ModelOutput

# from . import shared_functions
from .. import utils


class MentionTuple(NamedTuple):
    span: tuple[int, int]
    name: str
    entity_type: str
    entity_id: str | None


class EntityTuple(NamedTuple):
    mention_indices: list[int]
    entity_type: str
    entity_id: str


class BlinkCrossEncoderModel(nn.Module):

    def __init__(
        self,
        device,
        bert_pretrained_name_or_path,
        max_seg_len,
        entity_dict,
        mention_context_length
    ):
        """
        Parameters
        ----------
        device : str
        bert_pretrained_name_or_path : str
        max_seg_len : int
        entity_dict : dict[str, EntityPage]
        mention_context_length : int
        """
        super().__init__()

        ########################
        # Hyper parameters
        ########################

        self.device = device
        self.bert_pretrained_name_or_path = bert_pretrained_name_or_path
        self.max_seg_len = max_seg_len
        self.entity_dict = entity_dict
        self.mention_context_length = mention_context_length

        ########################
        # Components
        ########################

        # BERT, tokenizer
        self.bert, self.tokenizer = self._initialize_bert_and_tokenizer(
            pretrained_model_name_or_path=self.bert_pretrained_name_or_path
        )

        # Dimensionality
        self.hidden_dim = self.bert.config.hidden_size

        self.linear = nn.Linear(self.hidden_dim, 1)

        ######
        # Preprocessor
        ######

        self.preprocessor = BlinkCrossEncoderPreprocessor(
            tokenizer=self.tokenizer,
            max_seg_len=self.max_seg_len,
            entity_dict=self.entity_dict,
            mention_context_length=self.mention_context_length
        )

        ######
        # Loss Function
        ######

        self.loss_function = nn.CrossEntropyLoss(reduction="none")

    def _initialize_bert_and_tokenizer(self, pretrained_model_name_or_path):
        """
        Parameters
        ----------
        pretrained_model_name_or_path : str

        Returns
        -------
        tuple[AutoModel, AutoTokenizer]
        """
        bert = AutoModel.from_pretrained(
            pretrained_model_name_or_path,
            return_dict=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            additional_special_tokens=None
        )
        return bert, tokenizer

    ########################
    # For optimization
    ########################

    def get_params(self, named=False):
        """
        Parameters
        ----------
        named : bool
            by default False

        Returns
        -------
        tuple[list[tuple[str, Any]], list[tuple[str, Any]]]
        """
        bert_based_param, task_param = [], []
        for name, param in self.named_parameters():
            if name.startswith('bert'):
                to_add = (name, param) if named else param
                bert_based_param.append(to_add)
            else:
                to_add = (name, param) if named else param
                task_param.append(to_add)
        return bert_based_param, task_param

    ################
    # Forward pass
    ################

    def preprocess(
        self,
        document,
        candidate_entities_for_doc,
        max_n_candidates=None
    ):
        """
        Parameters
        ----------
        document : Document
        candidate_entities : dict[str, str | list[list[CandEntKeyInfo]]]
        max_n_candidates : int | None
            by default None

        Returns
        -------
        dict[str, Any]
        """
        return self.preprocessor.preprocess(
            document=document,
            candidate_entities_for_doc=candidate_entities_for_doc,
            max_n_candidates=max_n_candidates
        )

    def tensorize(self, preprocessed_data, mention_index, compute_loss):
        """
        Parameters
        ----------
        preprocessed_data : dict[str, Any]
        mention_index : int
        compute_loss : bool

        Returns
        -------
        dict[str, Any]
        """
        model_input = {}

        model_input["compute_loss"] = compute_loss

        # (n_candidates, max_seg_len)
        model_input["segments_id"] = torch.tensor(
            preprocessed_data["mention_index_to_bert_input"][mention_index]
            ["segments_id"],
            device=self.device
        )

        # (n_candidates, max_seg_len)
        model_input["segments_mask"] = torch.tensor(
            preprocessed_data["mention_index_to_bert_input"][mention_index]
            ["segments_mask"],
            device=self.device
        )

        # (n_candidates, max_seg_len)
        model_input["segments_token_type_id"] = torch.tensor(
            preprocessed_data["mention_index_to_bert_input"][mention_index]
            ["segments_token_type_id"],
            device=self.device
        )

        if not compute_loss:
            return model_input

        # For the training set, we assume that the first candidate
        #   is always the gold entity for the corresponding mention.
        # (1,)
        model_input["gold_candidate_entity_indices"] = torch.tensor(
            [0],
            device=self.device
        ).to(torch.long)

        return model_input

    def forward(
        self,
        segments_id,
        segments_mask,
        segments_token_type_id,
        compute_loss,
        gold_candidate_entity_indices=None
    ):
        """
        Parameters
        ----------
        segments_id : torch.Tensor
            shape of (n_candidates, max_seg_len)
        segments_mask : torch.Tensor
            shape of (n_candidates, max_seg_len)
        segments_token_type_id : torch.Tensor
            shape of (n_candidates, max_seg_len)
        compute_loss : bool
        gold_candidate_entity_indices : torch.Tensor | None
            shape of (1,); by default None

        Returns
        -------
        ModelOutput
        """
        # Encode tokens by BERT
        # (n_candidates, max_seg_len, hidden_dim)
        segments_vec = self.encode_tokens(
            segments_id=segments_id,
            segments_mask=segments_mask,
            segments_token_type_id=segments_token_type_id
        )

        # Get [CLS] vectors
        # (n_candidates, hidden_dim)
        candidate_entity_vectors = segments_vec[:, 0, :]

        # Compute logits by a linear layer
        # (n_candidates, 1)
        logits = self.linear(candidate_entity_vectors)
        # (1, n_candidates)
        logits = logits.unsqueeze(0).squeeze(-1)

        if not compute_loss:
            return ModelOutput(
                logits=logits
            )

        # Compute loss (summed over mentions)
        # (1,)
        loss = self.loss_function(logits, gold_candidate_entity_indices)
        loss = loss.sum() # Scalar

        # Compute accuracy
        # (1,)
        pred_candidate_entity_indices = torch.argmax(logits, dim=1)
        # (1,)
        acc = (
            pred_candidate_entity_indices == gold_candidate_entity_indices
        ).to(torch.float)
        acc = acc.sum().item() # Scalar

        return ModelOutput(
            logits=logits,
            loss=loss,
            acc=acc
        )

   ################
    # Subfunctions
    ################

    def encode_tokens(self, segments_id, segments_mask, segments_token_type_id):
        """
        Parameters
        ----------
        segments_id : torch.Tensor
            shape of (n_candidates, max_seg_len)
        segments_mask : torch.Tensor
            shape of (n_candidates, max_seg_len)
        segments_token_type_id : torch.Tensor
            shape of (n_candidates, max_seg_len)

        Returns
        -------
        torch.Tensor
            shape of (n_candidates, max_seg_len, hidden_dim)
        """
        bert_output = self.bert(
            input_ids=segments_id,
            attention_mask=segments_mask,
            token_type_ids=segments_token_type_id,
            output_attentions=False,
            output_hidden_states=False
        )
        # (n_candidates, max_seg_len, hidden_dim)
        segments_vec = bert_output["last_hidden_state"]
        return segments_vec


class BlinkCrossEncoderPreprocessor:

    def __init__(
        self,
        tokenizer,
        max_seg_len,
        entity_dict,
        mention_context_length
    ):
        """
        Parameters
        ----------
        tokenizer : PreTrainedTokenizer
        max_seg_len : int
        entity_dict: dict[str, EntityPage]
        mention_context_length : int
        """
        self.tokenizer = tokenizer
        self.max_seg_len = max_seg_len
        self.entity_dict = entity_dict
        self.mention_context_length = mention_context_length

        self.special_mention_begin_marker = "*"
        self.special_mention_end_marker = "*"
        self.special_entity_sep_marker = ":"

    def preprocess(
        self,
        document,
        candidate_entities_for_doc,
        max_n_candidates=None):
        """
        Parameters
        ----------
        document : Document
        candidate_entiteis : dict[str, str | list[list[CandEntKeyInfo]]]
        max_n_candidates : int | None
            by default None

        Returns
        -------
        dict[str, Any]
        """
        preprocessed_data = OrderedDict()

        #####
        # doc_key: str
        # sentences: list[list[str]]
        # mentions: list[MentionTuple]
        # entities: list[EntityTuple]
        #####

        preprocessed_data["doc_key"] = document["doc_key"]

        sentences = [s.split() for s in document["sentences"]]
        preprocessed_data["sentences"] = sentences

        if "entity_id" in document["mentions"][0]:
            with_supervision = True
        else:
            with_supervision = False
        if with_supervision:
            mentions = [
                MentionTuple(
                    tuple(m["span"]),
                    m["name"],
                    m["entity_type"],
                    m["entity_id"]
                )
                for m in document["mentions"]
            ]
        else:
            mentions = [
                MentionTuple(
                    tuple(m["span"]),
                    m["name"],
                    m["entity_type"],
                    None
                )
                for m in document["mentions"]
            ]
        preprocessed_data["mentions"] = mentions

        if with_supervision:
            entities = [
                EntityTuple(
                    e["mention_indices"],
                    e["entity_type"],
                    e["entity_id"]
                )
                for e in document["entities"]
            ]
            preprocessed_data["entities"] = entities

        #####
        # mention_index_to_sentence_index: list[int]
        # sentence_index_to_mention_indices: list[list[int]]
        # mention_index_to_entity_index: list[int]
        #####

        # Mention index to sentence index
        token_index_to_sent_index = [] # list[int]
        for sent_i, sent in enumerate(sentences):
            token_index_to_sent_index.extend([sent_i for _ in range(len(sent))])
        mention_index_to_sentence_index = [] # list[int]
        for mention in mentions:
            (begin_token_index, end_token_index) = mention.span
            sentence_index = token_index_to_sent_index[begin_token_index]
            assert token_index_to_sent_index[end_token_index] == sentence_index
            mention_index_to_sentence_index.append(sentence_index)
        preprocessed_data["mention_index_to_sentence_index"] \
            = mention_index_to_sentence_index

        # Sentence index to mention indices
        sentence_index_to_mention_indices = [None] * len(sentences)
        for mention_i, sent_i in enumerate(mention_index_to_sentence_index):
            if sentence_index_to_mention_indices[sent_i] is None:
                sentence_index_to_mention_indices[sent_i] = [mention_i]
            else:
                sentence_index_to_mention_indices[sent_i].append(mention_i)
        for sent_i in range(len(sentences)):
            if sentence_index_to_mention_indices[sent_i] is None:
                sentence_index_to_mention_indices[sent_i] = []
        preprocessed_data["sentence_index_to_mention_indices"] \
            = sentence_index_to_mention_indices

        if with_supervision:
            # Mention index to entity index
            # NOTE: Although a single mention may belong to multiple entities,
            #       we assign only one entity index to each mention
            mention_index_to_entity_index = [None] * len(mentions)
            for entity_i, entity in enumerate(entities):
                for mention_i in entity.mention_indices:
                    mention_index_to_entity_index[mention_i] = entity_i
            preprocessed_data["mention_index_to_entity_index"] \
                = mention_index_to_entity_index

        #####
        # mention_index_to_bert_input: list[dict[str, Any]]
        # segments: list[list[str]]
        # segments_id: list[list[int]]
        # segments_mask: list[list[int]]
        #####

        mention_index_to_bert_input = [] # list[dict[str, Any]]
        # list[list[CandEntKeyInfo]]
        cands_for_mentions = candidate_entities_for_doc["candidate_entities"]
        for mention_index in range(len(document["mentions"])):
            # list[CandEntKeyInfo]
            cands_for_mention = cands_for_mentions[mention_index]
            if max_n_candidates is not None:
                cands_for_mention = cands_for_mention[:max_n_candidates]
            (
                segments,
                segments_id,
                segments_mask,
                segments_token_type_id
            ) = self.tokenize_and_split(
                sentences=sentences,
                mention=mentions[mention_index],
                candidate_entities_for_mention=cands_for_mention
            )
            bert_input = {}
            bert_input["segments"] = segments
            bert_input["segments_id"] = segments_id
            bert_input["segments_mask"] = segments_mask
            bert_input["segments_token_type_id"] = segments_token_type_id
            mention_index_to_bert_input.append(bert_input)
        preprocessed_data["mention_index_to_bert_input"] \
            = mention_index_to_bert_input

        return preprocessed_data

    #####
    # Subfunctions
    #####

    def tokenize_and_split(
        self,
        sentences,
        mention,
        candidate_entities_for_mention
    ):
        """
        Parameters
        ----------
        sentences: list[list[str]]
        mention: MentionTuple
        candidate_entities_for_mention: list[CandEntKeyInfo]

        Returns
        -------
        tuple[list[list[str]], list[list[int]], list[list[int]], list[list[int]]]
        """
        # Make mention-side sequence
        words = utils.flatten_lists(sentences) # list[str]
        begin_i, end_i = mention.span
        left_context = " ".join(
            words[begin_i - self.mention_context_length : begin_i]
        )
        mention_string = " ".join(
            words[begin_i : end_i + 1]
        )
        right_context = " ".join(
            words[end_i + 1 : end_i + 1 + self.mention_context_length]
        )
        # "<left context> * <mention> * <right context>"
        mention_seq = " ".join([
            left_context,
            self.special_mention_begin_marker,
            mention_string,
            self.special_mention_end_marker,
            right_context
        ])
        # Make entity-side sequence
        entity_seqs = [] # list[str]
        for cand in candidate_entities_for_mention:
            entity_id = cand["entity_id"]
            epage = self.entity_dict[entity_id]
            canonical_name = epage["canonical_name"]
            # synonyms = epage["synonyms"]
            description = epage["description"]
            # "<canonical name> : <description>"
            entity_seq = " ".join([
                canonical_name,
                self.special_entity_sep_marker,
                description
            ])
            entity_seqs.append(entity_seq)
        # Combine and tokenize the sequences
        # [CLS] <left context> * <mention> * <right context> [SEP] <canonical name> : <description> [SEP]
        inputs = self.tokenizer(
            [mention_seq] * len(entity_seqs),
            entity_seqs,
            max_length=self.max_seg_len,
            padding=True,
            truncation="only_second",
            return_overflowing_tokens=False # NOTE
        )
        segments_id = inputs["input_ids"] # list[list[int]]
        segments = [
            self.tokenizer.convert_ids_to_tokens(seg)
            for seg in inputs["input_ids"]
        ] # list[list[str]]
        segments_mask = inputs["attention_mask"] # list[list[int]]
        segments_token_type_id = inputs["token_type_ids"] # list[list[int]]
        return (
            segments,
            segments_id,
            segments_mask,
            segments_token_type_id
        )

