from __future__ import annotations

from collections import defaultdict
from collections import OrderedDict
from typing import NamedTuple
import logging

import numpy as np
# import spacy_alignments as tokenizations
import torch
import torch.nn as nn
# import torch.nn.functional as F
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers.modeling_outputs import ModelOutput
# from opt_einsum import contract

from . import shared_functions
from .losses import FocalLoss
from .. import utils


logger = logging.getLogger(__name__)


class MentionTuple(NamedTuple):
    span: tuple[int, int]
    name: str
    entity_type: str


class BiaffineNERModel(nn.Module):

    def __init__(
        self,
        device,
        bert_pretrained_name_or_path,
        max_seg_len,
        dropout_rate,
        vocab_etype,
        loss_function_name,
        focal_loss_gamma=None
    ):
        """
        Parameters
        ----------
        device : str
        bert_pretrained_name_or_path : str
        max_seg_len : int
        dropout_rate : float
        vocab_etype : dict[str, int]
        loss_function_name : str
        focal_loss_gamma : float | None
            by default None
        """
        super().__init__()

        ########################
        # Hyper parameters
        ########################

        self.device = device
        self.bert_pretrained_name_or_path = bert_pretrained_name_or_path
        self.max_seg_len = max_seg_len
        self.dropout_rate = dropout_rate
        self.vocab_etype = vocab_etype
        self.loss_function_name = loss_function_name
        self.focal_loss_gamma = focal_loss_gamma

        self.n_entity_types = len(self.vocab_etype)

        ########################
        # Components
        ########################

        # BERT, tokenizer
        self.bert, self.tokenizer = self._initialize_bert_and_tokenizer(
            pretrained_model_name_or_path=self.bert_pretrained_name_or_path
        )

        # Dimensionality
        self.hidden_dim = self.bert.config.hidden_size

        # Entity type classification
        self.linear_begin = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear_end = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dropout_begin = nn.Dropout(p=self.dropout_rate)
        self.dropout_end = nn.Dropout(p=self.dropout_rate)
        self.biaffine = shared_functions.Biaffine(
            input_dim=self.hidden_dim,
            output_dim=self.n_entity_types,
            bias_x=True,
            bias_y=True
        )

        ######
        # Preprocessor
        ######

        self.preprocessor = BiaffineNERPreprocessor(
            tokenizer=self.tokenizer,
            max_seg_len=self.max_seg_len,
            vocab_etype=self.vocab_etype
        )

        ######
        # Loss Function
        ######

        if self.loss_function_name == "cross_entropy":
            self.loss_function = nn.CrossEntropyLoss(reduction="none")
        elif self.loss_function_name == "focal_loss":
            self.loss_function = FocalLoss(
                gamma=self.focal_loss_gamma,
                reduction="none"
            )
        else:
            raise Exception(
                f"Invalid loss_function: {self.loss_function_name}"
            )

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

    def preprocess(self, document):
        """
        Parameters
        ----------
        document : Document

        Returns
        -------
        dict[str, Any]
        """
        return self.preprocessor.preprocess(document=document)

    def tensorize(self, preprocessed_data, compute_loss):
        """
        Parameters
        ----------
        preprocessed_data : dict[str, Any]
        compute_loss : bool

        Returns
        -------
        dict[str, Any]
        """
        model_input = {}

        model_input["compute_loss"] = compute_loss

        # (n_segments, max_seg_len)
        model_input["segments_id"] = torch.tensor(
            preprocessed_data["bert_input"]["segments_id"],
            device=self.device
        )

        # (n_segments, max_seg_len)
        model_input["segments_mask"] = torch.tensor(
            preprocessed_data["bert_input"]["segments_mask"],
            device=self.device
        )

        if not compute_loss:
            return model_input

        # (n_tokens, n_tokens)
        model_input["matrix_valid_span_mask"] = torch.tensor(
            preprocessed_data["matrix_valid_span_mask"],
            device=self.device
        ).to(torch.float)

        # (n_tokens, n_tokens)
        model_input["matrix_gold_entity_type_labels"] = torch.tensor(
            preprocessed_data["matrix_gold_entity_type_labels"],
            device=self.device
        ).to(torch.long)

        return model_input

    def forward(
        self,
        segments_id,
        segments_mask,
        compute_loss,
        matrix_valid_span_mask=None,
        matrix_gold_entity_type_labels=None
    ):
        """
        Parameters
        ----------
        segments_id : torch.Tensor
            shape of (n_segments, max_seg_len)
        segments_mask : torch.Tensor
            shape of (n_segments, max_seg_len)
        compute_loss : bool
        matrix_valid_span_mask : torch.Tensor | None
            shape of (n_tokens, n_tokens); by default None
        matrix_gold_entity_type_labels : torch.Tensor | None
            shape of (n_tokens, n_tokens); by default None

        Returns
        -------
        ModelOutput
        """
        # Encode tokens by BERT
        # (n_tokens, hidden_dim)
        token_vectors = self.encode_tokens(
            segments_id=segments_id,
            segments_mask=segments_mask
        )

        # Compute logits by Biaffine
        # (n_tokens, n_tokens, n_entity_types)
        logits = self.compute_logits_by_biaffine(
            token_vectors=token_vectors
        )

        if not compute_loss:
            return ModelOutput(
                logits=logits
            )

        # Compute loss (summed over valid spans)
        # (n_tokens, n_tokens)
        loss = self.loss_function(
            logits.permute(2,0,1).unsqueeze(0),
            matrix_gold_entity_type_labels.unsqueeze(0)
        ).squeeze(0)
        loss = loss * matrix_valid_span_mask
        loss = loss.sum() # Scalar

        # Compute accuracy (summed over valid spans)
        # (n_tokens, n_tokens)
        matrix_pred_entity_type_labels = logits.argmax(dim=-1)
        # (n_tokens, n_tokens)
        acc = (
            matrix_pred_entity_type_labels == matrix_gold_entity_type_labels
        ).to(torch.float)
        acc = acc * matrix_valid_span_mask
        acc = acc.sum().item() # Scalar

        n_valid_spans = int(matrix_valid_span_mask.sum().item())

        return ModelOutput(
            logits=logits,
            loss=loss,
            acc=acc,
            n_valid_spans=n_valid_spans
        )

    ################
    # Subfunctions
    ################

    def encode_tokens(self, segments_id, segments_mask):
        """
        Parameters
        ----------
        segments_id : torch.Tensor
            shape of (n_segments, max_seg_len)
        segments_mask : torch.Tensor
            shape of (n_segments, max_seg_len)

        Returns
        -------
        torch.Tensor
            shape of (n_tokens, hidden_dim)
        """
        # Check
        n_segments, max_seg_len = segments_id.shape
        assert max_seg_len == self.max_seg_len

        # Encode segments by BERT
        bert_output = self.bert(
            input_ids=segments_id,
            attention_mask=segments_mask,
            output_attentions=False,
            output_hidden_states=False
        )
        # (n_segments, max_seg_len, hidden_dim)
        segments_vec = bert_output["last_hidden_state"]

        # Transform `segments_vec` to token vectors
        # (n_segments, max_seg_len)
        segments_mask_bool = segments_mask.to(torch.bool)
        # (n_tokens, hidden_dim)
        token_vectors = segments_vec[segments_mask_bool]

        return token_vectors

    def compute_logits_by_biaffine(self, token_vectors):
        """
        Parameters
        ----------
        token_vectors : torch.Tensor
            shape of (n_tokens, hidden_dim)

        Returns
        -------
        torch.Tensor
            shape of (n_tokens, n_tokens, n_entity_types)
        """
        # (n_tokens, hidden_dim)
        zb = self.dropout_begin(
            torch.tanh(
                self.linear_begin(token_vectors)
            )
        )
        # (n_tokens, hidden_dim)
        ze = self.dropout_end(
            torch.tanh(
                self.linear_end(token_vectors)
            )
        )
        # (batch_size=1, n_tokens, hidden_dim)
        zb = zb.unsqueeze(0)
        # (batch_size=1, n_tokens, hidden_dim)
        ze = ze.unsqueeze(0)
        # (batch_size=1, n_entity_types, n_tokens, n_tokens)
        logits = self.biaffine(zb.float(), ze.float())
        # (batch_size=1, n_tokens, n_tokens, n_entity_types)
        logits = logits.permute(0, 2, 3, 1)
        # (n_tokens, n_tokens, n_entity_types)
        logits = logits.squeeze(0)
        return logits


class BiaffineNERPreprocessor:

    def __init__(self, tokenizer, max_seg_len, vocab_etype):
        """
        Parameters
        ----------
        tokenizer: PreTrainedTokenizer
        max_seg_len: int
        vocab_etype: dict[str, int]
        """
        self.tokenizer = tokenizer
        self.max_seg_len = max_seg_len
        self.vocab_etype = vocab_etype

        self.cls_token = tokenizer.cls_token
        self.sep_token = tokenizer.sep_token

    # ---

    def preprocess(self, document):
        """
        Parameters
        ----------
        document : Document

        Returns
        -------
        dict[str, Any]
        """
        preprocessed_data = OrderedDict()

        #####
        # doc_key: str
        # sentences: list[list[str]]
        # mentions: list[MentionTuple]
        #####

        preprocessed_data["doc_key"] = document["doc_key"]

        sentences = [s.split() for s in document["sentences"]]
        preprocessed_data["sentences"] = sentences

        with_supervision = True if "mentions" in document else False
        if with_supervision:
            mentions = [
                MentionTuple(
                    tuple(m["span"]),
                    m["name"],
                    m["entity_type"]
                )
                for m in document["mentions"]
            ]
            preprocessed_data["mentions"] = mentions

        #####
        # bert_input: dict[str, Any]
        # segments: list[list[str]]
        # segments_id: list[list[int]]
        # segments_mask: list[list[int]]
        # subtoken_index_to_word_index: list[int]
        # word_index_to_subtoken_indices: list[ist[int]]
        # subtoken_index_to_sentence_index: list[int]
        #####

        (
            segments,
            segments_id,
            segments_mask,
            subtoken_index_to_word_index,
            word_index_to_subtoken_indices,
            subtoken_index_to_sentence_index
        ) = self.tokenize_and_split(sentences=sentences)
        bert_input = {}
        bert_input["segments"] = segments
        bert_input["segments_id"] = segments_id
        bert_input["segments_mask"] = segments_mask
        bert_input["subtoken_index_to_word_index"] \
            = subtoken_index_to_word_index
        bert_input["word_index_to_subtoken_indices"] \
            = word_index_to_subtoken_indices
        bert_input["subtoken_index_to_sentence_index"] \
            = subtoken_index_to_sentence_index
        preprocessed_data["bert_input"] = bert_input

        #####
        # matrix_valid_span_mask: list[list[float]]
        #####

        # We will get prediction and losses only for spans
        #   within the same sentence.
        n_subtokens = len(utils.flatten_lists(bert_input["segments"]))
        matrix_valid_span_mask = np.zeros((n_subtokens, n_subtokens))
        offset = 0
        for sent in sentences:
            for local_word_i in range(0, len(sent)):
                global_word_i = offset + local_word_i
                # first subtoken
                global_subtok_i \
                    = word_index_to_subtoken_indices[global_word_i][0]
                for local_word_j in range(local_word_i, len(sent)):
                    global_word_j = offset + local_word_j
                    # first subtoken
                    global_subtok_j \
                        = word_index_to_subtoken_indices[global_word_j][0]
                    matrix_valid_span_mask[global_subtok_i, global_subtok_j] \
                        = 1.0
            offset += len(sent)
        preprocessed_data["matrix_valid_span_mask"] = matrix_valid_span_mask

        #####
        # matrix_gold_entity_type_labels: list[list[int]]
        #####

        if with_supervision:
            matrix_gold_entity_type_labels = np.zeros(
                (n_subtokens, n_subtokens), dtype=np.int32
            )
            for mention in mentions:
                begin_word_i, end_word_i = mention.span
                begin_subtok_i = word_index_to_subtoken_indices[begin_word_i][0]
                end_subtok_i = word_index_to_subtoken_indices[end_word_i][0]
                etype_id = self.vocab_etype[mention.entity_type] # str -> int
                matrix_gold_entity_type_labels[begin_subtok_i, end_subtok_i] \
                    = etype_id
            preprocessed_data["matrix_gold_entity_type_labels"] \
                = matrix_gold_entity_type_labels

        return preprocessed_data

    #####
    # Subfunctions
    #####

    def tokenize_and_split(self, sentences):
        """
        Parameters
        ----------
        sentences: list[list[str]]

        Returns
        -------
        tuple[list[list[str]], list[list[int]], list[list[int]], list[int],
            list[list[int]], list[int]]
        """
        # subtoken分割、subtoken単位でのtoken終点、文終点のindicatorsの作成
        (
            sents_subtoken,
            sents_token_end,
            sents_sentence_end,
            subtoken_index_to_word_index
        ) = self.tokenize(sentences=sentences)

        # 1階層のリストに変換
        doc_subtoken = utils.flatten_lists(sents_subtoken) # list[str]
        doc_token_end = utils.flatten_lists(sents_token_end) # list[bool]
        doc_sentence_end = utils.flatten_lists(sents_sentence_end) # list[bool]

        # BERTセグメントに分割
        (
            segments,
            subtoken_index_to_word_index,
            word_index_to_subtoken_indices,
            subtoken_index_to_sentence_index
        ) = self.split(
             doc_subtoken=doc_subtoken,
             doc_sentence_end=doc_sentence_end,
             doc_token_end=doc_token_end,
             subtoken_index_to_word_index=subtoken_index_to_word_index
        )

        # subtoken IDへの変換とpaddingマスクの作成
        segments_id, segments_mask = self.convert_to_token_ids_with_padding(
            segments=segments
        )

        return (
            segments,
            segments_id,
            segments_mask,
            subtoken_index_to_word_index,
            word_index_to_subtoken_indices,
            subtoken_index_to_sentence_index
        )

    def tokenize(self, sentences):
        """
        Parameters
        ----------
        sentences: list[list[str]]

        Returns
        -------
        tuple[list[list[str]], list[list[bool]], list[list[bool]], list[int]]
        """
        sents_subtoken = [] # list[list[str]]
        sents_token_end = [] # list[list[bool]]
        sents_sentence_end = [] # list[list[bool]]
        subtoken_index_to_word_index = [] # list[int]

        word_idx = -1
        offset = 0
        for sent in sentences:
            sent_subtoken = [] # list[str]
            sent_token_end = [] # list[bool]
            sent_sentence_end = [] # list[bool]
            for token_i, token in enumerate(sent):
                word_idx += 1
                # サブトークン
                subtokens = self.tokenizer.tokenize(token)
                if len(subtokens) == 0:
                    subtokens = [self.tokenizer.unk_token]
                sent_subtoken.extend(subtokens)
                # トークンの終了位置
                sent_token_end += [False] * (len(subtokens) - 1) + [True]
                # 文の終了位置 (仮)
                sent_sentence_end += [False] * len(subtokens)
                # subtoken index -> word index
                subtoken_index_to_word_index += [word_idx] * len(subtokens)
            # 文の終了位置
            sent_sentence_end[-1] = True
            sents_subtoken.append(sent_subtoken)
            sents_token_end.append(sent_token_end)
            sents_sentence_end.append(sent_sentence_end)
            offset += len(sent)

        return (
            sents_subtoken,
            sents_token_end,
            sents_sentence_end,
            subtoken_index_to_word_index
        )

    def split(
        self,
        doc_subtoken,
        doc_sentence_end,
        doc_token_end,
        subtoken_index_to_word_index
    ):
        """
        Parameters
        ----------
        doc_subtoken: list[str]
        doc_sentence_end: list[bool]
        doc_token_end: list[bool]
        subtoken_index_to_word_index: list[int]

        Returns
        -------
        tuple[list[list[str]], list[int], list[list[int]], list[int]]
        """
        segments = [] # list[list[str]]
        segments_subtoken_map = [] # list[list[int]]

        n_subtokens = len(doc_subtoken)
        curr_idx = 0 # Index for subtokens
        while curr_idx < len(doc_subtoken):
            # Try to split at a sentence end point
            end_idx = min(curr_idx + self.max_seg_len - 1 - 2, n_subtokens - 1)
            while end_idx >= curr_idx and not doc_sentence_end[end_idx]:
                end_idx -= 1
            if end_idx < curr_idx:
                logger.warning("No sentence end found; split at token end")
                # If no sentence end point found,
                #   try to split at token end point.
                end_idx = min(
                    curr_idx + self.max_seg_len - 1 - 2,
                    n_subtokens - 1
                )
                seg_before \
                    = "\"" + " ".join(doc_subtoken[curr_idx:end_idx+1]) + "\""
                while end_idx >= curr_idx and not doc_token_end[end_idx]:
                    end_idx -= 1
                if end_idx < curr_idx:
                    logger.warning("Cannot split valid segment: no sentence end or token end")
                    raise Exception
                seg_after \
                    = "\"" + " ".join(doc_subtoken[curr_idx:end_idx+1]) + "\""
                logger.warning("------")
                logger.warning("Segment where no sentence-ending position was found:")
                logger.warning(seg_before)
                logger.warning("---")
                logger.warning("Segment splitted based on a token-ending position:")
                logger.warning(seg_after)
                logger.warning("------")

            segment = doc_subtoken[curr_idx: end_idx + 1]
            segment_subtoken_map \
                = subtoken_index_to_word_index[curr_idx: end_idx + 1]
            segment = [self.cls_token] + segment + [self.sep_token]
            # NOTE: [CLS] is treated as the first subtoken
            #     of the first word for each segment.
            # NOTE: [SEP] is treated as the last subtoken
            #     of the last word for each segment.
            segment_subtoken_map = (
                [segment_subtoken_map[0]]
                + segment_subtoken_map
                + [segment_subtoken_map[-1]]
            )

            segments.append(segment)
            segments_subtoken_map.append(segment_subtoken_map)

            curr_idx = end_idx + 1

        # Create a map from word index to subtoken indices (list)
        word_index_to_subtoken_indices \
            = self.get_word_index_to_subtoken_indices(
                segments_subtoken_map=segments_subtoken_map
            )

        # Subtoken index to word index
        subtoken_index_to_word_index = utils.flatten_lists(
            segments_subtoken_map
        )

        # Subtoken index to sentence index
        subtoken_index_to_sentence_index \
            = self.get_subtoken_index_to_sentence_index(
                segments=segments,
                doc_sentence_end=doc_sentence_end
            )

        return (
            segments,
            subtoken_index_to_word_index,
            word_index_to_subtoken_indices,
            subtoken_index_to_sentence_index
        )

    def get_word_index_to_subtoken_indices(self, segments_subtoken_map):
        """
        Parameters
        ----------
        segments_subtoken_map : list[list[int]]

        Returns
        -------
        list[list[int]]
        """
        word_index_to_subtoken_indices = defaultdict(list)
        offset = 0
        for segment_subtoken_map in segments_subtoken_map:
            for subtok_i, word_i in enumerate(segment_subtoken_map):
                if subtok_i == 0 or subtok_i == len(segment_subtoken_map) - 1:
                    continue
                word_index_to_subtoken_indices[word_i].append(offset + subtok_i)
            offset += len(segment_subtoken_map)
        return word_index_to_subtoken_indices

    def get_subtoken_index_to_sentence_index(self, segments, doc_sentence_end):
        """
        Parameters
        ----------
        segments : list[list[str]]
        doc_sentence_end : list[bool]

        Returns
        -------
        list[int]
        """
        assert len(doc_sentence_end) == sum([len(seg) - 2 for seg in segments])
        sent_map = [] # list[int]
        sent_idx, subtok_idx = 0, 0
        for segment in segments:
            sent_map.append(sent_idx) # [CLS]
            length = len(segment) - 2
            for i in range(length):
                sent_map.append(sent_idx)
                sent_idx += int(doc_sentence_end[subtok_idx]) # 0 or 1
                subtok_idx += 1
            # [SEP] is the current sentence's last token
            sent_map.append(sent_idx - 1)
        return sent_map

    def convert_to_token_ids_with_padding(
        self,
        segments
    ):
        """
        Parameters
        ----------
        segments: list[list[str]]

        Returns
        -------
        Tuple[list[list[int]], list[list[int]]]
        """
        segments_id = [] # list[list[int]]
        segments_mask = [] # list[list[int]]
        n_subtokens = sum([len(s) for s in segments])
        for segment in segments:
            segment_id = self.tokenizer.convert_tokens_to_ids(segment)
            segment_mask = [1] * len(segment_id)
            while len(segment_id) < self.max_seg_len:
                segment_id.append(0)
                segment_mask.append(0)
            segments_id.append(segment_id)
            segments_mask.append(segment_mask)
        assert np.sum(np.asarray(segments_mask)) == n_subtokens
        return segments_id, segments_mask


