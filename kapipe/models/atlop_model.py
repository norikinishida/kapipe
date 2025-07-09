from __future__ import annotations

from collections import defaultdict
from collections import OrderedDict
import logging
from typing import Any, NamedTuple

import numpy as np
# import spacy_alignments as tokenizations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from transformers import AutoModel, AutoTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.modeling_outputs import ModelOutput
from opt_einsum import contract

# from . import shared_functions
from .losses import AdaptiveThresholdingLoss
from .. import utils
from ..datatypes import Document


logger = logging.getLogger(__name__)


class MentionTuple(NamedTuple):
    span: tuple[int, int]
    name: str
    entity_type: str
    entity_id: str


class EntityTuple(NamedTuple):
    mention_indices: list[int]
    entity_type: str
    entity_id: str


class TripleTuple(NamedTuple):
    arg1: int
    relation: str
    arg2: int


class ATLOPModel(nn.Module):

    def __init__(
        self,
        device: str,
        bert_pretrained_name_or_path: str,
        max_seg_len: int,
        token_embedding_method: str,
        entity_pooling_method: str,
        use_localized_context_pooling: bool,
        bilinear_block_size: int,
        vocab_relation: dict[str, int],
        loss_function_name: str,
        possible_head_entity_types: list[str] | None = None,
        possible_tail_entity_types: list[str] | None = None
    ):
        super().__init__()

        ########################
        # Hyper parameters
        ########################

        self.device = device
        self.bert_pretrained_name_or_path = bert_pretrained_name_or_path
        self.max_seg_len = max_seg_len
        self.token_embedding_method = token_embedding_method
        self.entity_pooling_method = entity_pooling_method
        self.use_localized_context_pooling = use_localized_context_pooling
        self.bilinear_block_size = bilinear_block_size
        self.vocab_relation = vocab_relation
        self.loss_function_name = loss_function_name
        self.possible_head_entity_types = possible_head_entity_types
        self.possible_tail_entity_types = possible_tail_entity_types

        self.n_relations = len(self.vocab_relation)

        assert self.token_embedding_method in ["independent", "overlap"]
        assert (
            self.entity_pooling_method in ["sum", "mean", "max", "logsumexp"]
        )

        ########################
        # Components
        ########################

        # BERT, tokenizer
        self.bert, self.tokenizer = self._initialize_bert_and_tokenizer(
            pretrained_model_name_or_path=self.bert_pretrained_name_or_path
        )

        # Dimensionality
        self.hidden_dim = self.bert.config.hidden_size

        # DocRE classification
        self.linear_head = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.linear_tail = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.block_bilinear = nn.Linear(
            self.hidden_dim * self.bilinear_block_size,
            self.n_relations
        )

        ########################
        # Preprocessor
        ########################

        self.preprocessor = ATLOPPreprocessor(
            tokenizer=self.tokenizer,
            max_seg_len=self.max_seg_len,
            vocab_relation=self.vocab_relation,
            possible_head_entity_types=possible_head_entity_types,
            possible_tail_entity_types=possible_tail_entity_types
        )

        ########################
        # Loss function
        ########################

        if self.loss_function_name == "adaptive_thresholding_loss":
            self.loss_function = AdaptiveThresholdingLoss()
        else:
            raise Exception(f"Invalid loss function: {self.loss_function_name}")


    def _initialize_bert_and_tokenizer(
        self,
        pretrained_model_name_or_path: bool
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
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

    def get_params(self, named: bool = False) -> tuple[
        list[Parameter | tuple[str, Parameter]],
        list[Parameter | tuple[str, Parameter]]
    ]:
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

    def preprocess(self, document: Document) -> dict[str, Any]:
        return self.preprocessor.preprocess(document=document)

    def tensorize(
        self,
        preprocessed_data: dict[str, Any],
        compute_loss: bool
    ) -> dict[str, Any]:
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

        # (n_mentions,)
        model_input["mention_begin_token_indices"] = torch.tensor(
            preprocessed_data["mention_begin_token_indices"],
            device=self.device
        )

        # (n_mentions,)
        # model_input["mention_end_token_indices"] = torch.tensor(
        #     preprocessed_data["mention_end_token_indices"],
        #     device=self.device
        # )

        # (n_entities, n_mentions_for_each_entity)
        model_input["entity_index_to_mention_indices"] = [
            torch.tensor(mention_indices, device=self.device) \
            for mention_indices in preprocessed_data[
                "entity_index_to_mention_indices"
            ]
        ]

        # (n_entity_pairs,)
        model_input["pair_head_entity_indices"] = torch.tensor(
            preprocessed_data["pair_head_entity_indices"],
            device=self.device
        )

        # (n_entity_pairs,)
        model_input["pair_tail_entity_indices"] = torch.tensor(
            preprocessed_data["pair_tail_entity_indices"],
            device=self.device
        )

        if not compute_loss:
            return model_input

        # (n_entity_pairs, n_relations)
        model_input["pair_gold_relation_labels"] = torch.tensor(
            preprocessed_data["pair_gold_relation_labels"],
            device=self.device
        ).to(torch.float)

        return model_input

    def forward(
        self,
        segments_id: torch.Tensor,
        segments_mask: torch.Tensor,
        mention_begin_token_indices: torch.Tensor,
        entity_index_to_mention_indices: list[torch.Tensor],
        pair_head_entity_indices: torch.Tensor,
        pair_tail_entity_indices: torch.Tensor,
        compute_loss: bool,
        pair_gold_relation_labels: torch.Tensor | None = None,
    ) -> ModelOutput:
        """
        Shapes
        ---------
        segments_id : (n_segments, max_seg_len)
        segments_mask : (n_segments, max_seg_len)
        mention_begin_token_indices : (n_mentions,)
        entity_index_to_mention_indices : list of length n_entities; each tensor shape: (n_mentions_for_this_entity,)
        pair_head_entity_indices : (n_entity_pairs,)
        pair_tail_entity_indices : (n_entity_pairs,)
        pair_gold_relation_labels : (n_entity_pairs, n_relations)
        """
        # Encode tokens by BERT
        if self.token_embedding_method == "independent":
            # (n_tokens, hidden_dim), (n_heads, n_tokens, n_tokens)
            token_vectors, attentions \
                = self.encode_tokens_with_independent_segments(
                    segments_id=segments_id,
                    segments_mask=segments_mask
                )
        elif len(segments_id) == 1:
            # (n_tokens, hidden_dim), (n_heads, n_tokens, n_tokens)
            token_vectors, attentions \
                = self.encode_tokens_with_independent_segments(
                    segments_id=segments_id,
                    segments_mask=segments_mask
                )
        else:
            # (n_tokens, hidden_dim), (n_heads, n_tokens, n_tokens)
            token_vectors, attentions \
                = self.encode_tokens_with_overlapping_segments(
                    segments_id=segments_id,
                    segments_mask=segments_mask
                )

        # Compute mention vectors
        # (n_mentions, hidden_dim)
        mention_vectors = self.compute_mention_vectors(
            token_vectors=token_vectors,
            mention_begin_token_indices=mention_begin_token_indices
        )

        # Compute entity vectors
        # (n_entities, hidden_dim)
        entity_vectors = self.compute_entity_vectors(
            mention_vectors=mention_vectors,
            entity_index_to_mention_indices=entity_index_to_mention_indices
        )

        # Expand the entity vectors to entity pairs
        # (n_entity_pairs, hidden_dim), (n_entity_pairs, hidden_dim)
        (
            pair_head_entity_vectors,
            pair_tail_entity_vectors
        ) = self.expand_entity_vectors(
            entity_vectors=entity_vectors,
            pair_head_entity_indices=pair_head_entity_indices,
            pair_tail_entity_indices=pair_tail_entity_indices
        )

        # Compute entity-pair context vectors (Localized Context Pooling)
        if self.use_localized_context_pooling:
            # (n_entity_pairs, hidden_dim)
            pair_context_vectors = self.compute_entity_pair_context_vectors(
                token_vectors=token_vectors,
                attentions=attentions,
                entity_index_to_mention_indices=entity_index_to_mention_indices,
                mention_begin_token_indices=mention_begin_token_indices,
                pair_head_entity_indices=pair_head_entity_indices,
                pair_tail_entity_indices=pair_tail_entity_indices
            )
        else:
            # (n_entity_pairs, hidden_dim)
            pair_context_vectors = torch.zeros(
                pair_head_entity_vectors.shape,
                device=self.device
            )

        # Compute logits by block bilinear
        # (n_entity_pairs, n_relations)
        logits = self.compute_logits_by_block_bilinear(
            pair_head_entity_vectors=pair_head_entity_vectors,
            pair_tail_entity_vectors=pair_tail_entity_vectors,
            pair_context_vectors=pair_context_vectors
        )

        if not compute_loss:
            return ModelOutput(
                logits=logits
            )

        # Compute loss (summed over valid pairs)
        # (n_entity_pairs,)
        loss = self.loss_function(logits, pair_gold_relation_labels)
        loss = loss.sum() # Scalar

        # Compute accuracy (summed over valid triples)
        # (n_entity_pairs, n_relations)
        pair_pred_relation_labels = self.loss_function.get_labels(logits=logits)
        # (n_entity_pairs, n_relations)
        acc = (
            pair_pred_relation_labels == pair_gold_relation_labels
        ).to(torch.float)
        acc = acc.sum().item()

        n_valid_pairs, n_relations = pair_gold_relation_labels.shape
        n_valid_triples = n_valid_pairs * n_relations

        return ModelOutput(
            logits=logits,
            loss=loss,
            acc=acc,
            n_valid_pairs=n_valid_pairs,
            n_valid_triples=n_valid_triples
        )

    ################
    # Subfunctions
    ################

    def encode_tokens_with_independent_segments(
        self,
        segments_id: torch.Tensor,
        segments_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Shapes
        ----------
        segments_id : (n_segments, max_seg_len)
        segments_mask : (n_segments, max_seg_len)
        Output 1 : (n_tokens, hidden_dim)
        Output 1 : (n_heads, n_tokens, n_tokens)
        """
        # Check
        n_segments, max_seg_len = segments_id.shape

        # Encode segments by BERT
        bert_output = self.bert(
            input_ids=segments_id,
            attention_mask=segments_mask,
            output_attentions=True,
            output_hidden_states=False
        )
        # (n_segments, max_seg_len, hidden_dim)
        segments_vec = bert_output["last_hidden_state"]
        # (n_bert_layers, n_segments, n_heads, max_seg_len, max_seg_len)
        segments_att = bert_output["attentions"]
        # (n_segments, n_heads, max_seg_len, max_seg_len)
        segments_att = segments_att[-1]

        # Transform `segments_vec` to token vectors
        # (n_segments, max_seg_len)
        segments_mask_bool = segments_mask.to(torch.bool)
        # (n_tokens, hidden_dim)
        token_vectors = segments_vec[segments_mask_bool]

        # Get real token spans
        # (n_segments,)
        n_tokens_in_each_seg = segments_mask.sum(dim=1).int().tolist()
        start_in_each_seg = [] # list[int]
        end_in_each_seg = [] # list[int]
        offset = 0
        for n_tokens_in_this_seg in n_tokens_in_each_seg:
            start_in_each_seg.append(offset)
            end_in_each_seg.append(offset + n_tokens_in_this_seg - 1)
            offset += n_tokens_in_this_seg
        n_tokens = sum(n_tokens_in_each_seg)

        # Transform `segments_att` to token-to-token attentions
        attentions = []
        for seg_i in range(n_segments):
            n_tokens_in_this_seg = n_tokens_in_each_seg[seg_i]
            start_in_this_seg = start_in_each_seg[seg_i]
            end_in_this_seg = end_in_each_seg[seg_i]
            pad_left = start_in_this_seg
            pad_right = n_tokens - end_in_this_seg - 1
            pad_top = pad_left
            pad_bottom = pad_right
            # (n_heads, n_tokens_in_this_seg, n_tokens_in_this_seg)
            att = segments_att[
                seg_i,
                :,
                0:n_tokens_in_this_seg,
                0:n_tokens_in_this_seg
            ]

            # We mask the attention weights to [CLS] and [SEP] tokens
            # In-Place Operation Error
            # att[:, :, 0] = 0.0
            # att[:, :, -1] = 0.0
            temp_mask = torch.ones_like(att, device=self.device)
            temp_mask[:, :, 0] = 0.0
            temp_mask[:, :, -1] = 0.0
            att = temp_mask * att

            # (n_heads, n_tokens, n_tokens)
            att = F.pad(att, (pad_left, pad_right, pad_top, pad_bottom, 0, 0))
            attentions.append(att.unsqueeze(0))
        # (n_segments, n_heads, n_tokens, n_tokens)
        attentions = torch.cat(attentions, dim=0)
        # (n_heads, n_tokens, n_tokens)
        attentions = torch.sum(attentions, dim=0)
        attentions = attentions / (attentions.sum(dim=2, keepdim=True) + 1e-10)
        assert attentions.shape[1] == n_tokens
        assert attentions.shape[2] == n_tokens

        return token_vectors, attentions

    def encode_tokens_with_overlapping_segments(
        self,
        segments_id: torch.Tensor,
        segments_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Shapes
        ----------
        segments_id : (n_segments, max_seg_len)
        segments_mask : (n_segments, max_seg_len)
        Output 1 : (n_tokens, hidden_dim)
        Output 2 : (n_heads, n_tokens, n_tokens)
        """
        # Check
        n_segments, max_seg_len = segments_id.shape
        assert n_segments > 1

        # Combine adjacent two segments
        combined_segments_id = []
        combined_segments_mask = []
        for seg_i in range(0, n_segments - 1):
            new_seg_id, new_seg_mask = self.combine_two_segments(
                seg0_id=segments_id[seg_i],
                seg0_mask=segments_mask[seg_i],
                seg1_id=segments_id[seg_i+1],
                seg1_mask=segments_mask[seg_i+1],
                double_max_seg_len=2*max_seg_len
            )
            combined_segments_id.append(new_seg_id.unsqueeze(0))
            combined_segments_mask.append(new_seg_mask.unsqueeze(0))
        # (n_segments - 1, 2*max_seg_len)
        combined_segments_id = torch.cat(combined_segments_id, dim=0)
        # (n_segments - 1, 2*max_seg_len)
        combined_segments_mask = torch.cat(combined_segments_mask, dim=0)

        # Encode the combined segments by BERT
        bert_output = self.bert(
            input_ids=combined_segments_id,
            attention_mask=combined_segments_mask,
            output_attentions=True,
            output_hidden_states=False
        )
        # (n_segments - 1, 2*max_seg_len, hidden_dim)
        combined_segments_vec = bert_output["last_hidden_state"]
        # (n_bert_layers, n_segments-1, n_heads, 2*max_seg_len, 2*max_seg_len)
        combined_segments_att = bert_output["attentions"]
        # (n_segments - 1, n_heads, 2*max_seg_len, 2*max_seg_len)
        combined_segments_att = combined_segments_att[-1]

        # Get real token spans
        # (n_segments,)
        n_tokens_in_each_seg = segments_mask.sum(dim=1).int().tolist()
        n_tokens_in_each_comb_seg: list[int] = []
        start_in_each_comb_seg: list[int] = []
        end_in_each_comb_seg: list[int] = []
        offset = 0
        for i in range(0, n_segments - 1):
            n1 = n_tokens_in_each_seg[i]
            n2 = n_tokens_in_each_seg[i+1]
            n_tokens_in_each_comb_seg.append(n1 + n2)
            start_in_each_comb_seg.append(offset)
            end_in_each_comb_seg.append(offset + n1 + n2 - 1)
            offset += n1
        n_tokens = sum(n_tokens_in_each_seg)

        # Transform the token vectors and the attentions
        token_vectors = []
        masks = []
        attentions = []
        for seg_i in range(0, n_segments - 1):
            n_tokens_in_this_comb_seg = n_tokens_in_each_comb_seg[seg_i]
            start_in_this_comb_seg = start_in_each_comb_seg[seg_i]
            end_in_this_comb_seg = end_in_each_comb_seg[seg_i]
            pad_left = start_in_this_comb_seg
            pad_right = n_tokens - end_in_this_comb_seg - 1
            pad_top = pad_left
            pad_bottom = pad_right
            # (n_tokens_in_this_comb_seg, hidden_dim)
            vec = combined_segments_vec[
                seg_i, 0:n_tokens_in_this_comb_seg, :
            ]
            # (n_tokens_in_this_comb_seg,)
            mask = combined_segments_mask[
                seg_i, 0:n_tokens_in_this_comb_seg
            ]
            # (n_heads, n_tokens_in_this_comb_seg, n_tokens_in_this_comb_seg)
            att = combined_segments_att[
                seg_i,
                :,
                0:n_tokens_in_this_comb_seg,
                0:n_tokens_in_this_comb_seg
            ]

            # We mask the attention weights to [CLS] and [SEP] tokens
            # In-Place Operation Error
            # att[:, :, 0] = 0.0
            # att[:, :, -1] = 0.0
            temp_mask = torch.ones_like(att, device=self.device)
            temp_mask[:, :, 0] = 0.0
            temp_mask[:, :, -1] = 0.0
            att = temp_mask * att

            # (n_tokens, hidden_dim)
            vec = F.pad(vec, (0, 0, pad_top, pad_bottom))
            # (n_tokens,)
            mask = F.pad(mask, (pad_top, pad_bottom))
            # (n_heads, n_tokens, n_tokens)
            att = F.pad(att, (pad_left, pad_right, pad_top, pad_bottom, 0, 0))
            token_vectors.append(vec.unsqueeze(0))
            masks.append(mask.unsqueeze(0))
            attentions.append(att.unsqueeze(0))
        # (n_segments - 1, n_tokens, hidden_dim)
        token_vectors = torch.cat(token_vectors, dim=0)
        # (n_tokens, hidden_dim)
        token_vectors = torch.sum(token_vectors, dim=0)
        # (n_segments - 1, n_tokens)
        masks = torch.cat(masks, dim=0)
        # (n_tokens,)
        masks = torch.sum(masks, dim=0) + 1e-10
        # (n_tokens, hidden_dim)
        token_vectors = token_vectors / masks.unsqueeze(-1)
        # (n_segments - 1, n_heads, n_tokens, n_tokens)
        attentions = torch.cat(attentions, dim=0)
        # (n_heads, n_tokens, n_tokens)
        attentions = torch.sum(attentions, dim=0)
        attentions = attentions / (attentions.sum(dim=2, keepdim=True) + 1e-10)
        # assert token_vectors.shape[0] == n_tokens
        # assert masks.shape[0] == n_tokens
        # assert attentions.shape[1] == n_tokens
        # assert attentions.shape[2] == n_tokens

        return token_vectors, attentions

    def combine_two_segments(
        self,
        seg0_id: torch.Tensor,
        seg0_mask: torch.Tensor,
        seg1_id: torch.Tensor,
        seg1_mask: torch.Tensor,
        double_max_seg_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Shapes
        ----------
        seg0_id : (max_seg_len,)
        seg0_mask : (max_seg_len,)
        seg1_id : (max_seg_len,)
        seg1_mask : (max_seg_len,)
        Output 1 : (double_max_seg_len,)
        Output 2 : (double_max_seg_len,)
        """
        # Combine input IDs
        # (n_tokens_in_seg0,)
        seg0_id_wo_pad = seg0_id[seg0_mask.to(torch.bool)]
        # (n_tokens_in_seg1,)
        seg1_id_wo_pad = seg1_id[seg1_mask.to(torch.bool)]
        # (n_tokens_in_this_comb_seg,)
        new_seg_id = torch.cat([seg0_id_wo_pad, seg1_id_wo_pad], dim=0)

        # Create attention mask
        # new_seg_mask = [1] * len(new_seg_id)
        new_seg_mask = (
            [1] * (len(seg0_id_wo_pad) - 1)
            + [0, 0]
            + [1] * (len(seg1_id_wo_pad) - 1)
        )

        # Padding
        pad_len = double_max_seg_len - len(new_seg_id)
        # (double_max_seg_len,)
        new_seg_id = F.pad(new_seg_id, (0, pad_len))
        # (double_max_seg_len,)
        new_seg_mask = new_seg_mask + [0] * pad_len
        new_seg_mask = torch.tensor(new_seg_mask, device=self.device)

        return new_seg_id, new_seg_mask

    def compute_mention_vectors(
        self,
        token_vectors: torch.Tensor,
        mention_begin_token_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Shapes
        ----------
        token_vectors : (n_tokens, hidden_dim)
        mention_begin_token_indices : (n_mentions,)
        Output : (n_mentions, hidden_dim)
        """
        # (n_mentions, hidden_dim)
        mention_vectors = token_vectors[mention_begin_token_indices]
        return mention_vectors

    def compute_entity_vectors(
        self,
        mention_vectors: torch.Tensor,
        entity_index_to_mention_indices: list[torch.Tensor]
    ) -> torch.Tensor:
        """
        Shapes
        ----------
        mention_vectors : (n_mentions, hidden_dim)
        entity_index_to_mention_indices : list of length n_entities; each tensor shape: (n_mentions_for_this_entity,)
        Output : (n_entities, hidden_dim)
        """
        entity_vectors = []
        for mention_indices in entity_index_to_mention_indices:
            # Extract mention vectors for this entity
            # (n_mentions_for_this_entiy, hidden_dim)
            mention_vectors_subset = mention_vectors[mention_indices]
            # Pool the mention vectors
            if self.entity_pooling_method == "sum":
                # (1, hidden_dim)
                entity_vector = torch.sum(
                    mention_vectors_subset,
                    dim=0
                ).unsqueeze(0)
            elif self.entity_pooling_method == "mean":
                # (1, hidden_dim)
                entity_vector = torch.mean(
                    mention_vectors_subset,
                    dim=0
                ).unsqueeze(0)
            elif self.entity_pooling_method == "max":
                # (1, hidden_dim)
                entity_vector = torch.max(
                    mention_vectors_subset,
                    dim=0
                )[0].unsqueeze(0)
            elif self.entity_pooling_method == "logsumexp":
                # (1, hidden_dim)
                entity_vector \
                    = mention_vectors_subset.logsumexp(dim=0).unsqueeze(0)
            else:
                raise Exception(
                    f"Invalid entity_pooling_method={self.entity_pooling_method}"
                )
            entity_vectors.append(entity_vector)
        # (n_entities, hidden_dim)
        entity_vectors = torch.cat(entity_vectors, dim=0)
        return entity_vectors

    def expand_entity_vectors(
        self,
        entity_vectors: torch.Tensor,
        pair_head_entity_indices: torch.Tensor,
        pair_tail_entity_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Shapes
        ----------
        entity_vectors : (n_entities, hidden_dim)
        pair_head_entity_indices : (n_entity_pairs,)
        pair_tail_entity_indices : (n_entity_pairs,)
        Output 1 : (n_entity_pairs, hidden_dim)
        Output 2 : (n_entity_pairs, hidden_dim)
        """
        # Expand the entity vectors
        # (n_entity_pairs, hidden_dim)
        pair_head_entity_vectors = entity_vectors[pair_head_entity_indices]
        # (n_entity_pairs, hidden_dim)
        pair_tail_entity_vectors = entity_vectors[pair_tail_entity_indices]
        return pair_head_entity_vectors, pair_tail_entity_vectors

    def compute_entity_pair_context_vectors(
        self,
        token_vectors: torch.Tensor,
        attentions: torch.Tensor,
        entity_index_to_mention_indices: list[torch.Tensor],
        mention_begin_token_indices: torch.Tensor,
        pair_head_entity_indices: torch.Tensor,
        pair_tail_entity_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Shapes
        ----------
        token_vectors : (n_tokens, hidden_dim)
        attentions : (n_heads, n_tokens, n_tokens)
        entity_index_to_mention_indices : list of length n_entities; each tensor shape: (n_mentions_for_this_entity,)
        mention_begin_token_indices : (n_mentions,)
        pair_head_entity_indices : (n_entity_pairs,)
        pair_tail_entity_indices : (n_entity_pairs,)
        Output : (n_entity_pairs, hidden_dim)
        """
        # Pool the attentions over the heads, c.f., SAIS (Xiao et al., 2022)
        # (n_tokens, n_tokens)
        attentions = torch.sum(attentions, dim=0)

        # Compute entity-level attentions
        entity_attentions = []
        for mention_indices in entity_index_to_mention_indices:
            # Compute mention-level attentions
            # (n_mentions_for_this_entity, n_tokens)
            mention_attentions = attentions[
                mention_begin_token_indices[mention_indices],
                :
            ]
            # Pool the mention-level attentions
            # (n_tokens,)
            entity_att = torch.mean(mention_attentions, dim=0)
            entity_attentions.append(entity_att.unsqueeze(0))
        # (n_entities, n_tokens)
        entity_attentions = torch.cat(entity_attentions, dim=0)

        # Compute entity-pair-level attentions
        # (n_entity_pairs, n_tokens)
        pair_head_attentions = entity_attentions[pair_head_entity_indices]
        # (n_entity_pairs, n_tokens)
        pair_tail_attentions = entity_attentions[pair_tail_entity_indices]
        # (n_entity_pairs, n_tokens)
        pair_attentions = pair_head_attentions * pair_tail_attentions

        # Compute entity-pair-level context vectors
        # (n_entity_pairs, n_tokens)
        pair_attentions = pair_attentions / (
            pair_attentions.sum(dim=1, keepdim=True) + 1e-10
        )
        # (n_entity_pairs, hidden_dim)
        # pair_context_vectors = torch.matmul(pair_attentions, token_vectors)
        pair_context_vectors = contract(
            "ld,rl->rd",
            token_vectors,
            pair_attentions
        )

        return pair_context_vectors

    def compute_logits_by_block_bilinear(
        self,
        pair_head_entity_vectors: torch.Tensor,
        pair_tail_entity_vectors: torch.Tensor,
        pair_context_vectors: torch.Tensor
    ) -> torch.Tensor:
        """
        Shapes
        ----------
        pair_head_entity_vectors : (n_entity_pairs, hidden_dim)
        pair_tail_entity_vectors : (n_entity_pairs, hidden_dim)
        pair_context_vectors : (n_entity_pairs, hidden_dim)
        Output : (n_entity_pairs, n_relations)
        """
        n_entity_pairs = len(pair_head_entity_vectors)

        zh = torch.cat([pair_head_entity_vectors, pair_context_vectors], dim=1)
        zh = torch.tanh(self.linear_head(zh))

        zt = torch.cat([pair_tail_entity_vectors, pair_context_vectors], dim=1)
        zt = torch.tanh(self.linear_tail(zt))

        zh = zh.view(
            n_entity_pairs,
            self.hidden_dim // self.bilinear_block_size,
            self.bilinear_block_size
        )
        zt = zt.view(
            n_entity_pairs,
            self.hidden_dim // self.bilinear_block_size,
            self.bilinear_block_size
        )
        # (n_entity_pairs, hidden_dim * bilinear_block_size)
        input_to_block_bilinear = (
            zh.unsqueeze(3) * zt.unsqueeze(2)
        ).view(n_entity_pairs, self.hidden_dim * self.bilinear_block_size)

        # (n_entity_pairs, n_relations)
        logits = self.block_bilinear(input_to_block_bilinear)

        return logits


class ATLOPPreprocessor:

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_seg_len: int,
        vocab_relation: dict[str, int],
        possible_head_entity_types: list[str] | None = None,
        possible_tail_entity_types: list[str] | None = None
    ):
        self.tokenizer = tokenizer
        self.max_seg_len = max_seg_len
        self.vocab_relation = vocab_relation
        self.possible_head_entity_types = possible_head_entity_types
        self.possible_tail_entity_types = possible_tail_entity_types

        self.cls_token = tokenizer.cls_token
        self.sep_token = tokenizer.sep_token

        # self.special_mention_begin_marker = "<e>"
        # self.special_mention_end_marker = "</e>"
        self.special_mention_begin_marker = "*"
        self.special_mention_end_marker = "*"

    def preprocess(self, document: Document) -> dict[str, Any]:
        preprocessed_data = OrderedDict()

        #####
        # doc_key: str
        # sentences: list[list[str]]
        # mentions: list[MentionTuple]
        # entities: list[EntityTuple]
        # relations: list[TripleTuple]
        #####

        preprocessed_data["doc_key"] = document["doc_key"]

        sentences = [s.split() for s in document["sentences"]]
        preprocessed_data["sentences"] = sentences

        mentions = [
            MentionTuple(
                tuple(m["span"]),
                m["name"],
                m["entity_type"],
                m["entity_id"]
            )
            for m in document["mentions"]
        ] # list[MentionTuple]
        preprocessed_data["mentions"] = mentions

        entities = [
            EntityTuple(e["mention_indices"], e["entity_type"], e["entity_id"])
            for e in document["entities"]
        ] # list[EntityTuple]
        preprocessed_data["entities"] = entities

        with_supervision = True if "relations" in document else False
        if with_supervision:
            relations = [
                TripleTuple(r["arg1"], r["relation"], r["arg2"])
                for r in document["relations"]
            ] # list[TripleTuple]
            preprocessed_data["relations"] = relations

        #####
        # mention_index_to_sentence_index: list[int]
        # sentence_index_to_mention_indices: list[list[int]]
        # mention_index_to_entity_index: list[int]
        #####

        # Mention index to sentence index
        token_index_to_sent_index: list[int] = []
        for sent_i, sent in enumerate(sentences):
            token_index_to_sent_index.extend([sent_i for _ in range(len(sent))])
        mention_index_to_sentence_index: list[int] = []
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

        # Mention index to entity index
        # NOTE: Although a single mention may belong to multiple entities,
        #       we assign only one entity index to each mention.
        mention_index_to_entity_index: list[int] = [None] * len(mentions)
        for entity_i, entity in enumerate(entities):
            for mention_i in entity.mention_indices:
                mention_index_to_entity_index[mention_i] = entity_i
        preprocessed_data["mention_index_to_entity_index"] \
            = mention_index_to_entity_index

        #####
        # bert_input: dict[str, Any]
        # segments: list[list[str]]
        # segments_id: list[list[int]]
        # segments_mask: list[list[int]]
        # subtoken_index_to_word_index: list[int]
        # word_index_to_subtoken_indices: list[ist[int]]
        # subtoken_index_to_sentence_index: list[int]
        # mention_begin_token_indices: list[int]
        # mention_end_token_indices: list[int]
        #####

        (
            segments,
            segments_id,
            segments_mask,
            subtoken_index_to_word_index,
            word_index_to_subtoken_indices,
            subtoken_index_to_sentence_index,
            mention_begin_token_indices,
            mention_end_token_indices
        ) = self.tokenize_and_split(
            sentences=sentences,
            mentions=mentions
        )
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

        preprocessed_data["mention_begin_token_indices"] = mention_begin_token_indices
        preprocessed_data["mention_end_token_indices"] = mention_end_token_indices

        #####
        # entity_index_to_mention_indices: list[list[int]]
        #####

        # Entity index to mention indices
        entity_index_to_mention_indices = [e.mention_indices for e in entities]
        for ms in entity_index_to_mention_indices:
            assert len(ms) > 0
        preprocessed_data["entity_index_to_mention_indices"] \
            = entity_index_to_mention_indices

        #####
        # pair_head_entity_indices: list[int]
        # pair_tail_entity_indices: list[int]
        # pair_gold_relation_labels: list[list[int]]
        #####

        not_include_entity_pairs = None
        if "not_include_pairs" in document:
            # list[tuple[EntityIndex, EntityIndex]]
            epairs = [
                (epair["arg1"], epair["arg2"])
                for epair in document["not_include_pairs"]
            ]
            not_include_entity_pairs \
                = [(e1,e2) for e1,e2 in epairs] + [(e2,e1) for e1,e2 in epairs]

        pair_head_entity_indices: list[int] = []
        pair_tail_entity_indices: list[int] = []
        if with_supervision:
            pair_gold_relation_labels: list[list[int]] = []

        for head_entity_i in range(len(entities)):
            for tail_entity_i in range(len(entities)):
                # Skip diagonal
                if head_entity_i == tail_entity_i:
                    continue

                # Skip based on entity types if specified
                # e.g, Skip chemical-chemical, disease-disease,
                #      and disease-chemical pairs for CDR.
                if (
                    (self.possible_head_entity_types is not None)
                    and
                    (self.possible_tail_entity_types is not None)
                ):
                    head_entity_type = entities[head_entity_i].entity_type
                    tail_entity_type = entities[tail_entity_i].entity_type
                    if not (
                        (head_entity_type in self.possible_head_entity_types)
                        and
                        (tail_entity_type in self.possible_tail_entity_types)
                    ):
                        continue

                # Skip "not_include" pairs if specified
                if not_include_entity_pairs is not None:
                    if (head_entity_i, tail_entity_i) \
                            in not_include_entity_pairs:
                        continue

                pair_head_entity_indices.append(head_entity_i)
                pair_tail_entity_indices.append(tail_entity_i)

                if with_supervision:
                    rels = self.find_relations(
                        arg1=head_entity_i,
                        arg2=tail_entity_i,
                        relations=relations
                    )
                    multilabel_positive_indicators \
                        = [0] * len(self.vocab_relation)
                    if len(rels) == 0:
                        # Found no gold relation for this entity pair
                        multilabel_positive_indicators[0] = 1
                    else:
                        for rel in rels:
                            rel_id = self.vocab_relation[rel]
                            multilabel_positive_indicators[rel_id] = 1
                    pair_gold_relation_labels.append(
                        multilabel_positive_indicators
                    )

        pair_head_entity_indices = np.asarray(pair_head_entity_indices)
        pair_tail_entity_indices = np.asarray(pair_tail_entity_indices)
        preprocessed_data["pair_head_entity_indices"] = pair_head_entity_indices
        preprocessed_data["pair_tail_entity_indices"] = pair_tail_entity_indices
        if with_supervision:
            pair_gold_relation_labels = np.asarray(pair_gold_relation_labels)
            preprocessed_data["pair_gold_relation_labels"] = pair_gold_relation_labels

        return preprocessed_data

    #####
    # Subfunctions
    #####

    def tokenize_and_split(
        self,
        sentences: list[list[str]],
        mentions: list[MentionTuple]
    ) -> tuple[
        list[list[str]],
        list[list[int]],
        list[list[int]],
        list[int],
        list[list[int]],
        list[int],
        list[int],
        list[int]
    ]:
        # subtoken分割、subtoken単位でのtoken終点、文終点、メンション始点、
        # メンション終点のindicatorsの作成
        (
            sents_subtoken,
            sents_token_end,
            sents_sentence_end,
            subtoken_index_to_word_index,
            sents_mention_begin,
            sents_mention_end
        ) = self.tokenize(
            sentences=sentences,
            mentions=mentions
        )

        # 1階層のリストに変換
        doc_subtoken = utils.flatten_lists(sents_subtoken) # list[str]
        doc_token_end = utils.flatten_lists(sents_token_end) # list[bool]
        doc_sentence_end = utils.flatten_lists(sents_sentence_end) # list[bool]

        doc_mention_begin = utils.flatten_lists(sents_mention_begin)
        doc_mention_end = utils.flatten_lists(sents_mention_end)

        assert sum(doc_mention_begin) == sum(doc_mention_end)

        # BERTセグメントに分割
        (
            segments,
            subtoken_index_to_word_index,
            word_index_to_subtoken_indices,
            subtoken_index_to_sentence_index,
            mention_begin_token_indices,
            mention_end_token_indices
        ) = self.split(
            doc_subtoken=doc_subtoken,
            doc_sentence_end=doc_sentence_end,
            doc_token_end=doc_token_end,
            subtoken_index_to_word_index=subtoken_index_to_word_index,
            doc_mention_begin=doc_mention_begin,
            doc_mention_end=doc_mention_end
        )

        assert (
            len(mention_begin_token_indices)
            == len(mention_end_token_indices)
            == len(mentions)
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
            subtoken_index_to_sentence_index,
            mention_begin_token_indices,
            mention_end_token_indices
        )

    def tokenize(
        self,
        sentences: list[list[str]],
        mentions: list[MentionTuple]
    ) -> tuple[
        list[list[str]],
        list[list[bool]],
        list[list[bool]],
        list[int],
        list[list[bool]],
        list[list[bool]]
    ]:
        # subtokens
        sents_subtoken: list[list[str]] = []
        # subtoken-level indicators of token end point
        sents_token_end: list[list[bool]] = []
        # subtoken-level indicators of sentence end point
        sents_sentence_end: list[list[bool]] = []
        subtoken_index_to_word_index: list[int] = []
        # subtoken-level indicators of mention beginning point
        sents_mention_begin: list[list[bool]] = []
        # subtoken-level indicators of mention end point
        sents_mention_end: list[list[bool]] = []

        original_mention_begin_token_indices = np.asarray(
            [m.span[0] for m in mentions] + [-1]
        )
        original_mention_end_token_indices = np.asarray(
            [m.span[1] for m in mentions] + [-1]
        )

        word_idx = -1
        offset = 0
        for sent in sentences:
            sent_subtoken: list[str] = []
            sent_token_end: list[bool] = []
            sent_sentence_end: list[bool] = []
            sent_mention_begin: list[bool] = []
            sent_mention_end: list[bool] = []
            for token_i, token in enumerate(sent):
                word_idx += 1
                # もしこのトークン (token_i) がメンションの開始位置ならば、
                #   mention markerを挿入する.
                if (offset + token_i) in original_mention_begin_token_indices:
                    # サブトークン
                    sent_subtoken += [self.special_mention_begin_marker]
                    # トークンの終了位置
                    sent_token_end += [False]
                    # 文の終了位置
                    sent_sentence_end += [False]
                    # subtoken index -> word index
                    subtoken_index_to_word_index += [word_idx]
                    # メンションの開始位置、終了位置
                    # sent_mention_begin += [True]
                    # sent_mention_end += [False]
                    count = len(np.where(
                        original_mention_begin_token_indices
                        == (offset + token_i)
                    )[0])
                    assert count > 0
                    sent_mention_begin += [count]
                    sent_mention_end += [0]

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
                # メンションの開始位置、終了位置
                # sent_mention_begin += [False] * len(subtokens)
                # sent_mention_end += [False] * len(subtokens)
                sent_mention_begin += [0] * len(subtokens)
                sent_mention_end += [0] * len(subtokens)

                # もしこのトークン (token_i) がメンションの終了位置ならば、
                #   mention markerを挿入する.
                if (offset + token_i) in original_mention_end_token_indices:
                    # サブトークン
                    sent_subtoken += [self.special_mention_end_marker]
                    # トークンの終了位置
                    sent_token_end[-1] = False # トークンの終了位置をずらす
                    sent_token_end += [True]
                    # 文の終了位置 (仮)
                    sent_sentence_end += [False]
                    # subtoken index -> word index
                    subtoken_index_to_word_index += [word_idx]
                    # メンションの開始位置、終了位置
                    # sent_mention_begin += [False]
                    # sent_mention_end += [True]
                    sent_mention_begin += [0]
                    count = len(np.where(
                        original_mention_end_token_indices
                        == (offset + token_i)
                    )[0])
                    assert count > 0
                    sent_mention_end += [count]

                # メンションマーカーを挿入する場合は削除
                # for i in np.where(
                #     original_mention_begin_token_indices == (offset + token_i)
                # )[0]:
                #     sent_mention_begin[-len(subtokens)] += 1
                # for i in np.where(
                #     original_mention_end_token_indices == (offset + token_i)
                # )[0]:
                #     sent_mention_end[-1] += 1
            # 文の終了位置
            sent_sentence_end[-1] = True
            assert sum(sent_mention_begin) == sum(sent_mention_end)

            sents_subtoken.append(sent_subtoken)
            sents_token_end.append(sent_token_end)
            sents_sentence_end.append(sent_sentence_end)
            sents_mention_begin.append(sent_mention_begin)
            sents_mention_end.append(sent_mention_end)

            offset += len(sent)

        return (
            sents_subtoken,
            sents_token_end,
            sents_sentence_end,
            subtoken_index_to_word_index,
            sents_mention_begin,
            sents_mention_end
        )

    def split(
        self,
        doc_subtoken: list[str],
        doc_sentence_end: list[bool],
        doc_token_end: list[bool],
        subtoken_index_to_word_index: list[int],
        doc_mention_begin: list[bool],
        doc_mention_end: list[bool]
    ) -> tuple[
        list[list[str]],
        list[int],
        list[list[int]],
        list[int],
        list[int],
        list[int] 
    ]:
        segments: list[list[str]] = []
        segments_subtoken_map: list[list[int]] = []
        segments_mention_begin: list[list[bool]] = []
        segments_mention_end: list[list[bool]] = []

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
                seg_before = "\"" + " ".join(doc_subtoken[curr_idx:end_idx+1]) + "\""
                while end_idx >= curr_idx and not doc_token_end[end_idx]:
                    end_idx -= 1
                if end_idx < curr_idx:
                    logger.warning("Cannot split valid segment: no sentence end or token end")
                    raise Exception
                seg_after = "\"" + " ".join(doc_subtoken[curr_idx:end_idx+1]) + "\""
                logger.warning("------")
                logger.warning("Segment where no sentence-ending position was found:")
                logger.warning(seg_before)
                logger.warning("---")
                logger.warning("Segment splitted based on a token-ending position:")
                logger.warning(seg_after)
                logger.warning("------")

            segment = doc_subtoken[curr_idx: end_idx + 1]
            segment_subtoken_map = subtoken_index_to_word_index[curr_idx: end_idx + 1]
            segment_mention_begin = doc_mention_begin[curr_idx: end_idx + 1]
            segment_mention_end = doc_mention_end[curr_idx: end_idx + 1]
            segment = [self.cls_token] + segment + [self.sep_token]
            # NOTE: [CLS] is treated as the first subtoken of the first word for each segment
            # NOTE: [SEP] is treated as the last subtoken of the last word for each segment
            segment_subtoken_map = (
                [segment_subtoken_map[0]]
                + segment_subtoken_map
                + [segment_subtoken_map[-1]]
            )
            # segment_mention_begin = [False] + segment_mention_begin + [False]
            # segment_mention_end = [False] + segment_mention_end + [False]
            segment_mention_begin = [0] + segment_mention_begin + [0]
            segment_mention_end = [0] + segment_mention_end + [0]

            segments.append(segment)
            segments_subtoken_map.append(segment_subtoken_map)
            segments_mention_begin.append(segment_mention_begin)
            segments_mention_end.append(segment_mention_end)

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

        # Subtoken indices for mention spans
        mention_begin_token_indices = utils.flatten_lists(
            [
                [subtok_i] * cnt \
                for subtok_i, cnt in enumerate(
                    utils.flatten_lists(segments_mention_begin)
                )
            ]
        )
        mention_end_token_indices = utils.flatten_lists(
            [
                [subtok_i] * cnt \
                for subtok_i, cnt in enumerate(
                    utils.flatten_lists(segments_mention_end)
                )
            ]
        )

        return (
            segments,
            subtoken_index_to_word_index,
            word_index_to_subtoken_indices,
            subtoken_index_to_sentence_index,
            mention_begin_token_indices,
            mention_end_token_indices
        )

    def get_word_index_to_subtoken_indices(
        self,
        segments_subtoken_map: list[list[int]]
    ) -> list[list[int]]:
        word_index_to_subtoken_indices = defaultdict(list)
        offset = 0
        for segment_subtoken_map in segments_subtoken_map:
            for subtok_i, word_i in enumerate(segment_subtoken_map):
                if subtok_i == 0 or subtok_i == len(segment_subtoken_map) - 1:
                    continue
                word_index_to_subtoken_indices[word_i].append(offset + subtok_i)
            offset += len(segment_subtoken_map)
        return word_index_to_subtoken_indices

    def get_subtoken_index_to_sentence_index(
        self,
        segments: list[list[str]],
        doc_sentence_end: list[bool]
    ) -> list[int]:
        assert len(doc_sentence_end) == sum([len(seg) - 2 for seg in segments])
        sent_map = []
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
        segments: list[list[str]]
    ) -> tuple[list[list[int]], list[list[int]]]:
        segments_id: list[list[int]] = []
        segments_mask: list[list[int]] = []
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
        return (segments_id, segments_mask)

    def find_relations(
        self,
        arg1: int,
        arg2: int,
        relations: list[TripleTuple]
    ) -> list[str]:
        rels = [] # list[str]
        for triple in relations:
            if triple.arg1 == arg1 and triple.arg2 == arg2:
                rels.append(triple.relation)
        return rels

