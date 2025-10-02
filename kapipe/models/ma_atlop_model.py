from typing import NamedTuple
from collections import OrderedDict

import numpy as np
# import spacy_alignments as tokenizations
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers.modeling_outputs import ModelOutput
from opt_einsum import contract

# from . import shared_functions
from .losses import AdaptiveThresholdingLoss
from .. import utils


class MentionTuple(NamedTuple):
    span: tuple[int, int] | None
    name: str
    entity_type: str
    entity_id: str


class EntityTuple(NamedTuple):
    mention_indices: list[int]
    entity_type: str
    entity_id: str
    is_dummy: bool


class TripleTuple(NamedTuple):
    arg1: int
    relation: str
    arg2: int


class MAATLOPModel(nn.Module):

    def __init__(
        self,
        device,
        bert_pretrained_name_or_path,
        max_seg_len,
        entity_dict,
        entity_seq_length,
        use_localized_context_pooling,
        bilinear_block_size,
        use_entity_loss,
        vocab_relation,
        possible_head_entity_types=None,
        possible_tail_entity_types=None,
        use_mention_as_canonical_name=False
    ):
        """
        Parameters
        ----------
        device : str
        bert_pretrained_name_or_path : str
        max_seg_len : int
        entity_dict : dict[str, EntityPage]
        entity_seq_length : int
        use_localized_context_pooling : bool
        bilinear_block_size : int
        use_entity_loss : bool
        vocab_relation : dict[str, int]
        possible_head_entity_types: list[str] | None
            by default None
        possible_tail_entity_types: list[str] | None
            by default None
        use_mention_as_canonical_name : bool
            by default False
        """
        super().__init__()

        ########################
        # Hyper parameters
        ########################

        self.device = device
        self.bert_pretrained_name_or_path = bert_pretrained_name_or_path
        self.max_seg_len = max_seg_len
        self.entity_dict = entity_dict
        self.entity_seq_length = entity_seq_length
        self.use_localized_context_pooling = use_localized_context_pooling
        self.bilinear_block_size = bilinear_block_size
        self.use_entity_loss = use_entity_loss
        self.vocab_relation = vocab_relation
        self.possible_head_entity_types = possible_head_entity_types
        self.possible_tail_entity_types = possible_tail_entity_types
        self.use_mention_as_canonical_name = use_mention_as_canonical_name

        self.n_relations = len(self.vocab_relation)

        ########################
        # Components
        ########################

        # BERT, tokenizer
        self.bert, self.tokenizer = self._initialize_bert_and_tokenizer(
            pretrained_model_name_or_path=self.bert_pretrained_name_or_path
        )

        # Dimensionality
        self.hidden_dim = self.bert.config.hidden_size

        # Entity-level classification
        if self.use_entity_loss:
            self.entity_linear = nn.Linear(self.hidden_dim, 2)

        # DocRE classification
        if self.use_localized_context_pooling:
            self.linear_head = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
            self.linear_tail = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        else:
            self.linear_head = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.linear_tail = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.block_bilinear = nn.Linear(
            self.hidden_dim * self.bilinear_block_size,
            self.n_relations
        )

        ########################
        # Preprocessor
        ########################

        self.preprocessor = MAATLOPPreprocessor(
            tokenizer=self.tokenizer,
            max_seg_len=self.max_seg_len,
            entity_dict=self.entity_dict,
            entity_seq_length=self.entity_seq_length,
            vocab_relation=self.vocab_relation,
            possible_head_entity_types=self.possible_head_entity_types,
            possible_tail_entity_types=self.possible_tail_entity_types,
            use_mention_as_canonical_name=self.use_mention_as_canonical_name
        )

        ########################
        # Loss function
        ########################

        self.entity_loss_function = nn.CrossEntropyLoss(reduction="none")
        self.pair_loss_function = AdaptiveThresholdingLoss()

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
        document
    ):
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

        # (n_entities, max_seg_len)
        model_input["segments_id"] = torch.tensor(
            preprocessed_data["bert_input"]["segments_id"],
            device=self.device
        )

        # (n_entities, max_seg_len)
        model_input["segments_mask"] = torch.tensor(
            preprocessed_data["bert_input"]["segments_mask"],
            device=self.device
        )

        # (n_entities, max_seg_len)
        model_input["segments_token_type_id"] = torch.tensor(
            preprocessed_data["bert_input"]["segments_token_type_id"],
            device=self.device
        )

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

        if self.use_entity_loss:
            # (n_entities,)
            model_input["entity_gold_labels"] = torch.tensor(
                preprocessed_data["entity_gold_labels"],
                device=self.device
            ).to(torch.long)

        # (n_entity_pairs, n_relations)
        model_input["pair_gold_relation_labels"] = torch.tensor(
            preprocessed_data["pair_gold_relation_labels"],
            device=self.device
        ).to(torch.float)

        return model_input

    def forward(
        self,
        segments_id,
        segments_mask,
        segments_token_type_id,
        pair_head_entity_indices,
        pair_tail_entity_indices,
        compute_loss,
        entity_gold_labels=None,
        pair_gold_relation_labels=None,
    ):
        """
        Parameters
        ----------
        segments_id : torch.Tensor
            shape of (n_entities, max_seg_len)
        segments_mask : torch.Tensor
            shape of (n_entities, max_seg_len)
        segments_token_type_id : torch.Tensor
            shape of (n_entities, max_seg_len)
        pair_head_entity_indices : torch.Tensor
            shape of (n_entity_pairs,)
        pair_tail_entity_indices : torch.Tensor
            shape of (n_entity_pairs,)
        compute_loss : bool
        entity_gold_labels : torch.Tensor | None
            shape of (n_entities,); by default None
        pair_gold_relation_labels : torch.Tensor | None
            shape of (n_entity_pairs, n_relations); by default None

        Returns
        -------
        ModelOutput
        """
        # Encode tokens by BERT
        # (n_entities, max_seg_len, hidden_dim)
        # (n_entities, n_heads, max_seg_len, max_seg_len)
        segments_vec, segments_att = self.encode_tokens(
            segments_id=segments_id,
            segments_mask=segments_mask,
            segments_token_type_id=segments_token_type_id
        )

        # Get [CLS] vectors
        # (n_entities, hidden_dim)
        entity_vectors = segments_vec[:, 0, :]

        if self.use_entity_loss:
            # Compute entity-level logits by a linear layer
            # (n_entities, 2)
            entity_logits = self.entity_linear(entity_vectors)

        # Expand the entity vectors to entity pair
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
                segments_vec=segments_vec,
                segments_att=segments_att,
                segments_mask=segments_mask,
                segments_token_type_id=segments_token_type_id,
                pair_head_entity_indices=pair_head_entity_indices,
                pair_tail_entity_indices=pair_tail_entity_indices
            )
        else:
            # (n_entity_pairs, hidden_dim)
            pair_context_vectors = None

        # Compute pair-level logits by block bilinear
        # (n_entity_pairs, n_relations)
        pair_logits = self.compute_logits_by_block_bilinear(
            pair_head_entity_vectors=pair_head_entity_vectors,
            pair_tail_entity_vectors=pair_tail_entity_vectors,
            pair_context_vectors=pair_context_vectors
        )

        if not compute_loss:
            if self.use_entity_loss:
                return ModelOutput(
                    entity_logits=entity_logits,
                    pair_logits=pair_logits
                )
            else:
                return ModelOutput(
                    pair_logits=pair_logits
                )

        if self.use_entity_loss:
            # Compute entity-level loss (summed over entities)
            # (n_entities,)
            entity_loss = self.entity_loss_function(
                entity_logits,
                entity_gold_labels
            )
            entity_loss = entity_loss.sum() # Scalar

            n_entities = len(entity_gold_labels)

        # Compute pair-level loss (summed over valid pairs)
        # (n_entity_pairs,)
        pair_loss = self.pair_loss_function(
            pair_logits,
            pair_gold_relation_labels
        )
        pair_loss = pair_loss.sum() # Scalar

        # Compute pair-level accuracy (summed over valid triples)
        # (n_entity_pairs, n_relations)
        pair_pred_relation_labels \
            = self.pair_loss_function.get_labels(logits=pair_logits)
        # (n_entity_pairs, n_relations)
        pair_acc = (
            pair_pred_relation_labels == pair_gold_relation_labels
        ).to(torch.float)
        pair_acc = pair_acc.sum().item() # float

        n_valid_pairs, n_relations = pair_gold_relation_labels.shape
        n_valid_triples = n_valid_pairs * n_relations

        if self.use_entity_loss:
            return ModelOutput(
                pair_logits=pair_logits,
                pair_loss=pair_loss,
                pair_acc=pair_acc,
                n_valid_pairs=n_valid_pairs,
                n_valid_triples=n_valid_triples,
                #
                entity_logits=entity_logits,
                entity_loss=entity_loss,
                n_entities=n_entities,
            )
        else:
            return ModelOutput(
                pair_logits=pair_logits,
                pair_loss=pair_loss,
                pair_acc=pair_acc,
                n_valid_pairs=n_valid_pairs,
                n_valid_triples=n_valid_triples,
            )

    ################
    # Subfunctions
    ################

    def encode_tokens(self, segments_id, segments_mask, segments_token_type_id):
        """
        Parameters
        ----------
        segments_id : torch.Tensor
            shape of (n_entities, max_seg_len)
        segments_mask : torch.Tensor
            shape of (n_entities, max_seg_len)
        segments_token_type_id : torch.Tensor
            shape of (n_entities, max_seg_len)

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            shape of (n_entities, max_seg_len, hidden_dim)
            shape of (n_entities, n_heads, max_seg_len, hidden_dim)
        """
        bert_output = self.bert(
            input_ids=segments_id,
            attention_mask=segments_mask,
            token_type_ids=segments_token_type_id,
            output_attentions=True,
            output_hidden_states=False
        )
        # (n_entities, max_seg_len, hidden_dim)
        segments_vec = bert_output["last_hidden_state"]
        # (n_bert_layers, n_entities, n_heads, max_seg_len, max_seg_len)
        segments_att = bert_output["attentions"]
        # (n_entities, n_heads, max_seg_len, max_seg_len)
        segments_att = segments_att[-1]
        return segments_vec, segments_att

    def expand_entity_vectors(
        self,
        entity_vectors,
        pair_head_entity_indices,
        pair_tail_entity_indices
    ):
        """
        Parameters
        ----------
        entity_vectors : torch.Tensor
            shape of (n_entities, hidden_dim)
        pair_head_entity_indices : torch.Tensor
            shape of (n_entity_pairs,)
        pair_tail_entity_indices : torch.Tensor
            shape of (n_entity_pairs,)

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            1. shape of (n_entity_pairs, hidden_dim)
            2. shape of (n_entity_pairs, hidden_dim)
        """
        # Expand the entity vectors
        # (n_entity_pairs, hidden_dim)
        pair_head_entity_vectors = entity_vectors[pair_head_entity_indices]
        # (n_entity_pairs, hidden_dim)
        pair_tail_entity_vectors = entity_vectors[pair_tail_entity_indices]
        return pair_head_entity_vectors, pair_tail_entity_vectors

    def compute_entity_pair_context_vectors(
        self,
        segments_vec,
        segments_att,
        segments_mask,
        segments_token_type_id,
        pair_head_entity_indices,
        pair_tail_entity_indices
    ):
        """
        Parameters
        ----------
        segments_vec : torch.Tensor
            shape of (n_entities, max_seg_len, hidden_dim)
        segments_att : torch.Tensor
            shape of (n_entities, n_heads, max_seg_len, max_seg_len)
        segments_mask : torch.Tensor
            shape of (n_entities, max_seg_len)
        segments_token_type_id : torch.Tensor
            shape of (n_entities, max_seg_len)
        pair_head_entity_indices : torch.Tensor
            shape of (n_entity_pairs,)
        pair_tail_entity_indices : torch.Tensor
            shape of (n_entity_pairs,)

        Returns
        -------
        torch.Tensor
            shape of (n_entity_pairs, hidden_dim)
        """
        n_entities = segments_vec.shape[0]

        # The number of text-side tokens
        # Remove "[SEP]" from the count
        # (n_entities,)
        text_seq_lengths = segments_token_type_id.sum(dim=1) - 1
        text_seq_mask = segments_token_type_id.to(torch.bool)

        # First, for each entity, obtain token embeddings in the text-side sequence
        entity_token_vectors = []
        for e_i in range(n_entities):
            # Remove the "[SEP]" embedding
            # (n_tokens_for_this_entity, hidden_dim)
            entity_tok_vecs = segments_vec[e_i, text_seq_mask[e_i]][:-1]
            # (max_seg_len, hidden_dim)
            entity_tok_vecs = F.pad(
                entity_tok_vecs,
                (0, 0, 0, self.max_seg_len - text_seq_lengths[e_i])
            )
            entity_token_vectors.append(entity_tok_vecs.unsqueeze(0))
        # (n_entities, max_seg_len, hidden_dim)
        entity_token_vectors = torch.cat(entity_token_vectors, dim=0)

        # Second, for each entity, obtain attentions to the text-side sequence
        # Pool the attentions over the heads, c.f., SAIS (Xiao et al., 2022)
        # (n_entities, max_seg_len, max_seg_len)
        segments_att = segments_att.sum(dim=1)
        # Extract attentions from the first token in the entity-side sequence
        #   to the text-side sequence.
        # index=1 is for the first entity-side token (index=0 is for [CLS])
        # (n_entities, max_seg_len)
        segments_att = segments_att[:, 1]
        entity_attentions = []
        for e_i in range(n_entities):
            # (n_tokens_for_this_entity,)
            entity_att = segments_att[e_i, text_seq_mask[e_i]][:-1]
            # (max_seg_len,)
            entity_att = F.pad(
                entity_att,
                (0, self.max_seg_len - text_seq_lengths[e_i])
            )
            entity_attentions.append(entity_att.unsqueeze(0))
        # (n_entities, max_seg_len)
        entity_attentions = torch.cat(entity_attentions, dim=0)
        # Normalize the entity-level attentions
        # entity_attentions = entity_attentions / (
        #     entity_attentions.sum(dim=1, keepdim=True) + 1e-10
        # )

        # Third, compute entity-pair-level attentions
        # (n_entity_pairs, max_seg_len)
        pair_head_attentions = entity_attentions[pair_head_entity_indices]
        # (n_entity_pairs, max_seg_len)
        pair_tail_attentions = entity_attentions[pair_tail_entity_indices]
        # (n_entity_pairs, max_seg_len)
        pair_attentions = pair_head_attentions * pair_tail_attentions
        # Normalize the pair-level attentions
        pair_attentions = pair_attentions / (
            pair_attentions.sum(dim=1, keepdim=True) + 1e-10
        )

        # Fourth, compute entity-pair-level token vecotrs
        # (n_entity_pairs, max_seg_len, hidden_dim)
        pair_head_token_vectors = entity_token_vectors[pair_head_entity_indices]
        # (n_entity_pairs, max_seg_len, hidden_dim)
        pair_tail_token_vectors = entity_token_vectors[pair_tail_entity_indices]
        # (n_entity_pairs, max_seg_len, hidden_dim)
        pair_token_vectors = pair_head_token_vectors + pair_tail_token_vectors

        # Lastly, compute entity-pair-level context vectors
        # (n_entity_pairs, hidden_dim)
        pair_context_vectors = contract(
            "pld,pl->pd",
            pair_token_vectors,
            pair_attentions
        )

        return pair_context_vectors

    def compute_logits_by_block_bilinear(
        self,
        pair_head_entity_vectors,
        pair_tail_entity_vectors,
        pair_context_vectors
    ):
        """
        Parameters
        ----------
        pair_head_entity_vectors : torch.Tensor
            shape of (n_entity_pairs, hidden_dim)
        pair_tail_entity_vectors : torch.Tensor
            shape of (n_entity_pairs, hidden_dim)
        pair_context_vectors : torch.Tensor | None
            shape of (n_entity_pairs, hidden_dim)

        Returns
        -------
        torch.Tensor
            shape of (n_entity_pairs, n_relations)
        """
        n_entity_pairs = len(pair_head_entity_vectors)

        if self.use_localized_context_pooling:
            zh = torch.cat(
                [pair_head_entity_vectors, pair_context_vectors],
                dim=1
            )
            zt = torch.cat(
                [pair_tail_entity_vectors, pair_context_vectors],
                dim=1
            )
        else:
            zh = pair_head_entity_vectors
            zt = pair_tail_entity_vectors

        zh = torch.tanh(self.linear_head(zh))
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


class MAATLOPPreprocessor:

    def __init__(
        self,
        tokenizer,
        max_seg_len,
        entity_dict,
        entity_seq_length,
        vocab_relation,
        possible_head_entity_types=None,
        possible_tail_entity_types=None,
        use_mention_as_canonical_name=False,
    ):
        """
        Parameters
        ----------
        tokenizer : PreTrainedTokenizer
        max_seg_len : int
        entity_dict : dict[str, EntityPage]
        entity_seq_length : int
        vocab_relation: dict[str, int]
        possible_head_entity_types: list[str] | None
            by default None
        possible_tail_entity_types: list[str] | None
            by default None
        use_mention_as_canonical_name : bool
            by default False
        """
        self.tokenizer = tokenizer
        self.max_seg_len = max_seg_len
        self.entity_dict = entity_dict
        self.entity_seq_length = entity_seq_length
        self.vocab_relation = vocab_relation
        self.possible_head_entity_types = possible_head_entity_types
        self.possible_tail_entity_types = possible_tail_entity_types
        self.use_mention_as_canonical_name = use_mention_as_canonical_name

        self.special_entity_sep_marker = ":"

    def preprocess(
        self,
        document
    ):
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
        # entities: list[EntityTuple]
        # relations: list[TripleTuple]
        #####

        preprocessed_data["doc_key"] = document["doc_key"]

        sentences = [s.split() for s in document["sentences"]]
        preprocessed_data["sentences"] = sentences

        mentions = [
            MentionTuple(
                None,
                m["name"],
                m["entity_type"],
                m["entity_id"]
            )
            for m in document["mentions"]
        ] # list[MentionTuple]
        preprocessed_data["mentions"] = mentions

        entities = [
            EntityTuple(
                e["mention_indices"],
                e["entity_type"],
                e["entity_id"],
                e["is_dummy"] if "is_dummy" in e else True
            )
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
        # mention_index_to_entity_index: list[int]
        #####

        # Mention index to entity index
        # NOTE: Although a single mention may belong to multiple entities,
        #   we assign only one entity index to each mention
        mention_index_to_entity_index = [None] * len(mentions) # list[int]
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
        # segments_token_type_id: list[list[int]]
        #####

        (
            segments,
            segments_id,
            segments_mask,
            segments_token_type_id
        ) = self.tokenize_and_split(
            sentences=sentences,
            entities=entities,
            mentions=mentions if self.use_mention_as_canonical_name else None
        )
        bert_input = {}
        bert_input["segments"] = segments
        bert_input["segments_id"] = segments_id
        bert_input["segments_mask"] = segments_mask
        bert_input["segments_token_type_id"] = segments_token_type_id
        preprocessed_data["bert_input"] = bert_input

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

        pair_head_entity_indices = [] # list[int]
        pair_tail_entity_indices = [] # list[int]
        if with_supervision:
            pair_gold_relation_labels = [] # list[list[int]]

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
            preprocessed_data["pair_gold_relation_labels"] \
                = pair_gold_relation_labels

        #####
        # entity_gold_labels: list[int]
        #####
        entity_gold_labels = [int(e.is_dummy) for e in entities]
        preprocessed_data["entity_gold_labels"] = entity_gold_labels

        return preprocessed_data

    #####
    # Subfunctions
    #####

    def tokenize_and_split(self, sentences, entities, mentions):
        """
        Parameters
        ----------
        sentences : list[list[str]]
        entities : list[EntityTuple]
        mentions : list[MentionTuple]

        Returns
        -------
        tuple[list[list[str]], list[list[int]], list[list[int]], list[list[int]]]
        """
        # Make entity-side sequence
        entity_seqs = [] # list[str]
        for ent in entities:
            entity_id = ent.entity_id
            epage = self.entity_dict[entity_id]
            if self.use_mention_as_canonical_name:
                m_i = ent.mention_indices[0]
                canonical_name = mentions[m_i].name
            else:
                canonical_name = epage["canonical_name"]
            # synonyms = epage["synonyms"]
            description = epage["description"]
            # "<canonical name> : <description>"
            entity_seq = " ".join([
                canonical_name,
                self.special_entity_sep_marker,
                description
            ])
            entity_seq = " ".join(entity_seq.split(" ")[:self.entity_seq_length])
            entity_seqs.append(entity_seq)
        # Make text-side sequence
        text_seq = " ".join(utils.flatten_lists(sentences)) # str
        # Combine and tokenize the sequences
        # [CLS] <canonical name> : <description> [SEP] <text> [SEP]
        inputs = self.tokenizer(
            entity_seqs,
            [text_seq] * len(entity_seqs),
            max_length=self.max_seg_len,
            padding=True,
            truncation="only_second",
            return_overflowing_tokens=False
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

    def find_relations(self, arg1, arg2, relations):
        """
        Parameters
        ----------
        arg1 : int
        arg2 : int
        relations : list[TripleTuple]

        Returns
        -------
        list[str]
        """
        rels = [] # list[str]
        for triple in relations:
            if triple.arg1 == arg1 and triple.arg2 == arg2:
                rels.append(triple.relation)
        return rels

