from collections import OrderedDict
from typing import NamedTuple
import logging

# import numpy as np
# import spacy_alignments as tokenizations
import torch
import torch.nn as nn
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers.modeling_outputs import ModelOutput

from . import shared_functions
from .losses import FocalLoss
from .. import utils


logger = logging.getLogger(__name__)


HEAD_ENTITY_SLOT = "@@@HEAD###"
TAIL_ENTITY_SLOT = "@@@TAIL###"
TRIPLE_TO_QUESTION_TEMPLATES = {
    "cdr": {
        "CID": "Does @@@HEAD### induce @@@TAIL### ?"
    },
    "hoip": {
        "has result": "Does @@@HEAD### result in @@@TAIL### ?",
        "has part": "Does @@@HEAD### involve @@@TAIL### ?",
        # "has molecular reaction": "Does @@@HEAD### have molecular reaction of @@@TAIL### ?",
        # "part of": "Is @@@HEAD### part of @@@TAIL### ?",
    }
}


class MentionTuple(NamedTuple):
    span: tuple[int, int] | None
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


class QATuple(NamedTuple):
    triple: TripleTuple
    question: list[str]
    answer: str | None


class MAQAModel(nn.Module):

    def __init__(
        self,
        device,
        bert_pretrained_name_or_path,
        max_seg_len,
        entity_dict,
        dataset_name,
        dropout_rate,
        vocab_answer,
        loss_function_name,
        focal_loss_gamma=None,
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
        dataset_name : str
        dropout_rate : float
        vocab_answer : dict[str, int]
        loss_function_name : str
        focal_loss_gamma : float | None
            by default None
        possible_head_entity_types : list[str] | None
            by default None
        possible_tail_entity_types : list[str] | None
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
        self.dataset_name = dataset_name
        self.dropout_rate = dropout_rate
        self.vocab_answer = vocab_answer
        self.loss_function_name = loss_function_name
        self.focal_loss_gamma = focal_loss_gamma
        self.possible_head_entity_types = possible_head_entity_types
        self.possible_tail_entity_types = possible_tail_entity_types
        self.use_mention_as_canonical_name = use_mention_as_canonical_name

        self.n_answers = len(self.vocab_answer)

        ########################
        # Components
        ########################

        # BERT, tokenizer
        self.bert, self.tokenizer = self._initialize_bert_and_tokenizer(
            pretrained_model_name_or_path=self.bert_pretrained_name_or_path
        )

        # Dimensionality
        self.hidden_dim = self.bert.config.hidden_size

        # QA
        self.mlp = shared_functions.make_mlp(
            input_dim=2 * self.hidden_dim,
            hidden_dims=2 * self.hidden_dim,
            output_dim=self.n_answers,
            dropout_rate=self.dropout_rate
        )

        ######
        # Preprocessor
        ######

        self.preprocessor = MAQAPreprocessor(
            tokenizer=self.tokenizer,
            max_seg_len=self.max_seg_len,
            entity_dict=self.entity_dict,
            dataset_name=self.dataset_name,
            vocab_answer=self.vocab_answer,
            possible_head_entity_types=self.possible_head_entity_types,
            possible_tail_entity_types=self.possible_tail_entity_types,
            use_mention_as_canonical_name=self.use_mention_as_canonical_name
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
            raise Exception(f"Invalid loss_function: {self.loss_function_name}")

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

    def tensorize(self, preprocessed_data, qa_index, compute_loss):
        """
        Parameters
        ----------
        preprocessed_data : dict[str, Any]
        qa_index : int
        compute_loss : bool

        Returns
        -------
        dict[str, Any]
        """
        model_input = {}

        model_input["compute_loss"] = compute_loss

        # (n_segments, max_seg_len)
        model_input["segments_id"] = torch.tensor(
            preprocessed_data["qa_index_to_bert_input"][qa_index]["segments_id"],
            device=self.device
        )

        # (n_segments, max_seg_len)
        model_input["segments_mask"] = torch.tensor(
            preprocessed_data["qa_index_to_bert_input"][qa_index]["segments_mask"],
            device=self.device
        )

        # (n_segments, max_seg_len)
        model_input["segments_token_type_id"] = torch.tensor(
            preprocessed_data["qa_index_to_bert_input"][qa_index]
            ["segments_token_type_id"],
            device=self.device
        )

        if not compute_loss:
            return model_input

        # (1,)
        model_input["gold_answer_labels"] = torch.tensor(
            preprocessed_data["gold_answer_labels"][qa_index:qa_index+1],
            device=self.device
        ).to(torch.long)

        return model_input

    def forward(
        self,
        segments_id,
        segments_mask,
        segments_token_type_id,
        compute_loss,
        gold_answer_labels=None
    ):
        """
        Parameters
        ----------
        segments_id : torch.Tensor
            shape of (n_segments, max_seg_len)
        segments_mask : torch.Tensor
            shape of (n_segments, max_seg_len)
        segments_token_type_id : torch.Tensor
            shape of (n_segments, max_seg_len)
        compute_loss : bool
        gold_answer_labels : torch.Tensor | None
            shape of (1,); by default None

        Returns
        -------
        ModelOutput
        """
        # Encode tokens by BERT
        # (n_segments, max_seg_len, hidden_dim)
        segments_vec = self.encode_tokens(
            segments_id=segments_id,
            segments_mask=segments_mask,
            segments_token_type_id=segments_token_type_id
        )

        # Compute segment vectors
        # (n_segments, 2 * hidden_dim)
        segment_vectors = self.compute_segment_vectors(
            segments_vec=segments_vec,
            segments_mask=segments_mask
        )

        # Compute a document vector
        # (1, 2 * hidden_dim)
        document_vector = self.compute_document_vector(
            segment_vectors=segment_vectors
        )

        # Compute logits by MLP
        # (1, n_answers)
        logits = self.compute_logits_by_mlp(document_vector=document_vector)

        if not compute_loss:
            return ModelOutput(
                logits=logits
            )

        # Compute loss
        # (1,)
        loss = self.loss_function(logits, gold_answer_labels)
        loss = loss.sum() # Scalar

        # Compute accuracy
        # (1,)
        pred_answer_labels = torch.argmax(logits, dim=1)
        # (1,)
        acc = (pred_answer_labels == gold_answer_labels).to(torch.float)
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
            shape of (n_segments, max_seg_len)
        segments_mask : torch.Tensor
            shape of (n_segments, max_seg_len)
        segments_token_type_id : torch.Tensor
            shape of (n_segments, max_seg_len)

        Returns
        -------
        torch.Tensor
            shape of (n_segments, max_seg_len, hidden_dim)
        """
        # Check
        # assert max_seg_len == self.max_seg_len

        # Encode segments by BERT
        bert_output = self.bert(
            input_ids=segments_id,
            attention_mask=segments_mask,
            token_type_ids=segments_token_type_id,
            output_attentions=False,
            output_hidden_states=False
        )
        # (n_segments, max_seg_len, hidden_dim)
        segments_vec = bert_output["last_hidden_state"]

        return segments_vec

    def compute_segment_vectors(self, segments_vec, segments_mask):
        """
        Parameters
        ----------
        segments_vec : torch.Tensor
            shape of (n_segments, max_seg_len, hidden_dim)
        segments_mask : torch.Tensor
            shape of (n_segments, max_seg_len)

        Returns
        -------
        torch.Tensor
            shape of (n_segments, 2 * hidden_dim)
        """
        # Get [CLS] embeddings
        # (n_segments, hidden_dim)
        cls_vectors = segments_vec[:, 0, :]

        # Get average embeddings
        # (n_segments, max_seg_len, hidden_dim)
        avg_vectors = segments_vec * segments_mask.unsqueeze(-1)
        # (n_segments, hidden_dim)
        avg_vectors = avg_vectors.sum(dim=1)
        # (n_segments, 1)
        n_tokens_for_each_seg = segments_mask.sum(dim=1).unsqueeze(-1)
        # (n_segments, hidden_dim)
        avg_vectors = avg_vectors / n_tokens_for_each_seg

        # (n_segments, 2 * hidden_dim)
        segment_vectors = torch.cat([cls_vectors, avg_vectors], dim=1)
        return segment_vectors

    def compute_document_vector(self, segment_vectors):
        """
        Parameters
        ----------
        segment_vectors : torch.Tensor
            shape of (n_segments, 2 * hidden_dim)

        Returns
        -------
        torch.Tensor
            shape of (1, 2 * hidden_dim)
        """
        # (n_segments, 2 * hidden_dim) -> (1, 2 * hidden_dim)
        document_vector = segment_vectors.logsumexp(dim=0).unsqueeze(0)
        return document_vector

    def compute_logits_by_mlp(self, document_vector):
        """
        Parameters
        ----------
        document_vector : torch.Tensor
            shape of (1, 2 * hidden_dim)

        Returns
        -------
        torch.Tensor
            shape of (1, n_answers)
        """
        logits = self.mlp(document_vector) # (1, n_answers)
        return logits


class MAQAPreprocessor:

    def __init__(
        self,
        tokenizer,
        max_seg_len,
        entity_dict,
        dataset_name,
        vocab_answer,
        possible_head_entity_types=None,
        possible_tail_entity_types=None,
        use_mention_as_canonical_name=False
    ):
        """
        Parameters
        ----------
        tokenizer : PreTrainedTokenizer
        max_seg_len : int
        entity_dict : dict[str, EntityPage]
        dataset_name : str
        vocab_answer: dict[str, int]
        possible_head_entity_types : list[str] | None
            by default None
        possible_tail_entity_types : list[str] | None
            by default None
        use_mention_as_canonical_name : bool
            by default False
        """
        self.tokenizer = tokenizer
        self.max_seg_len = max_seg_len
        self.entity_dict = entity_dict
        self.dataset_name = dataset_name
        self.vocab_answer = vocab_answer
        self.possible_head_entity_types = possible_head_entity_types
        self.possible_tail_entity_types = possible_tail_entity_types
        self.use_mention_as_canonical_name = use_mention_as_canonical_name

        self.qa_generator = QAGenerator(
            dataset_name=self.dataset_name,
            entity_dict=self.entity_dict,
            possible_head_entity_types=self.possible_head_entity_types,
            possible_tail_entity_types=self.possible_tail_entity_types,
            use_mention_as_canonical_name=self.use_mention_as_canonical_name
        )

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
        # entities: list[EntityTuple]
        # relations: list[TripleTuple]
        #####

        preprocessed_data["doc_key"] = document["doc_key"]

        sentences = [s.split() for s in document["sentences"]]
        preprocessed_data["sentences"] = sentences

        mentions = [
            MentionTuple(None, m["name"], m["entity_type"], m["entity_id"])
            for m in document["mentions"]
        ]
        preprocessed_data["mentions"] = mentions

        entities = [
            EntityTuple(e["mention_indices"], e["entity_type"], e["entity_id"])
            for e in document["entities"]
        ]
        preprocessed_data["entities"] = entities

        with_supervision = True if "relations" in document else False
        if with_supervision:
            relations = [
                TripleTuple(r["arg1"], r["relation"], r["arg2"])
                for r in document["relations"]
            ]
            preprocessed_data["relations"] = relations

        #####
        # qas: list[QATuple]
        #####

        qas = self.qa_generator.generate(document=document) # list[QATuple]
        preprocessed_data["qas"] = qas

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
        # qa_index_to_bert_input: list[dict[str, Any]]
        # segments: list[list[str]]
        # segments_id: list[list[int]]
        # segments_mask: list[list[int]]
        # segments_token_type_id: list[list[int]]
        #####

        qa_index_to_bert_input = [] # list[dict[str, Any]]
        for qa in qas:
            (
                segments,
                segments_id,
                segments_mask,
                segments_token_type_id
            ) = self.tokenize_and_split(
                sentences=sentences,
                question=qa.question
            )
            bert_input = {}
            bert_input["segments"] = segments
            bert_input["segments_id"] = segments_id
            bert_input["segments_mask"] = segments_mask
            bert_input["segments_token_type_id"] = segments_token_type_id
            qa_index_to_bert_input.append(bert_input)
        preprocessed_data["qa_index_to_bert_input"] = qa_index_to_bert_input

        #####
        # gold_answer_labels: list[int]
        #####

        if with_supervision:
            gold_answer_labels = [
                self.vocab_answer[qa.answer]
                for qa in qas
            ] # list[int]
            preprocessed_data["gold_answer_labels"] = gold_answer_labels

        return preprocessed_data

    #####
    # Subfunctions
    #####

    def tokenize_and_split(self, sentences, question):
        """
        Parameters
        ----------
        sentences: list[list[str]]
        question: list[str]

        Returns
        -------
        tuple[list[list[str]], list[list[int]], list[list[int]],
            list[list[int]]]
        """
        first_seq = " ".join(question)
        second_seq = " ".join(utils.flatten_lists(sentences))
        inputs = self.tokenizer(
            first_seq,
            second_seq,
            max_length=self.max_seg_len,
            padding="max_length",
            truncation="only_second",
            return_overflowing_tokens=True
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


class QAGenerator:

    def __init__(
        self,
        dataset_name,
        entity_dict,
        possible_head_entity_types=None,
        possible_tail_entity_types=None,
        use_mention_as_canonical_name=False
    ):
        self.dataset_name = dataset_name
        self.entity_dict = entity_dict
        self.possible_head_entity_types = possible_head_entity_types
        self.possible_tail_entity_types = possible_tail_entity_types
        self.use_mention_as_canonical_name = use_mention_as_canonical_name
        self.templates = TRIPLE_TO_QUESTION_TEMPLATES[self.dataset_name]

    def generate(self, document):
        """
        Parameters
        ----------
        document : Document

        Returns
        -------
        list[QATuple]
        """
        qas = [] # list[QATuple]
        false_negative_triples = []

        entities = document["entities"]
        if "relations" in document:
            with_supervision = True
            relations = document["relations"]
        else:
            with_supervision = False

        # Generate a canonical name list for entities for template-based generation
        # An entity type list will be used for type-based filtering (dataset specific)
        entity_names = [] # list[str]
        entity_types = [] # list[str]
        for e_i in range(len(entities)):
            entity_id = entities[e_i]["entity_id"] # str
            if self.use_mention_as_canonical_name:
                m_i = entities[e_i]["mention_indices"][0]
                canonical_name = document["mentions"][m_i]["name"]
            else:
                epage = self.entity_dict[entity_id] # dict
                canonical_name = epage["canonical_name"] # str
            entity_type = entities[e_i]["entity_type"] # str
            entity_names.append(canonical_name)
            entity_types.append(entity_type)

        # `not_include_entity_pairs` will be used for filtering
        not_include_entity_pairs = None
        if "not_include_pairs" in document:
            # List[(int, int)]
            epairs = [
                (epair["arg1"], epair["arg2"])
                for epair in document["not_include_pairs"]
            ]
            not_include_entity_pairs \
                = [(e1,e2) for e1,e2 in epairs] + [(e2,e1) for e1,e2 in epairs]

        # Create QAs
        for head_entity_i in range(len(entities)):
            for tail_entity_i in range(len(entities)):
                # Skip diagonal
                if head_entity_i == tail_entity_i:
                    continue

                # Skip based on entity types if specified
                # e.g, Skip chemical-chemical, disease-disease,
                #   and disease-chemical pairs for CDR.
                if (
                    (self.possible_head_entity_types is not None)
                    and
                    (self.possible_tail_entity_types is not None)
                ):
                    head_entity_type = entity_types[head_entity_i]
                    tail_entity_type = entity_types[tail_entity_i]
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

                if with_supervision:
                    gold_rels = self.find_relations(
                        arg1=head_entity_i,
                        arg2=tail_entity_i,
                        relations=relations
                    )
                # NOTE:
                # Please note that we generate questions only for relations written in `templates`.
                # This can result in lower recall scores,
                # since gold triples for relations not written in `templates` cannot be predicted (i.e., always false negative).
                for relation in self.templates.keys():
                    # Generate a question
                    question = self.templates[relation].replace(
                        HEAD_ENTITY_SLOT, entity_names[head_entity_i]
                    ).replace(
                        TAIL_ENTITY_SLOT, entity_names[tail_entity_i]
                    )
                    # Generate the answer
                    if with_supervision:
                        if relation in gold_rels:
                            answer = "Yes"
                        else:
                            answer = "No"
                    # QA instance
                    if with_supervision:
                        qa = QATuple(
                                TripleTuple(
                                    int(head_entity_i),
                                    relation,
                                    int(tail_entity_i)
                                ),
                                question.split(),
                                answer
                            )
                    else:
                        qa = QATuple(
                                TripleTuple(
                                    int(head_entity_i),
                                    relation,
                                    int(tail_entity_i)
                                ),
                                question.split(),
                                None
                            )
                    qas.append(qa)
                if with_supervision:
                    # `false_negative_rels` cannot be generated
                    false_negative_rels \
                        = set(gold_rels) - set(self.templates.keys())
                    # We record such false-negative triples
                    if len(false_negative_rels) > 0:
                        false_negative_triples.extend([
                            (
                                entities[head_entity_i]["entity_id"],
                                r,
                                entities[tail_entity_i]["entity_id"]
                            )
                            for r in false_negative_rels
                        ])

        assert len(qas) > 0
        if len(false_negative_rels) > 0:
            logger.warning("Questions for the following triple(s) are not generated, since corresponding templates cannot be found for the relations:")
            for x in false_negative_triples:
                logger.warning(f"{x}")
        return qas

    def find_relations(self, arg1, arg2, relations):
        """
        Parameters
        ----------
        arg1 : int
        arg2 : int
        relations : list[dict[str, int|str]]

        Returns
        -------
        List[str]
        """
        rels = []
        for triple in relations:
            if triple["arg1"] == arg1 and triple["arg2"] == arg2:
                rels.append(triple["relation"])
        return rels

