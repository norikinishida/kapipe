from collections import defaultdict
from collections import OrderedDict
import logging
import math
import queue
from typing import NamedTuple

import numpy as np
# import spacy_alignments as tokenizations
import torch
import torch.nn as nn
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers.modeling_outputs import ModelOutput
from opt_einsum import contract
import torch.multiprocessing as mp
from tqdm.autonotebook import trange

# from . import shared_functions
from .. import utils


logger = logging.getLogger(__name__)


class MentionTuple(NamedTuple):
    span: tuple[int, int]
    name: str
    entity_type: str
    entity_id: str | None


class EntityTuple(NamedTuple):
    mention_indices: list[int]
    entity_type: str
    entity_id: str


class BlinkBiEncoderModel(nn.Module):

    def __init__(
        self,
        device,
        bert_pretrained_name_or_path,
        max_seg_len,
        entity_seq_length
    ):
        """
        Parameters
        ----------
        device : str
        bert_pretrained_name_or_path : str
        max_seg_len : int
        entity_seq_length : int
        """
        super().__init__()

        ########################
        # Hyper parameters
        ########################

        self.device = device
        self.bert_pretrained_name_or_path = bert_pretrained_name_or_path
        self.max_seg_len = max_seg_len
        self.entity_seq_length = entity_seq_length

        ########################
        # Components
        ########################

        # BERT, tokenizer
        self.bert_m, self.tokenizer = self._initialize_bert_and_tokenizer(
            pretrained_model_name_or_path=self.bert_pretrained_name_or_path
        )
        self.bert_e, _ = self._initialize_bert_and_tokenizer(
            pretrained_model_name_or_path=self.bert_pretrained_name_or_path
        )

        # Dimensionality
        self.hidden_dim = self.bert_m.config.hidden_size

        self.linear = nn.Linear(2 * self.hidden_dim, self.hidden_dim)

        ######
        # Preprocessor
        ######

        self.preprocessor = BlinkBiEncoderPreprocessor(
            tokenizer=self.tokenizer,
            max_seg_len=self.max_seg_len,
            entity_seq_length=self.entity_seq_length
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
    # Forward pass (entity encoding)
    ################

    def preprocess_entities(self, candidate_entity_passages):
        """
        Parameters
        ----------
        candidate_entity_passages: list[EntityPassage]

        Returns
        -------
        dict[str, Any]
        """
        return self.preprocessor.preprocess_entities(
            candidate_entity_passages=candidate_entity_passages
        )
 
    def tensorize_entities(self, preprocessed_data, compute_loss, target_device=None):
        """
        Parameters
        ----------
        preprocessed_data : dict[str, Any]
        compute_loss : bool
        target_device : str | None
            by default None

        Returns
        -------
        dict[str, Any]
        """
        # XXX
        if target_device is None:
            device = self.device
        else:
            device = target_device

        model_input = {}

        model_input["compute_loss"] = compute_loss

        # (batch_size, entity_seq_length)
        model_input["segments_id"] = torch.tensor(
            preprocessed_data["bert_input"]["segments_id"],
            # device=self.device # XXX
            device=device # XXX
        )

        # (batch_size, entity_seq_length)
        model_input["segments_mask"] = torch.tensor(
            preprocessed_data["bert_input"]["segments_mask"],
            # device=self.device # XXX
            device=device # XXX
        )

        return model_input

    def encode_entities(
        self,
        segments_id,
        segments_mask,
        compute_loss
    ):
        """
        Parameters
        ----------
        segments_id : torch.Tensor
            shape of (batch_size, entity_seq_length)
        segments_mask : torch.Tensor
            shape of (batch_size, entity_seq_length)
        compute_loss : bool

        Returns
        -------
        torch.Tensor
            shape of (batch_size, hidden_dim)
        """
        # Encode tokens by BERT
        # (batch_size, entity_seq_length, hidden_dim)
        segments_vec = self.encode_tokens_for_entities(
            segments_id=segments_id,
            segments_mask=segments_mask
        )

        # Get [CLS] vectors
        # (batch_size, hidden_dim)
        entity_vectors = segments_vec[:, 0, :]

        return entity_vectors

    ################
    # Forward pass (mention encoding)
    ################

    def preprocess_mentions(self, document):
        """
        Parameters
        ----------
        document : Document
        candidate_entity_passages: list[EntityPassage] | None

        Returns
        -------
        dict[str, Any]
        """
        return self.preprocessor.preprocess_mentions(
            document=document,
        )

    def tensorize_mentions(self, preprocessed_data, compute_loss):
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

        # (n_mentions,)
        model_input["mention_begin_token_indices"] = torch.tensor(
            preprocessed_data["mention_begin_token_indices"],
            device=self.device
        )

        # (n_mentions,)
        model_input["mention_end_token_indices"] = torch.tensor(
            preprocessed_data["mention_end_token_indices"],
            device=self.device
        )

        return model_input

    def encode_mentions(
        self,
        segments_id,
        segments_mask,
        mention_begin_token_indices,
        mention_end_token_indices,
        compute_loss
    ):
        """
        Parameters
        ----------
        segments_id : torch.Tensor
            shape of (n_segments, max_seg_len)
        segments_mask : torch.Tensor
            shape of (n_segments, max_seg_len)
        mention_begin_token_indices : torch.Tensor
            shape of (n_mentions,)
        mention_end_token_indices : torch.Tensor
            shape of (n_mentions,)
        compute_loss : bool

        Returns
        -------
        torch.Tensor
            shape of (n_mentions, hidden_dim)
        """
        # Encode tokens by BERT
        # (n_tokens, hidden_dim)
        token_vectors = self.encode_tokens_for_mentions(
            segments_id=segments_id,
            segments_mask=segments_mask
        )

        # Compute mention vectors
        # (n_mentions, hidden_dim)
        mention_vectors = self.compute_mention_vectors(
            token_vectors=token_vectors,
            mention_begin_token_indices=mention_begin_token_indices,
            mention_end_token_indices=mention_end_token_indices
        )
        return mention_vectors

    ################
    # Forward pass (scoring)
    ################

    def preprocess_for_scoring(self, mentions, candidate_entity_passages):
        """
        Parameters
        ----------
        mentions: list[Mention]
        candidate_entity_passages: list[EntityPassage] | None

        Returns
        -------
        dict[str, Any]
        """
        return self.preprocessor.preprocess_for_scoring(
            mentions=mentions,
            candidate_entity_passages=candidate_entity_passages
        )

    def tensorize_for_scoring(self, preprocessed_data, compute_loss):
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

        # We assume that this and `forward` functions
        #   (for scoring forward pass) is called only in training,
        #   since we use Approximate Nearest Neighbor Search library
        #   instead of the manual inter-product computation and search.
        assert compute_loss == True

        # (n_mentions,)
        model_input["mention_gold_candidate_entity_indices"] = torch.tensor(
            preprocessed_data["mention_gold_candidate_entity_indices"],
            device=self.device
        ).to(torch.long)

        return model_input

    def forward_for_scoring(
        self,
        mention_vectors,
        candidate_entity_vectors,
        compute_loss,
        mention_gold_candidate_entity_indices=None
    ):
        """
        Parameters
        ----------
        mention_vectors : torch.Tensor
            shape of (n_mentions, hidden_dim)
        candidate_entity_vectors : torch.Tensor
            shape of (n_candidates, hidden_dim)
        compute_loss : bool
        mention_gold_candidate_entity_indices : torch.Tensor | None
            shape of (n_mentions,)

        Returns
        -------
        ModelOutput
        """
        # (n_mentions, n_candidates)
        logits = contract(
            "md,ed->me",
            mention_vectors.to(torch.float),
            candidate_entity_vectors
        )

        if not compute_loss:
            return ModelOutput(
                logits=logits
            )

        # Compute loss (summed over mentions)
        # (n_mentions,)
        loss = self.loss_function(
            logits,
            mention_gold_candidate_entity_indices
        )
        loss = loss.sum() # Scalar

        n_mentions = len(mention_gold_candidate_entity_indices)

        return ModelOutput(
            logits=logits,
            loss=loss,
            n_mentions=n_mentions
        )

    ################
    # Subfunctions
    ################

    def encode_tokens_for_entities(self, segments_id, segments_mask):
        """
        Parameters
        ----------
        segments_id : torch.Tensor
            shape of (batch_size, entity_seq_length)
        segments_mask : torch.Tensor
            shape of (batch_size, entity_seq_length)

        Returns
        -------
        torch.Tensor
            shape of (batch_size, entity_seq_length, hidden_dim)
        """
        bert_output = self.bert_e(
            input_ids=segments_id,
            attention_mask=segments_mask,
            output_attentions=False,
            output_hidden_states=False
        )
        # (batch_size, entity_seq_length, hidden_dim)
        segments_vec = bert_output["last_hidden_state"]
        return segments_vec

    def encode_tokens_for_mentions(self, segments_id, segments_mask):
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
        bert_output = self.bert_m(
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

    def compute_mention_vectors(
        self,
        token_vectors,
        mention_begin_token_indices,
        mention_end_token_indices
    ):
        """
        Parameters
        ----------
        token_vectors : torch.Tensor
            shape of (n_tokens, hidden_dim)
        mention_begin_token_indices : torch.Tensor
            shape of (n_mentions,)
        mention_end_token_indices : torch.Tensor
            shape of (n_mentions,)

        Returns
        -------
        torch.Tensor
            shape of (n_mentions, hidden_dim)
        """
        # (n_mentions, hidden_dim)
        mention_begin_token_vectors = token_vectors[mention_begin_token_indices]
        # (n_mentions, hidden_dim)
        mention_end_token_vectors = token_vectors[mention_end_token_indices]
        # (n_mentions, 2 * hidden_dim)
        mention_vectors = torch.cat(
            [mention_begin_token_vectors, mention_end_token_vectors],
            dim=1
        )
        # (n_mentions, hidden_dim)
        mention_vectors = self.linear(mention_vectors)
        return mention_vectors

    ################
    # For multi-processing
    ################

    def start_multi_process_pool(self):
        # Identify target devices
        # TODO: Allow GPU-IDs selection
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                # With multi-GPU mode, we skip GPU-0 to avoid OOM error.
                # Here, we assume that GPU-0 is set for the BLINK model.
                target_devices = [f"cuda:{i}" for i in range(1, torch.cuda.device_count())]
            else:
                target_devices = ["cuda:0"]
        else:
            logger.info("CUDA is not available. Starting 4 CPU workers")
            target_devices = ["cpu"] * 4
        logger.info("Start multi-process pool on devices: {}".format(", ".join(map(str, target_devices))))

        self.to("cpu")
        self.share_memory()
        ctx = mp.get_context("spawn")
        input_queue = ctx.Queue()
        output_queue = ctx.Queue()
        processes = []

        for device_id in target_devices:
            p = ctx.Process(
                target=BlinkBiEncoderModel._encode_multi_process_worker,
                args=(device_id, self, input_queue, output_queue),
                daemon=True
            )
            p.start()
            processes.append(p)

        return {
            "input": input_queue,
            "output": output_queue,
            "processes": processes
        }

    @staticmethod
    def stop_multi_process_pool(pool):
        for p in pool["processes"]:
            p.terminate()
            
        for p in pool["processes"]:
            p.join()
            p.close()

        pool["input"].close()
        pool["output"].close()

    def encode_multi_process(self, entity_passages, pool):
        CHUNK_SIZE = 256

        n_examples = len(entity_passages)
        n_processes = len(pool["processes"])

        chunk_size = min(math.ceil(n_examples / n_processes / 10), CHUNK_SIZE)

        logger.info(f"Chunk data into {math.ceil(n_examples / chunk_size)} packages of size {chunk_size}")

        input_queue = pool["input"]
        last_chunk_id = 0
        chunk = []

        for passage in entity_passages:
            chunk.append(passage)
            if len(chunk) >= chunk_size:
                input_queue.put(
                    [last_chunk_id, chunk]
                )
                last_chunk_id += 1
                chunk = []

        if len(chunk) > 0:
            input_queue.put(
                [last_chunk_id, chunk]
            )
            last_chunk_id += 1

        output_queue = pool["output"]
        results_list = sorted(
            [output_queue.get() for _ in trange(last_chunk_id, desc="Chunks", disable=False)],
            key=lambda x: x[0]
        )

        embeddings = np.concatenate([result[1] for result in results_list])
        # embeddings = torch.cat([result[1] for result in results_list], dim=0).numpy()
        return embeddings

    @staticmethod
    def _encode_multi_process_worker(target_device, model, input_queue, results_queue):
        while True:
            try:
                chunk_id, chunk = input_queue.get()
                # print(target_device, chunk_id, len(passages))

                with torch.no_grad():
                    model.eval()

                    model.to(target_device)

                    # Preprocess entities
                    preprocessed_data_e = model.preprocess_entities(
                        candidate_entity_passages=chunk
                    )
    
                    # Tensorize entities
                    model_input_e = model.tensorize_entities(
                        preprocessed_data=preprocessed_data_e,
                        compute_loss=False,
                        target_device=target_device
                    )
    
                    # Encode entities
                    # (CHUNK_SIZE, hidden_dim)
                    embeddings = model.encode_entities(**model_input_e)
                    embeddings = embeddings.cpu().numpy()
                    # print(target_device, chunk_id, len(passages), embeddings.shape)

                results_queue.put([chunk_id, embeddings])

            except queue.Empty:
                break
 

class BlinkBiEncoderPreprocessor:

    def __init__(
        self,
        tokenizer,
        max_seg_len,
        entity_seq_length
    ):
        """
        Parameters
        ----------
        tokenizer : PreTrainedTokenizer
        max_seg_len : int
        entity_seq_length : int
        """
        self.tokenizer = tokenizer
        self.max_seg_len = max_seg_len
        self.entity_seq_length = entity_seq_length

        self.cls_token = tokenizer.cls_token
        self.sep_token = tokenizer.sep_token

    def preprocess_entities(self, candidate_entity_passages):
        """
        Parameters
        ----------
        candidate_entity_passages : list[EntityPassage]

        Returns
        -------
        dict[str, Any]
        """
        preprocessed_data = OrderedDict()

        #####
        # texts: list[list[str]]
        #   (n_candidates,)
        #####

        # (n_candidates,)
        texts = [
            (p["title"] + " : " + p["text"]).split()
            for p in candidate_entity_passages
        ]
        preprocessed_data["texts"] = texts

        #####
        # bert_input: dict[str, Any]
        # segments: list[list[str]]
        # segments_id: list[list[int]]
        # segments_mask: list[list[int]]
        #####
        inputs = self.tokenizer(
            [" ".join(t) for t in texts],
            # max_length=self.max_seg_len,
            max_length=self.entity_seq_length,
            padding=True,
            truncation=True,
            return_overflowing_tokens=False
        )
        bert_input = {}
        bert_input["segments_id"] = inputs["input_ids"]
        bert_input["segments"] = [
            self.tokenizer.convert_ids_to_tokens(seg)
            for seg in inputs["input_ids"]
        ]
        bert_input["segments_mask"] = inputs["attention_mask"]
        preprocessed_data["bert_input"] = bert_input

        return preprocessed_data

    def preprocess_mentions(self, document):
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
        # mention_gold_entity_ids: list[str]
        #   (n_mentions,)
        # mention_gold_candidate_entity_indices: list[int]
        #   (n_mention,)
        #####

        # if with_supervision:
        #     # Candidate entities should be determined beforehand for each document
        #     # list[str]; (n_candidates,)
        #     candidate_entity_ids = [
        #         p["id"] for p in candidate_entity_passages
        #     ]
        #     preprocessed_data["candidate_entity_ids"] = candidate_entity_ids

        #     # list[str]; (n_mentions,)
        #     mention_gold_entity_ids = [
        #         mention.entity_id for mention in mentions
        #     ]
        #     preprocessed_data["mention_gold_entity_ids"] = mention_gold_entity_ids

        #     # (n_mentions, n_candidates)
        #     matched = (
        #         np.asarray(mention_gold_entity_ids)[:,None]
        #         == np.asarray(candidate_entity_ids)
        #     )
        #     n_mentions = len(mention_gold_entity_ids)
        #     assert matched.sum() == n_mentions
        #     # list[int]; (n_mentions,)
        #     mention_gold_candidate_entity_indices = np.where(matched)[1].tolist()
        #     preprocessed_data["mention_gold_candidate_entity_indices"] \
        #         = mention_gold_candidate_entity_indices

        return preprocessed_data

    def preprocess_for_scoring(self, mentions, candidate_entity_passages):
        preprocessed_data = OrderedDict()

        #####
        # mention_gold_entity_ids: list[str]
        #   (n_mentions,)
        # mention_gold_candidate_entity_indices: list[int]
        #   (n_mention,)
        #####

        # Candidate entities should be determined beforehand for each document
        # list[str]; (n_candidates,)
        candidate_entity_ids = [
            p["id"] for p in candidate_entity_passages
        ]
        preprocessed_data["candidate_entity_ids"] = candidate_entity_ids

        # list[str]; (n_mentions,)
        mention_gold_entity_ids = [
            # mention.entity_id for mention in mentions
            mention["entity_id"] for mention in mentions
        ]
        preprocessed_data["mention_gold_entity_ids"] = mention_gold_entity_ids

        # (n_mentions, n_candidates)
        matched = (
            np.asarray(mention_gold_entity_ids)[:,None]
            == np.asarray(candidate_entity_ids)
        )
        n_mentions = len(mention_gold_entity_ids)
        assert matched.sum() == n_mentions
        # list[int]; (n_mentions,)
        mention_gold_candidate_entity_indices = np.where(matched)[1].tolist()
        preprocessed_data["mention_gold_candidate_entity_indices"] \
            = mention_gold_candidate_entity_indices

        return preprocessed_data

    #####
    # Subfunctions
    #####

    def tokenize_and_split(self, sentences, mentions):
        """
        Parameters
        ----------
        sentences: list[list[str]]
        mentions: list[MentionTuple]

        Returns
        -------
        tuple[list[list[str]], list[list[int]], list[list[int]], list[int],
            list[list[int]], list[int], list[int], list[int]]
        """
        # subtoken分割、subtoken単位でのtoken終点、文終点、メンション始点、
        #   メンション終点のindicatorsの作成
        (
            sents_subtoken,
            sents_token_end,
            sents_sentence_end,
            subtoken_index_to_word_index,
            sents_mention_begin,
            sents_mention_end
        ) = self.tokenize(sentences=sentences, mentions=mentions)

        # 1階層のリストに変換
        doc_subtoken = utils.flatten_lists(sents_subtoken) # list[str]
        doc_token_end = utils.flatten_lists(sents_token_end) # list[bool]
        doc_sentence_end = utils.flatten_lists(sents_sentence_end) # list[bool]
        doc_mention_begin = utils.flatten_lists(sents_mention_begin)
        doc_mention_end = utils.flatten_lists(sents_mention_end) # list[bool]

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

    def tokenize(self, sentences, mentions):
        """
        Parameters
        ----------
        sentences: list[list[str]]
        mentions: list[MentionTuple]

        Returns
        -------
        tuple[list[list[str]], list[list[bool]], list[list[bool]], list[int],
            list[list[bool]], list[list[bool]]]
        """
        sents_subtoken = [] # list[list[str]]
        sents_token_end = [] # list[list[bool]]
        sents_sentence_end = [] # list[list[bool]]
        subtoken_index_to_word_index = [] # list[int]

        sents_mention_begin = [] # list[list[bool]]
        sents_mention_end = [] # list[list[bool]]

        original_mention_begin_token_indices = np.asarray(
            [m.span[0] for m in mentions] + [-1]
        )
        original_mention_end_token_indices = np.asarray(
            [m.span[1] for m in mentions] + [-1]
        )

        word_idx = -1
        offset = 0
        for sent in sentences:
            sent_subtoken = [] # list[str]
            sent_token_end = [] # list[bool]
            sent_sentence_end = [] # list[bool]
            sent_mention_begin = [] # list[bool]
            sent_mention_end = [] # list[bool]
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
                # メンションの開始位置、終了位置
                begin_count = len(np.where(
                    original_mention_begin_token_indices == (offset + token_i)
                )[0])
                end_count = len(np.where(
                    original_mention_end_token_indices == (offset + token_i)
                )[0])
                sent_mention_begin += [begin_count] + [0] * (len(subtokens) - 1)
                sent_mention_end += [end_count] + [0] * (len(subtokens) - 1)
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
        doc_subtoken,
        doc_sentence_end,
        doc_token_end,
        subtoken_index_to_word_index,
        doc_mention_begin,
        doc_mention_end
    ):
        """
        Parameters
        ----------
        doc_subtoken: list[str]
        doc_sentence_end: list[bool]
        doc_token_end: list[bool]
        subtoken_index_to_word_index: list[int]
        doc_mention_begin: list[bool]
        doc_mention_end: list[bool]

        Returns
        -------
        tuple[list[list[str]], list[int], list[list[int]], list[int],
            list[int], list[int]]
        """
        segments = [] # list[list[str]]
        segments_subtoken_map = [] # list[list[int]]
        segments_mention_begin = [] # list[list[bool]]
        segments_mention_end = [] # list[list[bool]]

        n_subtokens = len(doc_subtoken)
        curr_idx = 0 # Index for subtokens
        while curr_idx < len(doc_subtoken):
            # Try to split at a sentence end point
            end_idx = min(curr_idx + self.max_seg_len - 1 - 2, n_subtokens - 1)
            while end_idx >= curr_idx and not doc_sentence_end[end_idx]:
                end_idx -= 1
            if end_idx < curr_idx:
                logger.warning("No sentence end found; split at token end")
                # If no sentence end point found, try to split at token end point
                end_idx = min(
                    curr_idx + self.max_seg_len - 1 - 2,
                    n_subtokens - 1
                )
                seg_before = \
                    "\"" + " ".join(doc_subtoken[curr_idx:end_idx+1]) + "\""
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
            segment_subtoken_map = \
                subtoken_index_to_word_index[curr_idx: end_idx + 1]
            segment_mention_begin = doc_mention_begin[curr_idx: end_idx + 1]
            segment_mention_end = doc_mention_end[curr_idx: end_idx + 1]
            segment = [self.cls_token] + segment + [self.sep_token]
            # NOTE: [CLS] is treated as the first subtoken
            #       of the first word for each segment
            # NOTE: [SEP] is treated as the last subtoken
            #       of the last word for each segment
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
                [subtok_i] * cnt
                for subtok_i, cnt in enumerate(
                    utils.flatten_lists(segments_mention_begin)
                )
            ]
        )
        mention_end_token_indices = utils.flatten_lists(
            [
                [subtok_i] * cnt
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

    def convert_to_token_ids_with_padding(self, segments):
        """
        Parameters
        ----------
        segments: list[list[str]]

        Returns
        -------
        tuple[list[list[int]], list[list[int]]]
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

