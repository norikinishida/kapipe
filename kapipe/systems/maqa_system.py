import copy
import logging

# import numpy as np
import torch
from tqdm import tqdm

from ..models import MAQAModel
from .. import utils


logger = logging.getLogger(__name__)


class MAQASystem:
    """Mention-Agnostic QA-based DocRE system (Oumaima and Nishida et al., 2024)
    """

    def __init__(
        self,
        device,
        config,
        path_entity_dict,
        vocab_answer,
        path_model=None,
        verbose=True
    ):
        """
        Parameters
        ----------
        device: str
        config: ConfigTree | str
        path_entity_dict : str
        vocab_answer: dict[str, int] | str
        path_model: str | None
            by default None
        verbose: bool
            by default True
        """
        self.verbose = verbose
        if self.verbose:
            logger.info(">>>>>>>>>> MAQASystem Initialization >>>>>>>>>>")
        self.device = device

        ######
        # Config
        ######

        if isinstance(config, str):
            tmp = config
            config = utils.get_hocon_config(
                config_path=config,
                config_name=None
            )
            if self.verbose:
                logger.info(f"Loaded configuration from {tmp}")
        self.config = config
        if self.verbose:
            logger.info(utils.pretty_format_dict(self.config))

        ######
        # Vocabulary (answer types)
        ######

        if isinstance(vocab_answer, str):
            tmp = vocab_answer
            vocab_answer = utils.read_vocab(vocab_answer)
            if self.verbose:
                logger.info(f"Loaded answer type vocabulary from {tmp}")
        self.vocab_answer = vocab_answer
        self.ivocab_answer = {i:l for l, i in self.vocab_answer.items()}

        ######
        # Model
        ######

        self.model_name = config["model_name"]

        self.entity_dict = {
            epage["entity_id"]: epage
            for epage in utils.read_json(path_entity_dict)
        }
        if self.verbose:
            logger.info(f"Loaded entity dictionary from {path_entity_dict}")

        if self.model_name == "maqamodel":
            self.model = MAQAModel(
                device=device,
                bert_pretrained_name_or_path=config[
                    "bert_pretrained_name_or_path"
                ],
                max_seg_len=config["max_seg_len"],
                entity_dict=self.entity_dict,
                dataset_name=config["dataset_name"],
                dropout_rate=config["dropout_rate"],
                vocab_answer=self.vocab_answer,
                loss_function_name=config["loss_function"],
                focal_loss_gamma=config["focal_loss_gamma"] \
                    if config["loss_function"] == "focal_loss" else None,
                possible_head_entity_types=config["possible_head_entity_types"],
                possible_tail_entity_types=config["possible_tail_entity_types"],
                use_mention_as_canonical_name=config[
                    "use_mention_as_canonical_name"
                ]
            )
        else:
            raise Exception(f"Invalid model_name: {self.model_name}")

        # Show parameter shapes
        # logger.info("Model parameters:")
        # for name, param in self.model.named_parameters():
        #     logger.info(f"{name}: {tuple(param.shape)}")

        # Load trained model parameters
        if path_model is not None:
            self.load_model(path=path_model)
            if self.verbose:
                logger.info(f"Loaded model parameters from {path_model}")

        self.model.to(self.model.device)

        if self.verbose:
            logger.info("<<<<<<<<<< MAQASystem Initialization <<<<<<<<<<")

    def load_model(self, path):
        """
        Parameters
        ----------
        path : str
        """
        self.model.load_state_dict(
            torch.load(path, map_location=torch.device("cpu")),
            strict=False
        )

    def save_model(self, path):
        """
        Parameters
        ----------
        path : str
        """
        torch.save(self.model.state_dict(), path)

    # ---

    def compute_loss(self, document, qa_index):
        """
        Parameters
        ----------
        document : Document
        qa_index: int

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
        """
        # Switch to training mode
        self.model.train()

        # Preprocess
        preprocessed_data = self.model.preprocess(document=document)

        # Tensorize
        model_input = self.model.tensorize(
            preprocessed_data=preprocessed_data,
            qa_index=qa_index,
            compute_loss=True
        )

        # Forward
        model_output = self.model.forward(**model_input)

        return (
            model_output.loss,
            model_output.acc
        )

    def extract(self, document):
        """
        Parameters
        ----------
        document : Document

        Returns
        -------
        Document
        """
        with torch.no_grad():
            # Switch to inference mode
            self.model.eval()

            # Preprocess
            preprocessed_data = self.model.preprocess(document=document)

            # Get outputs by iterating over the questions
            triples = [] # list[Triple]
            qas = preprocessed_data["qas"] # list[QATuple]
            for qa_index in range(len(qas)):
                # Tensorize
                model_input = self.model.tensorize(
                    preprocessed_data=preprocessed_data,
                    qa_index=qa_index,
                    compute_loss=False
                )

                # Forward
                model_output = self.model.forward(**model_input)

                # (1, n_answers)
                logits = model_output.logits

                # Structurize
                pred_answer_label \
                    = torch.argmax(logits, dim=1).cpu().item() # int
                pred_answer = self.ivocab_answer[pred_answer_label] # str
                if pred_answer_label != 0:
                    head_entity_i, relation, tail_entity_i \
                        = qas[qa_index].triple
                    triples.append({
                        "arg1": int(head_entity_i),
                        "relation": relation,
                        "arg2": int(tail_entity_i),
                        "question": " ".join(qas[qa_index].question),
                        "answer": pred_answer
                    })
            # assert len(answers) == len(preprocessed_data["qas"])

            # Integrate
            result_document = copy.deepcopy(document)
            result_document["relations"] = triples
            return result_document

    def batch_extract(self, documents):
        """
        Parameters
        ----------
        documents : list[Document]

        Returns
        -------
        list[Document]
        """
        result_documents = []
        for document in tqdm(documents, desc="extraction steps"):
            result_document = self.extract(
                document=document
            )
            result_documents.append(result_document)
        return result_documents

