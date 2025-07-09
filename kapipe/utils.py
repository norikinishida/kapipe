from collections import OrderedDict
import datetime
from importlib.resources import files, as_file
import io
import json
import logging
import os
import time
from typing import Any, Callable

import numpy as np
import pyhocon
from pyhocon.converter import HOCONConverter
from pyhocon import ConfigTree

from .datatypes import Document, Mention, Entity, Passage


logger = logging.getLogger(__name__)


########
# IO utilities
########


def read_lines(path: str, encoding: str = "utf-8") -> list[str]:
    with open(path, encoding=encoding) as f:
        lines = [l.strip() for l in f]
    return lines


def read_json(path: str, encoding: str | None = None) -> dict[Any, Any]:
    if encoding is None:
        with open(path) as f:
            dct = json.load(f)
    else:
        with io.open(path, "rt", encoding=encoding) as f:
            line = f.read()
            dct = json.loads(line)
    return dct


def write_json(path: str, dct: dict[Any, Any], ensure_ascii: bool = True) -> None:
    with open(path, "w") as f:
        json.dump(dct, f, ensure_ascii=ensure_ascii, indent=4)


def read_vocab(path: str) -> dict[str, int]:
    # begin_time = time.time()
    # logger.info("Loading a vocabulary from %s" % path)
    vocab = OrderedDict()
    for line in open(path):
        items = line.strip().split("\t")
        if len(items) == 2:
            word, word_id = items
        elif len(items) == 3:
            word, word_id, freq = items
        else:
            raise Exception("Invalid line: %s" % items)
        vocab[word] = int(word_id)
    # end_time = time.time()
    # logger.info("Loaded. %f [sec.]" % (end_time - begin_time))
    # logger.info("Vocabulary size: %d" % len(vocab))
    return vocab


def write_vocab(
    path: str,
    data: list[tuple[str, int]] | list[str],
    write_frequency: bool = True
) -> None:
    with open(path, "w") as f:
        if write_frequency:
            for word_id, (word, freq) in enumerate(data):
                f.write("%s\t%d\t%d\n" % (word, word_id, freq))
        else:
            for word_id, word in enumerate(data):
                f.write("%s\t%d\n" % (word, word_id))


def get_hocon_config(config_path: str, config_name: str | None = None) -> ConfigTree:
    config = pyhocon.ConfigFactory.parse_file(config_path)
    if config_name is not None:
        config = config[config_name]
    config.config_path = config_path
    config.config_name = config_name
    # logger.info(pyhocon.HOCONConverter.convert(config, "hocon"))
    return config


def dump_hocon_config(path_out: str, config: ConfigTree) -> None:
    with open(path_out, "w") as f:
        f.write(HOCONConverter.to_hocon(config) + "\n")


def mkdir(path: str, newdir: str | None = None) -> None:
    if newdir is None:
        target = path
    else:
        target = os.path.join(path, newdir)
    if not os.path.exists(target):
        os.makedirs(target)
        logger.info("Created a new directory: %s" % target)


def print_list(
    lst: list[Any],
    with_index: bool = False,
    process: Callable[[Any], Any] | None = None
) -> None:
    for i, x in enumerate(lst):
        if process is not None:
            x = process(x)
        if with_index:
            logger.info(f"{i}: {x}")
        else:
            logger.info(x)


def safe_json_loads(
    generated_text: str,
    fallback: Any = None,
    list_type: bool = False
) -> Any:
    """
    Parse the report into a JSON object
    """
    if list_type:
        begin_index = generated_text.find("[")
        end_index = generated_text.rfind("]")
    else:
        begin_index = generated_text.find("{")
        end_index = generated_text.rfind("}")
    if begin_index < 0 or end_index < 0:
        logger.info(f"Failed to parse the generated text into a JSON object: '{generated_text}'")
        return fallback

    json_text = generated_text[begin_index: end_index + 1]

    try:
        json_obj = json.loads(json_text)
    except Exception as e:
        logger.info(f"Failed to parse the generated text into a JSON object: '{json_text}'")
        logger.info(e)
        return fallback

    if list_type:
        if not isinstance(json_obj, list):
            logger.info(f"The parsed JSON object is not a list: '{json_obj}'")
            return fallback
    else:
        if not isinstance(json_obj, dict):
            logger.info(f"The parsed JSON object is not a dictionary: '{json_obj}'")
            return fallback

    return json_obj

            
########
# Data utilities
########


def flatten_lists(list_of_lists: list[list[Any]]) -> list[Any]:
    return [elem for lst in list_of_lists for elem in lst]


def pretty_format_dict(dct: dict[Any, Any]) -> str:
    return "{}".format(json.dumps(dct, indent=4))


########
# Time utilities
########


def get_current_time() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


class StopWatch(object):

    def __init__(self):
        self.dictionary: dict[str | None, dict[str, float]] = {}

    def start(self, name: str | None = None):
        start_time = time.time()
        self.dictionary[name] = {}
        self.dictionary[name]["start"] = start_time

    def stop(self, name: str | None = None):
        stop_time = time.time()
        self.dictionary[name]["stop"] = stop_time

    def get_time(self, name: str | None = None, minute: bool = False) -> float:
        start_time = self.dictionary[name]["start"]
        stop_time = self.dictionary[name]["stop"]
        span = stop_time - start_time
        if minute:
            span /= 60.0
        return span


########
# Training utilities
########


class BestScoreHolder(object):

    def __init__(self, scale: float = 1.0, higher_is_better: bool = True):
        self.scale = scale
        self.higher_is_better = higher_is_better

        if higher_is_better:
            self.comparison_function = lambda best, cur: best < cur
        else:
            self.comparison_function = lambda best, cur: best > cur

        if higher_is_better:
            self.best_score = -np.inf
        else:
            self.best_score = np.inf
        self.best_step = 0
        self.patience = 0

    def init(self) -> None:
        if self.higher_is_better:
            self.best_score = -np.inf
        else:
            self.best_score = np.inf
        self.best_step = 0
        self.patience = 0

    def compare_scores(self, score: float, step: int) -> bool:
        if self.comparison_function(self.best_score, score):
            # Update the score
            logger.info("(best_score = %.02f, best_step = %d, patience = %d) -> (%.02f, %d, %d)" % \
                    (self.best_score * self.scale, self.best_step, self.patience,
                     score * self.scale, step, 0))
            self.best_score = score
            self.best_step = step
            self.patience = 0
            return True
        else:
            # Increment the patience
            logger.info("(best_score = %.02f, best_step = %d, patience = %d) -> (%.02f, %d, %d)" % \
                    (self.best_score * self.scale, self.best_step, self.patience,
                     self.best_score * self.scale, self.best_step, self.patience+1))
            self.patience += 1
            return False

    def ask_finishing(self, max_patience: int) -> bool:
        if self.patience >= max_patience:
            return True
        else:
            return False


########
# Task-specific utilities
########


def aggregate_mentions_to_entities(document: Document, mentions: list[Mention]):
    entity_id_to_info: dict[str, dict[str, Any]] = {}
    for m_i in range(len(document["mentions"])):
        name = document["mentions"][m_i]["name"]
        entity_type = document["mentions"][m_i]["entity_type"]
        entity_id = mentions[m_i]["entity_id"]
        if entity_id in entity_id_to_info:
            entity_id_to_info[entity_id]["mention_indices"].append(m_i)
            entity_id_to_info[entity_id]["mention_names"].append(name)
            # TODO
            # Confliction of entity types can appear, if EL model does not care about it.
            # assert (
            #     entity_id_to_info[entity_id]["entity_type"]
            #     == entity_type
            # )
        else:
            entity_id_to_info[entity_id] = {}
            entity_id_to_info[entity_id]["mention_indices"] = [m_i]
            entity_id_to_info[entity_id]["mention_names"] = [name]
            # TODO
            entity_id_to_info[entity_id]["entity_type"] = entity_type
    entities: list[Entity] = []
    for entity_id in entity_id_to_info.keys():
        mention_indices = entity_id_to_info[entity_id]["mention_indices"]
        mention_names = entity_id_to_info[entity_id]["mention_names"]
        entity_type = entity_id_to_info[entity_id]["entity_type"]
        entities.append({
            "mention_indices": mention_indices,
            "mention_names": mention_names,
            "entity_type": entity_type,
            "entity_id": entity_id,
        })
    return entities


def create_text_from_passage(passage: Passage, sep: str) -> str:
    if not "title" in passage:
        text = passage["text"]
    elif passage["text"].strip() == "":
        text = passage["title"]
    else:
        text = passage["title"] + sep + passage["text"]
    return text


def read_prompt_template(prompt_template_name_or_path: str) -> str:
    # List text files in "prompt_template" directory
    prompt_template_names = [
        x.name for x in files("kapipe.prompt_templates").iterdir()
        if x.name.endswith(".txt") and x.is_file() and not x.name.startswith("_")
    ]

    # Load the prompt template
    candidate_filename = prompt_template_name_or_path + ".txt"        
    if candidate_filename in prompt_template_names:
        template_path = files("kapipe.prompt_templates").joinpath(candidate_filename)
        with as_file(template_path) as path:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    else:
        assert os.path.isfile(prompt_template_name_or_path)
        with open(prompt_template_name_or_path, "r", encoding="utf-8") as f:
            return f.read()
 