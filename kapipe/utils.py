from collections import OrderedDict
import datetime
from importlib.resources import files, as_file
import io
import json
import logging
import os
import time

import numpy as np
import pyhocon
from pyhocon.converter import HOCONConverter


logger = logging.getLogger(__name__)


########
# IO utilities
########


def read_lines(path, encoding="utf-8"):
    with open(path, encoding=encoding) as f:
        lines = [l.strip() for l in f]
    return lines


def read_json(path, encoding=None):
    """
    Parameters
    ----------
    path: str
    encoding: str or None, default None

    Returns
    -------
    dict[Any, Any]
    """
    if encoding is None:
        with open(path) as f:
            dct = json.load(f)
    else:
        with io.open(path, "rt", encoding=encoding) as f:
            line = f.read()
            dct = json.loads(line)
    return dct


def write_json(path, dct, ensure_ascii=True):
    """
    Parameters
    ----------
    path: str
    dct: dict[Any, Any]
    ensure_ascii: bool
        by default True
    """
    with open(path, "w") as f:
        json.dump(dct, f, ensure_ascii=ensure_ascii, indent=4)


def read_vocab(path):
    """
    Parameters
    ----------
    path: str

    Returns
    -------
    dict[str, int]
    """
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


def write_vocab(path, data, write_frequency=True):
    """
    Parameters
    ----------
    path: str
    data: list[(str, int)] or list[str]
    write_frequency: bool, default True
    """
    with open(path, "w") as f:
        if write_frequency:
            for word_id, (word, freq) in enumerate(data):
                f.write("%s\t%d\t%d\n" % (word, word_id, freq))
        else:
            for word_id, word in enumerate(data):
                f.write("%s\t%d\n" % (word, word_id))


def get_hocon_config(config_path, config_name=None):
    """
    Generate a configuration dictionary.

    Parameters
    ----------
    config_path : str
    config_name : str, default None

    Returns
    -------
    ConfigTree
    """
    config = pyhocon.ConfigFactory.parse_file(config_path)
    if config_name is not None:
        config = config[config_name]
    config.config_path = config_path
    config.config_name = config_name
    # logger.info(pyhocon.HOCONConverter.convert(config, "hocon"))
    return config


def dump_hocon_config(path_out, config):
    with open(path_out, "w") as f:
        f.write(HOCONConverter.to_hocon(config) + "\n")


def mkdir(path, newdir=None):
    """
    Parameters
    ----------
    path: str
    newdir: str or None, default None
    """
    if newdir is None:
        target = path
    else:
        target = os.path.join(path, newdir)
    if not os.path.exists(target):
        os.makedirs(target)
        logger.info("Created a new directory: %s" % target)


def print_list(lst, with_index=False, process=None):
    """
    Parameters
    ----------
    lst: list[Any]
    with_index: bool, default False
    process: function: Any -> Any
    """
    for i, x in enumerate(lst):
        if process is not None:
            x = process(x)
        if with_index:
            logger.info(f"{i}: {x}")
        else:
            logger.info(x)


def safe_json_loads(generated_text, fallback=None, list_type=False):
    """
    Parse the report into a JSON object
    """
    # try:
    #     return json.loads(generated_text)
    # except json.JSONDecodeError as e:
    #     cleaned = (
    #         generated_text.strip()
    #         .removeprefix("```json").removesuffix("```")
    #         .strip("` \n")
    #     )
    #     try:
    #         return json.loads(cleaned)
    #     except json.JSONDecodeError as e2:
    #         print("[JSONDecodeError]", e)
    #         print("[Raw Output]", generated_text[:300])
    #         return fallback

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


def flatten_lists(list_of_lists):
    """
    Parameters
    ----------
    list_of_lists: list[list[Any]]

    Returns
    -------
    list[Any]
    """
    return [elem for lst in list_of_lists for elem in lst]


def pretty_format_dict(dct):
    """
    Parameters
    ----------
    dct: dict[Any, Any]

    Returns
    -------
    str
    """
    return "{}".format(json.dumps(dct, indent=4))


########
# Time utilities
########


def get_current_time():
    """
    Returns
    -------
    str
    """
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


class StopWatch(object):

    def __init__(self):
        self.dictionary = {}

    def start(self, name=None):
        """
        Parameters
        ----------
        name: str or None, default None
        """
        start_time = time.time()
        self.dictionary[name] = {}
        self.dictionary[name]["start"] = start_time

    def stop(self, name=None):
        """
        Parameters
        ----------
        name: str or None, default None
        """
        stop_time = time.time()
        self.dictionary[name]["stop"] = stop_time

    def get_time(self, name=None, minute=False):
        """
        Parameters
        ----------
        name: str or None, default None
        minute: bool, default False

        Returns
        -------
        float
        """
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

    def __init__(self, scale=1.0, higher_is_better=True):
        """
        Parameters
        ----------
        scale: float, default 1.0
        higher_is_better: bool, default True
        """
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

    def init(self):
        if self.higher_is_better:
            self.best_score = -np.inf
        else:
            self.best_score = np.inf
        self.best_step = 0
        self.patience = 0

    def compare_scores(self, score, step):
        """
        Parameters
        ----------
        score: float
        step: int

        Returns
        -------
        bool
        """
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

    def ask_finishing(self, max_patience):
        """
        Parameters
        ----------
        max_patience: int

        Returns
        -------
        bool
        """
        if self.patience >= max_patience:
            return True
        else:
            return False


########
# Task-specific utilities
########


def aggregate_mentions_to_entities(document, mentions):
    entity_id_to_info = {} # dict[str, dict[str, Any]]
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
    entities = [] # list[Entity]
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


def create_text_from_passage(passage, sep):
    # if not "title" in passage:
    #     text = passage["text"]
    # else:
    #     text = passage["title"] + sep + passage["text"]
    if not "title" in passage:
        text = passage["text"]
    elif passage["text"].strip() == "":
        text = passage["title"]
    else:
        text = passage["title"] + sep + passage["text"]
    return text


def read_prompt_template(prompt_template_name_or_path):
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
 