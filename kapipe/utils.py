from collections import OrderedDict
import datetime
import io
import json
import logging
import os
import time

import numpy as np
import pyhocon
from pyhocon.converter import HOCONConverter


logger = logging.getLogger(__name__)



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


def write_json(path, dct):
    """
    Parameters
    ----------
    path: str
    dct: dict[Any, Any]
    """
    with open(path, "w") as f:
        json.dump(dct, f, indent=4)


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


def get_current_time():
    """
    Returns
    -------
    str
    """
    return datetime.datetime.now().strftime("%b%d_%H-%M-%S")


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


