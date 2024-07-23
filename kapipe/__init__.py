# Types defined
from . import datatypes

# Top-level modules
from . import systems

# Submodules for `systems`
from . import models
from . import decoders
from . import prompt_processors
from . import misc

# Top modules for training and evaluation
from . import trainers

# Evaluation metrics
from . import evaluation

# Retrievers for few-shot in-context learning
from . import demonstration_retrievers

from . import utils


from .pipeline import Pipeline


__version__ = "0.0.3"


def load(identifier, gpu_map=None):
    ka = Pipeline(identifier=identifier, gpu_map=gpu_map)
    return ka


def blank(gpu_map=None):
    ka = Pipeline(identifier=None, gpu_map=gpu_map)
    return ka

