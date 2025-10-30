__version__ = "0.1.1"

__all__ = []

#####
# Core
#####

# from . import datatypes
# from . import utils

__all__ += ["datatypes", "utils"]

#####
# Components for Knowledge Extraction
#####

# from . import ner
# from . import ed_retrieval
# from . import ed_reranking
# from . import docre

__all__ += ["ner", "ed_retrieval", "ed_reranking", "docre"]

#####
# Components for Knowledge Organization
#####

# from . import knowledge_graph_construction
# from . import community_clustering
# from . import report_generation
# from . import chunking

__all__ += ["knowledge_graph_construction", "community_clustering", "report_generation", "chunking"]

#####
# Components for Knowledge Retrieval
#####

# from . import passage_retrieval

__all__ += ["passage_retrieval"]

#####
# Components for Knowledge Utilization
#####

# from . import qa

__all__ += ["qa"]

#####
# Components for LLM
#####

# from . import llms
# from . import prompt_templates
# from . import demonstration_retrieval

__all__ += ["llms", "prompt_templates", "demonstration_retrieval"]

#####
# Others
#####

# from . import nn_utils
# from . import evaluation

__all__ += ["nn_utils", "evaluation"]

#####
# Pipelines for Knowledge Acquisition
#####

# from . import pipelines

__all__ += ["pipelines"]

#####
# Lazy import
#####

import importlib

def __getattr__(name):
    if name in __all__:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__} has no attribute {name}")
