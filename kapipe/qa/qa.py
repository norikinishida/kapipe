from __future__ import annotations

import logging
import os
from os.path import expanduser

from .llmqa import LLMQA
from .. import utils
from ..datatypes import (
    Config,
    Question,
    ContextsForOneExample
)


logger = logging.getLogger(__name__)


class QA:
    
    def __init__(self, identifier: str, gpu: int = 0):
        self.identifier = identifier
        self.gpu = gpu
        
        root_config: Config = utils.get_hocon_config(
            os.path.join(expanduser("~"), ".kapipe", "download", "config")
        )
        self.module_config: Config = root_config["qa"][identifier]

        # Initialize the QA module
        if self.module_config["method"] == "llmqa":
            self.answerer = LLMQA(
                device=f"cuda:{self.gpu}",
                path_snapshot=self.module_config["snapshot"]
            )
        else:
            raise Exception(f"Invalid method: {self.module_config['method']}")

    def answer(
        self,
        question: Question,
        contexts_for_question: ContextsForOneExample | None
    ) -> Question: 
        return self.answerer.answer(
            question=question,
            demonstrations_for_question=None,
            contexts_for_question=contexts_for_question
        )
 
