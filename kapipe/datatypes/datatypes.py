from typing import Any, TypeAlias

##########
# Data types for documents/passages
##########

Passage : TypeAlias = dict[str, Any]
# Required fields:
# - passage_key: str
# - text: str
# Optional fields:
# - title: str

Document : TypeAlias = dict[str, Any]
# Required fields:
# - doc_key: str
# - sentences: list[str]
# Optional fields:
# - mentions: list[Mention]
# - entities: list[Entity]
# - relations: list[Triple]

##########
# Data types for NER
##########

Mention : TypeAlias = dict[str, Any]
# Required fields:
# - span: tuple[int, int] | list[int] | None
# - name: str
# - entity_type: str
# Optional fields:
# - entity_id: str

##########
# Data types for Entity Disambiguation
##########

Entity : TypeAlias = dict[str, Any]
# Required fields:
# - mention_indices: list[int]
# - mention_names: list[str]
# - entity_type: str
# - entity_id: str

EntityPage : TypeAlias = dict[str, Any]
# Required fields:
# - entity_id: str
# - canonical_name: str
# - description: str
# Optional fields:
# - synonyms: list[str]
# - entity_type: str

CandEntKeyInfo : TypeAlias = dict[str, Any]
# Required fields:
# - entity_id: str
# - canonical_name: str
# - score: float

CandidateEntitiesForDocument : TypeAlias = dict[str, Any]
# Required fields:
# - doc_key: str
# - candidate_entities: list[list[CandEntKeyInfo]]

EntityPassage : TypeAlias = dict[str, Any]
# Required fields:
# - entity_id: str
# - title: str
# - text: str

##########
# Data types for Document-level Relation Extraction
##########

Triple : TypeAlias = dict[str, Any]
# Required fields:
# - arg1: int
# - relation: str
# - arg2: int

##########
# Data types for Community Clustering
##########

CommunityRecord : TypeAlias = dict[str, Any]
# Required fields:
# - community_key: str
# - nodes: list[str] | None
# - level: int
# - parent_community_key: str | None
# - child_community_keys: list[str]

##########
# Data types for QA
##########

ContextsForOneExample : TypeAlias = dict[str, Any]
# Required fields:
# - doc_key or question_key: str
# - contexts: list[Passage]

Question : TypeAlias = dict[str, Any]
# Required fields:
# - question_key: str
# - question: str
# Optional fields:
# - answers: list[Answer]
# - output_answer: str
# - rationale: str
# - helpfulness_score: float

Answer : TypeAlias = dict[str, Any]
# Required fields:
# - answer: str
# Optional fields:
# - answer_type: str
# - list_index: int
# list_index is required when answer_type is "list"
