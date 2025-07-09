from typing import Any, TypeAlias
from pyhocon import ConfigTree

##########
# Config
##########

Config : TypeAlias = ConfigTree 

##########
# Passage
##########

Passage : TypeAlias = dict[str, Any]
# {
#     "title": str # optional
#     "text": str,
# }

##########
# Document
##########

DocKey : TypeAlias = str

Document : TypeAlias = dict[str, Any]
# {
#     "doc_key": str,
#     "sentences": list[str], 
#     "mentions": list[Mention], # optional
#     "entities": list[Entity], # optional
#     "relations": list[Triple] # optional
# }

Mention : TypeAlias = dict[str, Any]
# {
#     "span": tuple[int],
#     "name": str,
#     "entity_type": str, # optional
#     "entity_id": str # optional
# }

Entity : TypeAlias = dict[str, Any]
# {
#     "mention_indices": list[int],
#     "mention_names": list[str],
#     "entity_type": str,
#     "entity_id": str
# }

Triple : TypeAlias = dict[str, Any]
# {
#     "arg1": int,
#     "relation": str,
#     "arg2": int
# }

##########
# Entity Dictionary
##########

EntityPage : TypeAlias = dict[str, Any]
# {
#     "entity_id": str,
#     "canonical_name": str,
#     "synonyms": list[str],
#     "description": str
# }

##########
# Candidate Entities
##########

CandidateEntitiesForDocument : TypeAlias = dict[str, Any]
# {
#     "doc_key": str,
#     "candidate_entities": list[list[CandEntKeyInfo]]
# }

CandEntKeyInfo : TypeAlias = dict[str, Any]
# {
#     "entity_id": str,
#     "score": float
# }

EntityPassage : TypeAlias = dict[str, Any]
# {
#     "id": str,
#     "title": str,
#     "text": str
# }

##########
# Community
##########

CommunityRecord : TypeAlias = dict[str, Any]
# {
#     "community_id": str,
#     "nodes": list[str],
#     "level": int, 
#     "parent_community_id": str,
#     "child_community_ids": list[str]
# }

##########
# Contexts
##########

ContextsForOneExample : TypeAlias = dict[str, Any]
# {
#     "question_key": str,
#     "contexts": list[Passage]
# }

##########
# Question
##########

QuestionKey : TypeAlias = str

Question : TypeAlias = dict[str, Any]
# {
#     "question_key": str,
#     "question": str,
#     "answers": list[Answer], # optional
# }

Answer : TypeAlias = dict[str, Any]
# {
#     "answer": str
# }

##########
# Demonstrations for In-Context Learning
##########

DemonstrationsForOneExample : TypeAlias = dict[str, Any]
# {
#     "doc_key" or "question_key": str
#     "demonstrations": list[DemoKeyInfo]
# }

DemoKeyInfo : TypeAlias = dict[str, Any]
# {
#     "doc_key" or "question_key": str,
#     "score": float
# }


