from typing import Any, TypeAlias


DocKey : TypeAlias = str

Document : TypeAlias = dict[str, Any]
# {
#     "doc_key": str,
#     "sentences": list[str],
#     "mentions": list[Mention],
#     "entities": list[Entity],
#     "relations": list[Triple]
# }

Mention : TypeAlias = dict[str, Any]
# {
#     "span": tuple[int],
#     "name": str,
#     "entity_type": str
# }

Entity : TypeAlias = dict[str, Any]
# {
#     "mention_indices": list[int],
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
# For candidate entities in ED
##########

EntityPage : TypeAlias = dict[str, Any]
# {
#     "entity_id": str,
#     "canonical_name": str,
#     "synonyms": list[str],
#     "description": str
# }

CandEntKeyInfo : TypeAlias = dict[str, Any]
# {
#     "entity_id": str,
#     "score": float
# }

EntDoc: TypeAlias = dict[str, str]
# {
#     "entity_id": str,
#     "canonical_name": str,
#     "text": str
# }

##########
# For in-context learning
##########

DemoKeyInfo : TypeAlias = dict[str, Any]
# {
#     "doc_key": str,
#     "score": float
# }
