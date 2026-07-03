import argparse
import os

from tqdm import tqdm

from kapipe import utils


KEY_MAP = {
    "Class": {
        "desc": "@DescriptorClass",
        "supp": "@SCRClass",
    },
    "RecordSet": {
        "desc": "DescriptorRecordSet",
        "supp": "SupplementalRecordSet",
    },
    "Record": {
        "desc": "DescriptorRecord",
        "supp": "SupplementalRecord",
    },
    "UI": {
        "desc": "DescriptorUI",
        "supp": "SupplementalRecordUI",
    },
    "Name": {
        "desc": "DescriptorName",
        "supp": "SupplementalRecordName",
    }
}


ENTITY_TYPE_MAP = {
    "A": "Anatomy",
    "B": "Organism",
    "C": "Disease",
    "D": "Chemical",
    "E": "TechniqueAndEquipment",
    "F": "PhychiatryAndPsychology",
    "G": "PhenomemonAndProcess",
    "H": "Discripline",
    "I": "Social",
    "J": "Industry",
    "K": "Humanity",
    "L": "InformationScience",
    "M": "NamedGroup",
    "N": "HealthCare",
    "V": "Publication",
    "Z": "Geographical",
}


def main(args):
    """
    References:
        - https://www.nlm.nih.gov/databases/download/mesh.html
        - https://www.nlm.nih.gov/mesh/xml_data_elements.html
        - https://www.nlm.nih.gov/mesh/xmlconvert_ascii.html
    """
    path_input_dir = args.input_dir
    path_output_file = args.output_file
    utils.mkdir(os.path.dirname(path_output_file))

    entities = process(path_input_file=os.path.join(path_input_dir, "desc2015.json"), mode="desc", entities=None)
    entities = process(path_input_file=os.path.join(path_input_dir, "supp2015.json"), mode="supp", entities=entities)

    entities = list(entities.values())

    utils.write_json(path_output_file, entities)
    print(f"Saved {len(entities)} entities to {path_output_file}")


def process(path_input_file, mode, entities=None):
    if entities is None:
        entities = {}

    records = utils.read_json(path_input_file)

    assert len(records.keys()) == 1
    records = records[KEY_MAP["RecordSet"][mode]]
    assert records["@LanguageCode"] == "eng"
    records = records[KEY_MAP["Record"][mode]] # List[Dict]
    print(f"Found {len(records)} records")

    for desc_record in tqdm(records):
        # Entity ID
        entity_id = desc_record[KEY_MAP["UI"][mode]]

        assert not entity_id in entities

        # Canonical name
        canonical_name = desc_record[KEY_MAP["Name"][mode]]["String"]

        # Description
        if "Annotation" in desc_record:
            description = desc_record["Annotation"]
        else:
            description = ""

        # Synonyms
        # NOTE: We consider only the "preferred" concept and corresponding terms
        synonyms = []
        for concept_record in transform_to_list(desc_record["ConceptList"]["Concept"]):
            if concept_record["@PreferredConceptYN"] == "Y":
                assert concept_record["ConceptName"]["String"] == canonical_name
                # If there is  a `ScopeNote` entry, we append the ScopeNote to the description (`Annotation`)
                if "ScopeNote" in concept_record:
                    if description == "":
                        description = concept_record["ScopeNote"]
                    else:
                        description = description + " " + concept_record["ScopeNote"]
                for term_record in transform_to_list(concept_record["TermList"]["Term"]):
                    if term_record["@ConceptPreferredTermYN"] == "Y":
                        assert term_record["String"] == canonical_name
                    else:
                        synonyms.append(term_record["String"])

        # Tree Numbers
        # if desc_record[KEY_MAP["Class"][mode]] != "3":
        if "TreeNumberList" in desc_record:
            tree_numbers = transform_to_list(desc_record["TreeNumberList"]["TreeNumber"]) # List[str]
        else:
            tree_numbers = []

        # Entity Type
        # MeSH descriptors are organized into 16 categories.
        if len(tree_numbers) > 0:
            first_char = tree_numbers[0][0]
            entity_type = ENTITY_TYPE_MAP[first_char]
        else:
            entity_type = None

        entity = {
            "entity_id": entity_id,
            "entity_index": len(entities),
            "entity_type": entity_type,
            "canonical_name": canonical_name,
            "synonyms": synonyms,
            "description": description,
            "tree_numbers": tree_numbers,
        }
        entities[entity_id] = entity

    return entities


def transform_to_list(xs):
    if not isinstance(xs, list):
        return [xs]
    else:
        return xs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    main(args=args)

