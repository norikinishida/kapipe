import argparse
import json
import os

from tqdm import tqdm

import sys
sys.path.insert(0, "../../..")
from kapipe import utils



def main(args):
    path_input_file = args.input_file
    path_output_file = args.output_file
    target_answer_types = args.target_answer_types
    path_pubmed_abstracts = args.pubmed_abstracts
    size = args.size

    if target_answer_types is None:
        target_answer_types = ["yesno", "factoid", "list", "summary", "long"]

    utils.mkdir(os.path.dirname(path_output_file))

    dataset = utils.read_json(path_input_file)
    print(f"Loaded {len(dataset['questions'])} questions from {path_input_file}")

    if path_pubmed_abstracts is not None:
        print(f"Loading PubMed abstracts (IDs) from {path_pubmed_abstracts}")
        possible_pubmed_ids = []
        with open(path_pubmed_abstracts) as f:
            for line in f:
                json_obj = json.loads(line.strip())
                possible_pubmed_ids.append(json_obj["pmid"])
        possible_pubmed_ids = set(possible_pubmed_ids)
        print(f"Loaded {len(possible_pubmed_ids)} PubMed abstracts (IDs) from {path_pubmed_abstracts}")

    questions = []
    contexts = []
    for data in tqdm(dataset["questions"]):
        # Extract example ID
        question_key = str(data["id"])

        # Extract gold context passages from `snippets`
        # We ignore `documents`, `concepts` (entity concepts), and `triples`` attributes
        contexts_for_question = []
        for snippet_dict in data["snippets"]:
            text = snippet_dict["text"]
            url = snippet_dict["document"]
            begin_section = snippet_dict["beginSection"]
            end_section = snippet_dict["endSection"]
            offset_in_begin_section = snippet_dict["offsetInBeginSection"]
            offset_in_end_section = snippet_dict["offsetInEndSection"]
            context = {
                "text": text,
                "url": url,
                "begin_section": begin_section,
                "end_section": end_section,
                "offset_in_begin_section": offset_in_begin_section,
                "offset_in_end_section": offset_in_end_section
            }
            contexts_for_question.append(context)
        contexts_for_question = sorted(contexts_for_question, key=lambda c: (c["url"], c["begin_section"], c["offset_in_begin_section"], c["end_section"], c["offset_in_end_section"]))

        # If PubMed abstracts are specified as a command-line argument,
        # we check whether all the context passages are involved in the specified PubMed abstracts.
        # Then, if all of the context passages are included in the specified PubMed abstracts, proceed;
        # otherwise, skip the question instance.
        if path_pubmed_abstracts is not None:
            pubmed_ids = set([extract_pmid_from_url(c["url"]) for c in contexts_for_question])
            diff = pubmed_ids - possible_pubmed_ids
            if len(diff) != 0:
                diff = list(diff)
                print(f"Skip an example because some context passages ({diff}) are not included in the provided PubMed abstracts")
                continue

        contexts_for_question = {
            "question_key": question_key,
            "contexts": contexts_for_question
        }

        # Extract question
        question = data["body"]
        
        # Extract answers
        answers = []
        answer_type = data["type"]
        if answer_type == "yesno" and answer_type in target_answer_types:
            assert isinstance(data["exact_answer"], str)
            answer_text = data["exact_answer"]
            answers.append({
                "answer": answer_text,
                "answer_type": answer_type
            })
        elif answer_type == "factoid" and answer_type in target_answer_types:
            assert isinstance(data["exact_answer"], list)
            for answer_text in data["exact_answer"]:
                answers.append({
                    "answer": answer_text,
                    "answer_type": answer_type
                })
        elif answer_type == "list" and answer_type in target_answer_types:
            # From exact_answer
            assert isinstance(data["exact_answer"], list)
            # for answer_list in data["exact_answer"]:
            #     answers.append({
            #         "answer": answer_list,
            #         "answer_type": answer_type
            #     })
            for list_i, answer_list in enumerate(data["exact_answer"]):
                for answer_text in answer_list:
                    answers.append({
                        "answer": answer_text,
                        "answer_type": answer_type,
                        "list_index": list_i,
                    })
        elif answer_type == "summary" and answer_type in target_answer_types:
            # NOTE: no short-form answers
            pass
      
        if "long" in target_answer_types:
            assert isinstance(data["ideal_answer"], list)
            for answer_text in data["ideal_answer"]:
                answers.append({
                    "answer": answer_text,
                    "answer_type": "long",
                })

        # if len([a for a in answers if a["answer"].strip() != ""]) == 0:
        if len(answers) == 0:
            continue

        question = {
            "question_key": question_key,
            "question": question,
            "answers": answers
        }

        # -------

        contexts.append(contexts_for_question)
        questions.append(question)

        if size > 0 and len(questions) >= size:
            break
        
    utils.write_json(path_output_file, questions)
    print(f"Processed and saved {len(questions)} questions into {path_output_file}")

    utils.write_json(path_output_file.replace(".json", ".contexts.json"), contexts)
    print(f"Processed and saved {len(contexts)} gold contexts into {path_output_file.replace('.json', '.contexts.json')}")

    
def extract_pmid_from_url(url: str) -> str:
    prefix = "http://www.ncbi.nlm.nih.gov/pubmed/"
    assert prefix in url
    return url.replace(prefix, "")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--target_answer_types", nargs="+")
    parser.add_argument("--pubmed_abstracts", type=str, default=None)
    parser.add_argument("--size", type=int, default=-1)
    args = parser.parse_args()
    main(args)

 