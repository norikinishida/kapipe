import re

from tqdm import tqdm

from ...models import OpenAILLM
from ... import utils


def llm4eval(pred_path, gold_path):
    """
    Parameters
    ----------
    pred_path : str | list[Question]
    gold_path : str | list[Question]

    Returns
    -------
    dict[str, Any]
    """
    scores = {}

    # Load
    if isinstance(pred_path, str):
        pred_questions = utils.read_json(pred_path)
    else:
        pred_questions = pred_path
    assert isinstance(pred_questions, list)
    if isinstance(gold_path, str):
        gold_questions = utils.read_json(gold_path)
    else:
        gold_questions = gold_path
    assert isinstance(gold_questions, list)

    # Check
    assert len(pred_questions) == len(gold_questions)
    for pred_question, gold_question in zip(pred_questions, gold_questions):
        assert pred_question["question_key"] == gold_question["question_key"]

    # Evaluate
    scores["llm4eval"] = _llm4eval(
        pred_questions=pred_questions,
        gold_questions=gold_questions
    )
    return scores

    
def _llm4eval(pred_questions, gold_questions):
    scores = {}
    
    total_count = 0
    total_score = 0

    # Prepare prompt template
    prompt_template = utils.read_prompt_template(prompt_template_name_or_path="llm4eval_01_zeroshot")

    # Prepare LLM
    model = OpenAILLM(
        # Model
        openai_model_name="gpt-4o",
        # Generation
        max_new_tokens=512
    )

    generated_text_list = []
    for pred_question, gold_question in tqdm(zip(pred_questions, gold_questions), total=len(pred_questions)):
        total_count += 1
        question_str = gold_question["question"]
        pred_ans_str = pred_question["output_answer"]
        index_to_synonyms = {} # dict[int, list[str]]
        for gold_ans in gold_question["answers"]:
            if gold_ans["answer_type"] == "list":
                gold_ans_str = gold_ans["answer"]
                list_index = gold_ans["list_index"]
                if not list_index in index_to_synonyms:
                    index_to_synonyms[list_index] = []
                index_to_synonyms[list_index].append(gold_ans_str)
        gold_ans_list = []
        for list_index, synonyms in index_to_synonyms.items():
            gold_ans_list.append(synonyms[0])
        gold_ans_str = "Answer: " + ", ".join(gold_ans_list)

        # Ask LLM
        prompt = prompt_template.format(
           question=question_str,
           predicted_answer=pred_ans_str,
           gold_answer=gold_ans_str
        )
        generated_text = model.generate(prompt)
        generated_text_list.append(generated_text)
        score = parse_generated_text(generated_text=generated_text)

        total_score += score

    scores["total_count"] = total_count
    scores["total_score"] = total_score

    total_count = float(total_count)
    total_score = float(total_score)
    scores["average_score"] = (
        total_score / total_count
        if total_count != 0 else 0.0
    )

    scores["generated_text_list"] = generated_text_list
    return scores


def parse_generated_text(generated_text: str) -> float:
    # Parse each generated line
    generated_lines = generated_text.split("\n")
    score = 0.0
    for generated_line in generated_lines:
        generated_text = generated_line.strip()
        if generated_line == "":
            continue
        # Parse the generated_line
        if generated_line.startswith("Score:"):
            # score = float(generated_line[len("Score:"):].strip())
            match = re.search(r"Score:\s*([\d.]+)%?", generated_line)
            if match:
                score_str = match.group(1)
                score = float(score_str)
                if f"{score_str}%" in generated_line:
                    score /= 100.0
            else:
                # score = generated_line[len("Score:"):].strip()
                # score = float(score.split(" ")[0])
                score = 0.0
            break
    return score
 