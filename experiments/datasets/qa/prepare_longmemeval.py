import argparse
import datetime
import os
from typing import Any

from tqdm import tqdm

from kapipe import utils


QUESTION_TYPES: set[str] = {
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "temporal-reasoning",
    "knowledge-update",
    "multi-session",
}


def main(args: argparse.Namespace) -> None:
    input_small_file: str = args.input_small_file
    input_medium_file: str = args.input_medium_file
    output_dir: str = args.output_dir

    # Validate that both official LongMemEval files exist before proceeding
    for input_file in [input_small_file, input_medium_file]:
        if not os.path.isfile(input_file):
            raise FileNotFoundError(
                f"Missing a LongMemEval input file: {input_file}"
            )

    # Convert the two released history-length settings
    # ("small" and "medium") independently.
    setting_to_input_file = {
        "small": input_small_file,
        "medium": input_medium_file,
    }
    setting_to_questions: dict[str, list[dict[str, Any]]] = {}
    setting_to_sessions: dict[str, list[dict[str, Any]]] = {}
    for setting, input_file in setting_to_input_file.items():
        # Read and validate the original LongMemEval data for the current setting
        original_data = utils.read_json(input_file)
        if not isinstance(original_data, list):
            raise TypeError(
                f"Expected a LongMemEval list in {input_file}"
            )

        # Convert the original data into questions and corresponding session records
        questions, sessions = convert_dataset(
            original_data=original_data,
            setting=setting,
        )
        setting_to_questions[setting] = questions
        setting_to_sessions[setting] = sessions

    # Create the shared destination after validating every released instance
    utils.mkdir(output_dir)

    # Save questions and session records
    for setting in ["small", "medium"]:
        output_questions_file = os.path.join(
            output_dir,
            f"{setting}.json",
        )
        output_sessions_file = os.path.join(
            output_dir,
            f"{setting}.sessions.json",
        )
        questions = setting_to_questions[setting]
        sessions = setting_to_sessions[setting]

        utils.write_json(
            output_questions_file,
            questions,
            ensure_ascii=False,
        )
        utils.write_json(
            output_sessions_file,
            sessions,
            ensure_ascii=False,
        )

        print(
            f"Processed and saved {len(questions)} LongMemEval {setting} "
            f"questions into {output_questions_file}"
        )
        print(
            f"Processed and saved {len(sessions)} LongMemEval {setting} "
            f"session histories into {output_sessions_file}"
        )


def convert_dataset(
    original_data: list[dict[str, Any]],
    setting: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assert setting in {"small", "medium"}

    questions: list[dict[str, Any]] = []
    sessions_for_questions: list[dict[str, Any]] = []
    seen_question_keys: set[str] = set()

    # Convert every evaluation instance while preserving the released order
    for data in tqdm(
        original_data,
        desc=f"Processing LongMemEval {setting}",
    ):
        validate_original_instance(data=data)

        question_key = f"longmemeval/{setting}/{data['question_id']}"
        if question_key in seen_question_keys:
            raise ValueError(
                f"Duplicate LongMemEval question key: {question_key}"
            )
        seen_question_keys.add(question_key)

        # Disambiguate repeated official IDs without adding a separate index
        session_ids = build_unique_session_ids(
            original_session_ids=data["haystack_session_ids"],
        )

        # Move all answer annotations into the question-side record
        answer_sessions = build_answer_sessions(
            data=data,
            session_ids=session_ids,
        )
        question = {
            "question_key": question_key,
            "question": data["question"],
            "question_date": data["question_date"],
            "question_type": data["question_type"],
            "answers": [
                {
                    "answer": str(data["answer"]),
                }
            ],
            "answer_sessions": answer_sessions,
        }

        # Keep only information available before the question in sessions
        sessions = build_sessions(
            data=data,
            session_ids=session_ids,
        )
        sessions_for_question = {
            "question_key": question_key,
            "sessions": sessions,
        }

        questions.append(question)
        sessions_for_questions.append(sessions_for_question)

    return questions, sessions_for_questions


def validate_original_instance(data: dict[str, Any]) -> None:
    # Detect changes to the complete official LongMemEval instance schema
    assert isinstance(data, dict)
    assert set(data) == {
        "question_id",
        "question_type",
        "question",
        "answer",
        "question_date",
        "haystack_session_ids",
        "haystack_dates",
        "haystack_sessions",
        "answer_session_ids",
    }, data.keys()

    # Validate every question-side field used by the conversion
    assert isinstance(data["question_id"], str), data["question_id"]
    assert data["question_type"] in QUESTION_TYPES, data["question_id"]
    assert isinstance(data["question"], str), data["question_id"]
    assert isinstance(data["answer"], (str, int)), data["question_id"]
    assert isinstance(data["question_date"], str), data["question_id"]
    parse_session_date(session_date=data["question_date"])

    # Require the three parallel session arrays to have identical lengths
    assert isinstance(data["haystack_session_ids"], list), data["question_id"]
    assert isinstance(data["haystack_dates"], list), data["question_id"]
    assert isinstance(data["haystack_sessions"], list), data["question_id"]
    n_sessions = len(data["haystack_session_ids"])
    if len(data["haystack_dates"]) != n_sessions:
        raise ValueError(
            f"Mismatched session dates for {data['question_id']}"
        )
    if len(data["haystack_sessions"]) != n_sessions:
        raise ValueError(
            f"Mismatched session contents for {data['question_id']}"
        )

    # Validate every session identifier, date, and structured utterance
    assert all(
        isinstance(session_id, str)
        for session_id in data["haystack_session_ids"]
    ), data["question_id"]
    for session_date in data["haystack_dates"]:
        assert isinstance(session_date, str), data["question_id"]
        parse_session_date(session_date=session_date)
    for session in data["haystack_sessions"]:
        assert isinstance(session, list), data["question_id"]
        for utterance in session:
            assert isinstance(utterance, dict), data["question_id"]
            assert set(utterance).issubset(
                {"role", "content", "has_answer"}
            ), data["question_id"]
            assert {"role", "content"}.issubset(utterance), data["question_id"]
            assert utterance["role"] in {"user", "assistant"}, data[
                "question_id"
            ]
            assert isinstance(utterance["content"], str), data["question_id"]
            if "has_answer" in utterance:
                assert isinstance(
                    utterance["has_answer"],
                    bool,
                ), data["question_id"]

    # Require every released answer-session reference to resolve uniquely
    assert isinstance(data["answer_session_ids"], list), data["question_id"]
    assert all(
        isinstance(session_id, str)
        for session_id in data["answer_session_ids"]
    ), data["question_id"]
    if len(set(data["answer_session_ids"])) != len(
        data["answer_session_ids"]
    ):
        raise ValueError(
            f"Duplicate answer session IDs for {data['question_id']}"
        )
    for answer_session_id in data["answer_session_ids"]:
        n_matching_sessions = data["haystack_session_ids"].count(
            answer_session_id
        )
        if n_matching_sessions != 1:
            raise ValueError(
                f"Expected one session for answer session ID "
                f"{answer_session_id} in {data['question_id']}, but found "
                f"{n_matching_sessions}"
            )


def build_unique_session_ids(
    original_session_ids: list[str],
) -> list[str]:
    session_id_to_total: dict[str, int] = {}

    # Count every occurrence before assigning deterministic suffixes
    for session_id in original_session_ids:
        if session_id not in session_id_to_total:
            session_id_to_total[session_id] = 0
        session_id_to_total[session_id] += 1

    session_id_to_seen: dict[str, int] = {}
    session_ids: list[str] = []

    # Keep unique official IDs and suffix only repeated official IDs
    for original_session_id in original_session_ids:
        if session_id_to_total[original_session_id] == 1:
            session_id = original_session_id
        else:
            if original_session_id not in session_id_to_seen:
                session_id_to_seen[original_session_id] = 0
            session_id_to_seen[original_session_id] += 1
            session_id = (
                f"{original_session_id}#"
                f"{session_id_to_seen[original_session_id]}"
            )
        session_ids.append(session_id)

    # Reject a collision with an official ID that already contains the suffix
    if len(set(session_ids)) != len(session_ids):
        raise ValueError("Failed to create unique LongMemEval session IDs")
    return session_ids


def build_answer_sessions(
    data: dict[str, Any],
    session_ids: list[str],
) -> list[dict[str, Any]]:
    session_id_to_utterances = {
        original_session_id: (session_id, utterances)
        for original_session_id, session_id, utterances in zip(
            data["haystack_session_ids"],
            session_ids,
            data["haystack_sessions"],
            strict=True,
        )
    }
    answer_sessions: list[dict[str, Any]] = []

    # Preserve official answer-session order and relocate turn annotations
    for original_answer_session_id in data["answer_session_ids"]:
        answer_session_id, utterances = session_id_to_utterances[
            original_answer_session_id
        ]
        answer_utterance_indices = [
            utterance_index
            for utterance_index, utterance in enumerate(utterances)
            if (
                "has_answer" in utterance
                and utterance["has_answer"] is True
            )
        ]
        answer_sessions.append(
            {
                "answer_session_id": answer_session_id,
                "answer_utterance_indices": answer_utterance_indices,
            }
        )

    return answer_sessions


def build_sessions(
    data: dict[str, Any],
    session_ids: list[str],
) -> list[dict[str, Any]]:
    sessions: list[dict[str, Any]] = []

    # Copy structured histories without carrying question-derived labels
    for session_id, session_date, original_utterances in zip(
        session_ids,
        data["haystack_dates"],
        data["haystack_sessions"],
        strict=True,
    ):
        utterances = [
            {
                "role": original_utterance["role"],
                "content": original_utterance["content"],
            }
            for original_utterance in original_utterances
        ]
        sessions.append(
            {
                "session_id": session_id,
                "session_date": session_date,
                "utterances": utterances,
            }
        )

    # Sort chronologically while preserving released order for tied timestamps
    sessions.sort(
        key=lambda session: parse_session_date(
            session_date=session["session_date"]
        )
    )
    return sessions


def parse_session_date(session_date: str) -> datetime.datetime:
    # Ignore the redundant English weekday to avoid locale-dependent parsing
    date_and_time = f"{session_date[:10]} {session_date[-5:]}"
    return datetime.datetime.strptime(date_and_time, "%Y/%m/%d %H:%M")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_small_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--input_medium_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    main(args)
