from __future__ import annotations

from autotagging_loop.runner.hf_sampling import (
    parquet_columns_to_read,
    plan_full_ranges,
    plan_sample_ranges,
    rows_to_tasks,
    thin_dataset_row,
)


def test_plan_sample_ranges_caps_to_target():
    ranges = plan_sample_ranges(total_size=1000, n_target=100, max_chunk=25)
    assert sum(length for _, length in ranges) == 100
    assert len(ranges) == 4


def test_plan_full_ranges_covers_total_size():
    ranges = plan_full_ranges(total_size=105, max_chunk=50)
    assert ranges == [(0, 50), (50, 50), (100, 5)]


def test_parquet_columns_to_read_skips_livecodebench_private_tests():
    columns = parquet_columns_to_read([
        "question_title",
        "private_test_cases",
        "public_test_cases",
    ])
    assert columns == ["question_title", "public_test_cases"]


def test_rows_to_tasks_normalizes_question_answer_and_choices():
    rows = [{"question": "Q?", "choices": ["A", "B"], "answer": "A"}]
    tasks = rows_to_tasks("Toy Bench", rows)
    assert tasks[0]["item_id"] == "toybench_00000"
    assert tasks[0]["question"] == "Q?"
    assert tasks[0]["choices"] == ["A", "B"]
    assert tasks[0]["reviewer_status"] == "reviewed"


def test_thin_dataset_row_keeps_livecodebench_prompt_not_private_tests():
    row = thin_dataset_row({
        "question_title": "Anti",
        "question_content": "Solve it.",
        "public_test_cases": "[public]",
        "private_test_cases": "secret" * 1000,
    })
    assert "Anti" in row["question"]
    assert "Solve it." in row["question"]
    assert "public" in row["question"]
    assert "private_test_cases" not in row


def test_thin_dataset_row_keeps_scicode_prompt_not_tests():
    row = thin_dataset_row({
        "problem_name": "Berendsen_thermostat",
        "problem_description_main": "Write a script.",
        "problem_io": "Input/output contract.",
        "general_tests": ["secret"],
    })
    assert "Berendsen_thermostat" in row["question"]
    assert "Input/output contract." in row["question"]
    assert "general_tests" not in row
