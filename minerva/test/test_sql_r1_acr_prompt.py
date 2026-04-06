from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_sql_prompt_helpers():
    module_path = Path(__file__).resolve().parents[1] / "sql_r1_acr_prompt.py"
    spec = importlib.util.spec_from_file_location("sql_r1_acr_prompt_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module.is_sql_r1_preformatted_messages, module.try_build_sql_r1_acr_messages


def _prompt_messages() -> list[dict[str, str]]:
    prompt = (
        "<|im_start|>system\n"
        "You are a helpful SQL expert assistant.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "How many users are there?\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>"
    )
    return [{"role": "user", "content": prompt}]


def test_try_build_sql_r1_acr_messages_inserts_block():
    is_sql_r1_preformatted_messages, try_build_sql_r1_acr_messages = _load_sql_prompt_helpers()
    messages = _prompt_messages()

    assert is_sql_r1_preformatted_messages(messages) is True

    acr_messages, skipped = try_build_sql_r1_acr_messages(
        messages,
        "SELECT COUNT(*) FROM users;",
        prompt_too_long=lambda _: False,
    )

    assert skipped is False
    assert acr_messages[0]["content"].endswith("<|im_start|>assistant\n<think>")
    assert "GROUND_TRUTH_SQL:" in acr_messages[0]["content"]
    assert "SELECT COUNT(*) FROM users;" in acr_messages[0]["content"]


def test_try_build_sql_r1_acr_messages_respects_prompt_limit():
    _, try_build_sql_r1_acr_messages = _load_sql_prompt_helpers()
    messages = _prompt_messages()

    acr_messages, skipped = try_build_sql_r1_acr_messages(
        messages,
        "SELECT COUNT(*) FROM users;",
        prompt_too_long=lambda _: True,
    )

    assert skipped is True
    assert acr_messages == messages
