import json

import pytest

from hoid.extensions.guardrails import block_keywords, llm_guard, strip_pii

# --- block_keywords ---


def test_block_keywords_raises_on_match():
    guard = block_keywords(["forbidden"])
    with pytest.raises(ValueError, match="forbidden"):
        guard("this is a forbidden word")


def test_block_keywords_case_insensitive():
    guard = block_keywords(["Forbidden"])
    with pytest.raises(ValueError):
        guard("FORBIDDEN text here")


def test_block_keywords_no_false_positive():
    guard = block_keywords(["secret"])
    result = guard("this is fine")
    print(f"clean text returned unchanged: {result!r}")
    assert result == "this is fine"


def test_block_keywords_returns_text_when_clean():
    guard = block_keywords(["bad"])
    result = guard("completely safe message")
    print(f"clean text: {result!r}")
    assert result == "completely safe message"


def test_block_keywords_regex_special_chars_escaped():
    # a word with regex special chars must not be treated as a regex pattern
    guard = block_keywords(["c++"])
    with pytest.raises(ValueError):
        guard("we use c++ here")


def test_block_keywords_multiple_words():
    guard = block_keywords(["alpha", "beta"])
    with pytest.raises(ValueError):
        guard("beta is present")
    assert guard("neither") == "neither"


# --- strip_pii ---


def test_strip_pii_replaces_email():
    guard = strip_pii()
    result = guard("contact alice@example.com for help")
    print(f"after strip_pii: {result!r}")
    assert "[EMAIL]" in result
    assert "alice@example.com" not in result


def test_strip_pii_replaces_phone_dashes():
    guard = strip_pii()
    result = guard("call 555-867-5309 now")
    print(f"after strip_pii: {result!r}")
    assert "[PHONE]" in result
    assert "555-867-5309" not in result


def test_strip_pii_replaces_phone_dots():
    guard = strip_pii()
    result = guard("reach me at 555.867.5309")
    print(f"after strip_pii: {result!r}")
    assert "[PHONE]" in result


def test_strip_pii_replaces_phone_no_separator():
    guard = strip_pii()
    result = guard("number is 5558675309")
    print(f"after strip_pii: {result!r}")
    assert "[PHONE]" in result


def test_strip_pii_clean_text_unchanged():
    guard = strip_pii()
    text = "nothing sensitive here"
    assert guard(text) == text


def test_strip_pii_replaces_both_email_and_phone():
    guard = strip_pii()
    result = guard("email: bob@test.org phone: 555-123-4567")
    print(f"after strip_pii: {result!r}")
    assert "[EMAIL]" in result
    assert "[PHONE]" in result
    assert "bob@test.org" not in result
    assert "555-123-4567" not in result


# --- llm_guard ---


async def test_llm_guard_allow_passes_text_through():
    class AllowClient:
        async def chat_completions(self, messages, **kwargs):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({"status": "ALLOW", "reason": ""})
                        }
                    }
                ]
            }

    guard = llm_guard(AllowClient(), policy="no hate speech")
    result = await guard("perfectly fine text")
    print(f"llm_guard ALLOW returned: {result!r}")
    assert result == "perfectly fine text"


async def test_llm_guard_block_raises_with_reason():
    class BlockClient:
        async def chat_completions(self, messages, **kwargs):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"status": "BLOCK", "reason": "contains hate speech"}
                            )
                        }
                    }
                ]
            }

    guard = llm_guard(BlockClient(), policy="no hate speech")
    with pytest.raises(ValueError, match="contains hate speech"):
        await guard("hateful content")


async def test_llm_guard_passes_policy_in_system_message():
    captured = {}

    class CapturingClient:
        async def chat_completions(self, messages, **kwargs):
            captured["messages"] = messages
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({"status": "ALLOW", "reason": ""})
                        }
                    }
                ]
            }

    guard = llm_guard(CapturingClient(), policy="no profanity")
    await guard("clean text")
    system_content = captured["messages"][0]["content"]
    print(f"system message sent to LLM:\n{system_content}")
    assert "no profanity" in system_content
