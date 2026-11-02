"""実APIなしで会話の引き継ぎと失敗時の履歴を確認する。"""
from unittest.mock import Mock

import pytest
from google.genai import errors, types

from app.ask import ask, chat


def response(text="回答", usage=True):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(
            role="model", parts=[types.Part(text=text)]))],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=12, candidates_token_count=7) if usage else None,
    )


def test_second_question_receives_previous_question_and_answer(capsys):
    client = Mock()
    client.models.generate_content.side_effect = [response("猫です"), response("猫")]
    history = []
    assert ask("好きな動物は猫です", history, client=client) == "猫です"
    assert ask("何が好きでしたか", history, client=client) == "猫"
    sent = client.models.generate_content.call_args.kwargs["contents"]
    assert [c.role for c in sent] == ["user", "model", "user"]
    assert [c.parts[0].text for c in sent] == ["好きな動物は猫です", "猫です", "何が好きでしたか"]
    assert len(history) == 4
    assert "入力: 12 / 出力: 7" in capsys.readouterr().out


@pytest.mark.parametrize("failure", [errors.APIError(429, {"message": "rate limit"}), RuntimeError("network")])
def test_failure_keeps_history(failure):
    client = Mock()
    history = [types.Content(role="user", parts=[types.Part(text="前の質問")])]
    before = list(history)
    client.models.generate_content.side_effect = failure
    with pytest.raises(type(failure)):
        ask("次の質問", history, client=client)
    assert history == before


@pytest.mark.parametrize("answer", ["", "   "])
def test_empty_answer_not_added(answer):
    client = Mock()
    client.models.generate_content.return_value = response(answer)
    history = []
    with pytest.raises(ValueError):
        ask("質問", history, client=client)
    assert history == []


def test_blank_question_does_not_call_api():
    client = Mock()
    with pytest.raises(ValueError):
        ask("   ", [], client=client)
    client.models.generate_content.assert_not_called()


def test_usage_missing_is_unknown(capsys):
    client = Mock()
    client.models.generate_content.return_value = response(usage=False)
    assert ask("質問", client=client) == "回答"
    assert "入力: 不明 / 出力: 不明" in capsys.readouterr().out


def test_reset_and_exit(monkeypatch):
    client = Mock()
    client.models.generate_content.return_value = response()
    entries = iter(["最初の質問", "/reset", "次の質問", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(entries))
    chat(client)
    calls = client.models.generate_content.call_args_list
    assert len(calls) == 2
    assert len(calls[1].kwargs["contents"]) == 1


def test_eof_exits_without_request(monkeypatch):
    client = Mock()
    def eof(_):
        raise EOFError
    monkeypatch.setattr("builtins.input", eof)
    chat(client)
    client.models.generate_content.assert_not_called()
