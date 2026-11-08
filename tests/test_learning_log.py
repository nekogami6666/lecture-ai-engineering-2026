"""第3回: 保存・承認・拒否のテスト。実APIや個人の学習記録は使わない。"""

import json
import stat
from types import SimpleNamespace

import pytest
from pydantic_ai import DeferredToolRequests, DeferredToolResults
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from app import tools
from app.agent import ApprovalLimitExceeded, agent, print_tools, run_with_approval


@pytest.fixture
def log_dir(tmp_path, monkeypatch):
    directory = tmp_path / "learning_log"
    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", directory)
    return directory


def one_save_model(*, count=1):
    def respond(messages, info):
        returns = [p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]
        if returns:
            return ModelResponse(parts=[TextPart(str(returns[-1].content))])
        return ModelResponse(parts=[
            ToolCallPart("save_learning_log", {"title": f"第3回-{i}", "content": "承認前には保存しない。"}, tool_call_id=f"log-{i}")
            for i in range(count)
        ])
    return FunctionModel(respond)


def test_save_round_trip_unicode_and_private_file(log_dir):
    result = json.loads(tools.save_learning_log("  第3回  ", "  承認と拒否\nのテスト  "))
    assert result["saved"] is True
    path = log_dir / result["path"].split("/")[-1]
    record = json.loads(path.read_text())
    assert record["title"] == "第3回"
    assert record["content"] == "承認と拒否\nのテスト"
    assert record["created_at"].endswith("+00:00")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.parametrize("title,content", [("", "a"), (" \n", "a"), ("a", ""), ("a", " \n"), ("a" * 201, "b"), ("a", "b" * 10001)])
def test_invalid_record_does_not_create_directory(log_dir, title, content):
    assert "error" in json.loads(tools.save_learning_log(title, content))
    assert not log_dir.exists()


def test_boundary_lengths_are_accepted(log_dir):
    assert json.loads(tools.save_learning_log("a" * 200, "b" * 10000))["saved"]


def test_title_cannot_escape_directory_and_duplicate_titles_do_not_overwrite(log_dir):
    first = json.loads(tools.save_learning_log("../../escape", "one"))
    second = json.loads(tools.save_learning_log("../../escape", "two"))
    assert first["path"] != second["path"]
    assert len(list(log_dir.glob("*.json"))) == 2
    assert not (log_dir.parent / "escape").exists()


def test_symlink_directory_is_rejected(log_dir):
    outside = log_dir.parent / "outside"
    outside.mkdir()
    log_dir.symlink_to(outside, target_is_directory=True)
    assert "error" in json.loads(tools.save_learning_log("題", "本文"))
    assert not list(outside.iterdir())


def test_filename_collision_does_not_overwrite(log_dir, monkeypatch):
    monkeypatch.setattr(tools, "uuid4", lambda: SimpleNamespace(hex="collision"))
    assert json.loads(tools.save_learning_log("first", "original"))["saved"]
    assert "error" in json.loads(tools.save_learning_log("second", "replacement"))
    assert json.loads((log_dir / "collision.json").read_text())["content"] == "original"


def test_write_failure_removes_partial_file(log_dir, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("simulated disk error")
    monkeypatch.setattr(tools.json, "dump", fail)
    assert "error" in json.loads(tools.save_learning_log("題", "本文"))
    assert list(log_dir.iterdir()) == []


def test_file_as_directory_returns_error(log_dir):
    log_dir.write_text("existing content")
    assert "error" in json.loads(tools.save_learning_log("題", "本文"))
    assert log_dir.read_text() == "existing content"


def test_real_sdk_defers_before_any_write(log_dir):
    result = agent.run_sync("保存して", model=one_save_model())
    assert isinstance(result.output, DeferredToolRequests)
    assert result.output.approvals[0].tool_name == "save_learning_log"
    assert not log_dir.exists()


@pytest.mark.parametrize("answer", ["y", "Y", " yes ", "はい"])
def test_explicit_approval_resumes_and_saves_once(log_dir, answer):
    def confirm(prompt):
        assert not log_dir.exists()  # 表示・確認が終わる前には保存しない
        return answer
    result = run_with_approval(agent, "保存して", one_save_model(), input_fn=confirm)
    assert json.loads(result.output)["saved"] is True
    assert len(list(log_dir.glob("*.json"))) == 1
    assert result.usage.requests == 2


@pytest.mark.parametrize("answer", ["", "n", "no", "maybe", "保存", "y\nno"])
def test_default_and_ambiguous_input_deny_without_writes(log_dir, answer):
    result = run_with_approval(agent, "保存して", one_save_model(), input_fn=lambda _: answer)
    assert not log_dir.exists()
    assert "承認しませんでした" in result.output


@pytest.mark.parametrize("error", [EOFError, KeyboardInterrupt])
def test_interrupted_input_denies_all_pending_calls(log_dir, error):
    calls = []
    def interrupt(prompt):
        calls.append(prompt)
        raise error
    run_with_approval(agent, "保存して", one_save_model(count=2), input_fn=interrupt)
    assert len(calls) == 1
    assert not log_dir.exists()


def test_each_parallel_tool_call_requires_its_own_approval(log_dir):
    answers = iter(["y", "n"])
    run_with_approval(agent, "二つ保存して", one_save_model(count=2), input_fn=lambda _: next(answers))
    records = [json.loads(path.read_text()) for path in log_dir.glob("*.json")]
    assert [record["title"] for record in records] == ["第3回-0"]


def test_new_round_requires_fresh_confirmation(log_dir):
    count = 0
    def repeated(messages, info):
        nonlocal count
        count += 1
        if count <= 2:
            return ModelResponse(parts=[ToolCallPart("save_learning_log", {"title": f"記録{count}", "content": "内容"}, tool_call_id=f"round-{count}")])
        return ModelResponse(parts=[TextPart("完了")])
    answers = iter(["y", "n"])
    run_with_approval(agent, "保存", FunctionModel(repeated), input_fn=lambda _: next(answers))
    assert len(list(log_dir.glob("*.json"))) == 1


def test_repeated_denials_stop_at_confirmation_limit(log_dir):
    count = 0
    def repeated(messages, info):
        nonlocal count
        count += 1
        return ModelResponse(parts=[ToolCallPart("save_learning_log", {"title": "記録", "content": "内容"}, tool_call_id=f"call-{count}")])
    with pytest.raises(ApprovalLimitExceeded):
        run_with_approval(agent, "保存", FunctionModel(repeated), input_fn=lambda _: "n", max_rounds=2)
    assert not log_dir.exists()


def test_preview_shows_complete_values_before_prompt(log_dir, capsys):
    def confirm(prompt):
        output = capsys.readouterr().out
        assert '"title": "第3回-0"' in output
        assert '"content": "承認前には保存しない。"' in output
        assert "learning_log/" in output
        return "n"
    run_with_approval(agent, "保存", one_save_model(), input_fn=confirm)


def test_list_tools_exposes_new_tools_without_saving(log_dir, capsys):
    print_tools(agent)
    output = capsys.readouterr().out
    assert "[save_learning_log]" in output and "[make_quiz]" in output
    assert not log_dir.exists()


def test_quiz_uses_requested_lecture_definition():
    quiz = json.loads(tools.make_quiz(2))
    terms = [e for e in tools.load_glossary() if e["lecture"] == 2]
    assert quiz["lecture"] == 2
    assert any(quiz["question"] == f"「{e['term']}」の意味は？" and quiz["answer"] == e["definition"] for e in terms)


@pytest.mark.parametrize("lecture", [0, -1, 999])
def test_quiz_missing_lecture_is_safe(lecture):
    assert "error" in json.loads(tools.make_quiz(lecture))


def test_sdk_testmodel_dummy_arguments_do_not_crash(log_dir):
    model = TestModel()
    result = agent.run_sync("配線を確認", model=model)
    assert isinstance(result.output, DeferredToolRequests)
    rejected = DeferredToolResults(approvals={p.tool_call_id: False for p in result.output.approvals})
    final = agent.run_sync(model=model, message_history=result.all_messages(), deferred_tool_results=rejected)
    assert isinstance(final.output, str)
    assert not log_dir.exists()
