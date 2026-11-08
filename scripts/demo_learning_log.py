"""第3回の承認フローを再現する。実LLM・APIキー・個人データは使わない。

実行: uv run python -m scripts.demo_learning_log
FunctionModel が固定のツール要求を返し、Pydantic AI が実際に保留・再開する。
保存先は一時ディレクトリで、終了時に削除する。
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic_ai import DeferredToolRequests
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from app import tools
from app.agent import agent, print_trace, run_with_approval


def scripted_model():
    def respond(messages, info):
        returns = [p for message in messages for p in message.parts if isinstance(p, ToolReturnPart)]
        if returns:
            return ModelResponse(parts=[TextPart(str(returns[-1].content))])
        return ModelResponse(parts=[ToolCallPart(
            "save_learning_log",
            {"title": "第3回の確認", "content": "保存前に利用者の承認を求め、拒否された場合は保存しない。"},
            tool_call_id="demo-log-1",
        )])
    return FunctionModel(respond)


def main():
    original_directory = tools.LEARNING_LOG_DIR
    try:
        with TemporaryDirectory() as root:
            tools.LEARNING_LOG_DIR = Path(root) / "pending"
            pending = agent.run_sync("学習内容を保存して", model=scripted_model())
            print("model=FunctionModel (固定応答のテスト用モデル、実LLMではない)")
            print("approval_pending=", isinstance(pending.output, DeferredToolRequests))
            print("directory_exists_before_approval=", tools.LEARNING_LOG_DIR.exists())
            print("requests_before_approval=", pending.usage.requests)

            for scenario, choice in [("approve", "y"), ("deny", "n"), ("empty", "")]:
                tools.LEARNING_LOG_DIR = Path(root) / scenario
                print(f"\n=== {scenario}: input={choice!r} ===")
                result = run_with_approval(agent, "学習内容を保存して", scripted_model(), input_fn=lambda _: choice)
                print_trace(result)
                files = list(tools.LEARNING_LOG_DIR.glob("*.json"))
                print("saved_files=", len(files))
                if files:
                    record = json.loads(files[0].read_text(encoding="utf-8"))
                    print("saved_title=", record["title"])
                    print("saved_content=", record["content"])
                print("requests_total=", result.usage.requests)
    finally:
        tools.LEARNING_LOG_DIR = original_directory


if __name__ == "__main__":
    main()
