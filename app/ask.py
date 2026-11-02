"""第1回: LLMへの単発質問と、会話履歴を保持する連続質問。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --chat     # 履歴を保持して連続質問する

外部情報を検索するツールは持たない。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors, types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(question: str, history: list[types.Content] | None = None, *, client=None) -> str:
    """成功した質問とモデルの応答だけを履歴に追加する。"""
    if not question.strip():
        raise ValueError("質問を入力してください。")
    if client is None:
        with genai.Client(api_key=get_api_key()) as owned_client:
            return ask(question, history, client=owned_client)
    contents = list(history) if history is not None else []
    user_content = types.Content(role="user", parts=[types.Part(text=question)])
    contents.append(user_content)
    response = client.models.generate_content(model=MODEL, contents=contents)
    answer = response.text
    if not answer or not answer.strip():
        raise ValueError("空の回答が返りました。履歴は変更していません。")
    # 署名などを含む応答パーツを失わないよう、モデルのContentをそのまま保持する。
    model_content = response.candidates[0].content
    if model_content is None:
        raise ValueError("応答の内容がありません。履歴は変更していません。")
    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count if usage else None
    output_tokens = usage.candidates_token_count if usage else None
    print(f"[tokens] 入力: {input_tokens if input_tokens is not None else '不明'} / "
          f"出力: {output_tokens if output_tokens is not None else '不明'}")
    if history is not None:
        history.extend([user_content, model_content])
    return answer


def chat(client) -> None:
    """プロセス内で履歴を保持して連続質問する。"""
    history: list[types.Content] = []
    print("連続質問モード: /reset で履歴消去、/exit で終了")
    while True:
        try:
            question = input("質問> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if question == "/exit":
            return
        if question == "/reset":
            history.clear()
            print("履歴を消去しました。")
            continue
        if not question:
            continue
        try:
            print(ask(question, history, client=client))
        except errors.APIError as e:
            print(api_error_message(e.code, e.message))
        except ValueError as e:
            print(e)


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--chat", action="store_true", help="会話履歴を保持して連続質問する")
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    if args.chat:
        with genai.Client(api_key=get_api_key()) as client:
            chat(client)
        return
    print(f"[質問] {args.question}\n")
    try:
        print(ask(args.question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)
    except ValueError as e:
        print(e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
