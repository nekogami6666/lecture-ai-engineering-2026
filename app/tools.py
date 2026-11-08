"""第3回: エージェントに持たせるツール。

ここに書いた関数の docstring と型ヒントが、そのままモデルへの「ツールの説明文」になる。
説明文を書き換えると、モデルがツールを使うかどうか・どんな引数で呼ぶかが変わる。
"""

import json
import os
import random
from datetime import datetime, timezone
from functools import cache
from pathlib import Path
from uuid import uuid4

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MAX_RESULTS = 3
LEARNING_LOG_DIR = Path(__file__).resolve().parent.parent / "learning_log"
MAX_TITLE_LENGTH = 200
MAX_CONTENT_LENGTH = 10_000


@cache
def load_glossary() -> list[dict]:
    with open(DATA_DIR / "glossary.json", encoding="utf-8") as f:
        return json.load(f)["terms"]


def _normalize(text: str) -> str:
    return text.casefold().replace(" ", "").replace("　", "")


def _names(entry: dict) -> list[str]:
    return [_normalize(entry["term"]), _normalize(entry["id"])] + [
        _normalize(a) for a in entry["aliases"]
    ]


def search_glossary(query: str) -> str:
    """講座の用語集から用語を検索し、定義と、どの回で扱うかを返す。

    AIエンジニアリングの用語や、この講座（AIE2026）に固有の言葉について
    答える前に使うこと。query には調べたい用語を1つだけ入れる（例: "LoRA"）。
    """
    q = _normalize(query)
    if not q:
        return "検索語が空です。調べたい用語を1つ指定してください。"
    glossary = load_glossary()

    exact = [e for e in glossary if q in _names(e)]
    # 「r」のような短い別名が何にでも当たらないよう、名前側が短いものは除く
    partial = [
        e for e in glossary
        if e not in exact and any(q in n or (len(n) >= 3 and n in q) for n in _names(e))
    ]
    in_definition = [
        e for e in glossary
        if e not in exact and e not in partial and q in _normalize(e["definition"])
    ]
    hits = (exact + partial + in_definition)[:MAX_RESULTS]

    if not hits:
        return f"用語集に「{query}」は見つかりませんでした。"
    return json.dumps(
        [
            {
                "term": e["term"],
                "lecture": e["lecture"],
                "definition": e["definition"],
                "related": e["related"],
            }
            for e in hits
        ],
        ensure_ascii=False,
    )


def get_lecture_summary(lecture: int) -> str:
    """指定した回（1始まりの整数）の講義の要点を返す。

    「第3回では何を学ぶ？」のように、回の内容や関係を聞かれたときに使う。
    """
    path = DATA_DIR / "lectures" / f"{lecture:02d}.md"
    if not path.exists():
        available = sorted(p.stem for p in (DATA_DIR / "lectures").glob("*.md"))
        return f"第{lecture}回の要点はまだありません。用意されている回: {', '.join(available)}"
    return path.read_text(encoding="utf-8")


def make_quiz(lecture: int) -> str:
    """指定した回の用語集から、確認問題を1問とその正解を返す。

    「第2回の確認問題を出して」など、復習の出題を求められたときに使う。
    lecture は1始まりの講義番号。利用者にはまず問題だけを出し、解答後に正解を示す。
    """
    candidates = [entry for entry in load_glossary() if entry["lecture"] == lecture]
    if not candidates:
        return json.dumps({"error": f"第{lecture}回の用語はありません。"}, ensure_ascii=False)
    entry = random.choice(candidates)
    return json.dumps(
        {"question": f"「{entry['term']}」の意味は？", "answer": entry["definition"], "lecture": lecture},
        ensure_ascii=False,
    )


def save_learning_log(title: str, content: str) -> str:
    """利用者が保存を希望した学習記録を、learning_log/ に新規保存する。

    title は1〜200文字、content は1〜10000文字。タイトルと本文を利用者に
    確認してもらってから実行する承認必須ツール。パスや秘密情報は渡さない。
    既存の記録は上書きしない。成功時は saved と相対パス、失敗時は error を返す。
    """
    title, content = title.strip(), content.strip()
    if not (1 <= len(title) <= MAX_TITLE_LENGTH and 1 <= len(content) <= MAX_CONTENT_LENGTH):
        return json.dumps({"error": "タイトルは1〜200文字、本文は1〜10000文字で指定してください。"}, ensure_ascii=False)

    record = {"title": title, "content": content, "created_at": datetime.now(timezone.utc).isoformat()}
    # モデルには保存先を決めさせない。固定ディレクトリを開き、その中だけに新規作成する。
    # dir_fd と O_NOFOLLOW は講座の Linux / Codespaces 環境で利用できる。
    filename = f"{uuid4().hex}.json"
    try:
        LEARNING_LOG_DIR.mkdir(mode=0o700, exist_ok=True)
        directory_fd = os.open(LEARNING_LOG_DIR, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            file_fd = os.open(
                filename, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600, dir_fd=directory_fd,
            )
            try:
                with os.fdopen(file_fd, "w", encoding="utf-8") as file:
                    json.dump(record, file, ensure_ascii=False, indent=2)
                    file.write("\n")
            except BaseException:
                # 書き込み失敗時に不完全な記録を残さない。
                os.unlink(filename, dir_fd=directory_fd)
                raise
        finally:
            os.close(directory_fd)
    except OSError:
        return json.dumps({"error": "学習記録を保存できませんでした。保存先と空き容量を確認してください。"}, ensure_ascii=False)
    return json.dumps({"saved": True, "path": f"learning_log/{filename}"}, ensure_ascii=False)
