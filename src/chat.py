"""Baseline RAG 멀티턴 대화.

ask.py와 달리 대화 맥락을 유지한다. 핵심은 검색어 재작성이다.
"분실하면?" 같은 후속 질문은 그대로 검색하면 엉뚱한 청크가 나오므로,
이전 대화를 참고해 독립형 질문으로 바꾼 뒤 검색한다.

사용법:  python src/chat.py     (종료: quit)
"""
import json
import pathlib

import numpy as np
import requests

from ask import DATA, LLM, answer_rules, build_context, search

ROOT = pathlib.Path(__file__).resolve().parent.parent
KEEP_TURNS = 4  # 프롬프트에 남길 이전 대화 수
KEEP_CHARS = 200  # 이전 답변은 앞부분만 유지 (컨텍스트 절약)

# 개선 전. 대명사를 자주 놓치고("그건 누가 결재해?" 를 그대로 둠),
# 재작성할 때는 문장을 늘여 검색을 희석시켰다("이수증은?" -> "...어디에서 발급받는가").
REWRITE_V1 = """아래 대화에 이어지는 질문을, 문맥 없이도 검색 가능한 독립 질문으로 바꿔라.
대명사나 생략된 대상을 이전 대화에서 채워 넣어라.
설명 없이 바꾼 질문 한 줄만 출력하라.

[이전 대화]
{history}

[질문] {question}
[독립 질문]"""

# 개선 후. 대명사 처리를 명시하고, 용어 보존과 간결성을 요구하고,
# 바꿀 필요가 없으면 그대로 두게 했다. few-shot 은 형식 붕괴를 일으켜 쓰지 않았다.
REWRITE_V2 = """아래 대화에 이어지는 질문을, 문맥 없이도 검색 가능한 독립 질문으로 바꿔라.
대명사(그거, 그건, 이거)나 생략된 대상은 앞 대화의 구체적 대상으로 바꿔라.
원문의 전문용어·시스템명·메뉴명은 그대로 유지하라.
원문에 없던 내용을 덧붙이지 말고, 짧은 명사구로 써라.
바꿀 필요가 없으면 원문을 그대로 출력하라.
설명 없이 한 줄만 출력하라.

[이전 대화]
{history}

[질문] {question}
[독립 질문]"""

REWRITE = REWRITE_V2

ANSWER = """당신은 DGIST 행정 매뉴얼 안내 도우미다.

아래 [참고 문서]만 근거로 답하라. 문서에 없는 내용은 절대 지어내지 말고,
근거가 없으면 "제공된 매뉴얼에서 찾을 수 없습니다"라고만 답하라.
절차를 설명할 때는 각 단계 끝에 [문서명 p.페이지] 형식으로 출처를 붙여라.
{rules}
[이전 대화]
{history}

[참고 문서]
{context}

[질문]
{question}"""


def call(prompt):
    """Ollama 단발 호출."""
    res = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": LLM,
            "messages": [{"role": "user", "content": prompt}],
            "think": False,
            "stream": False,
            "options": {"num_ctx": 8192, "temperature": 0.2},
        },
        timeout=600,
    )
    res.raise_for_status()
    return res.json()["message"]["content"].strip()


def call_stream(prompt):
    """Ollama 스트리밍 호출. 생성되는 대로 조각을 내보낸다."""
    with requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": LLM,
            "messages": [{"role": "user", "content": prompt}],
            "think": False,
            "stream": True,
            "options": {"num_ctx": 8192, "temperature": 0.2},
        },
        stream=True,
        timeout=600,
    ) as res:
        res.raise_for_status()
        for line in res.iter_lines():
            if not line:
                continue
            piece = json.loads(line).get("message", {}).get("content", "")
            if piece:
                yield piece


def rewrite(history, question, improved=True):
    """후속 질문을 검색용 독립 질문으로 바꾼다.

    improved=False 면 개선 전 프롬프트를 쓴다 (데모에서 전후 비교용).

    8B 모델이라 가끔 프롬프트 형식을 흘리거나 여러 줄을 내놓는다.
    첫 줄만 쓰고, 비거나 지나치게 길면 원문을 그대로 쓴다.
    """
    template = REWRITE_V2 if improved else REWRITE_V1
    out = call(template.format(history=format_history(history), question=question))
    first = out.split("\n")[0].strip().strip('"')
    return first if 0 < len(first) <= 120 else question


def format_history(history):
    """이전 청크는 버리고 Q/A 텍스트만 남긴다 (컨텍스트가 무한히 늘지 않도록)."""
    if not history:
        return "(없음)"
    return "\n".join(
        f"Q: {q}\nA: {a[:KEEP_CHARS]}" for q, a in history[-KEEP_TURNS:]
    )


def main():
    chunks = json.loads((DATA / "chunks.json").read_text(encoding="utf-8"))
    vectors = np.load(DATA / "vectors.npy")
    history = []

    print("DGIST 매뉴얼 도우미 (종료: quit)\n")
    while True:
        try:
            question = input("질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question in ("quit", "exit", "종료"):
            break

        # 후속 질문이면 검색 전에 독립형으로 재작성
        query = question
        if history:
            query = rewrite(history, question)
            print(f"  [검색어] {query}")

        hits = search(query, chunks, vectors)
        for rank, (c, score) in enumerate(hits, start=1):
            print(f"  #{rank} {score:.3f}  {c['doc']} p.{c['page']}")

        answer = call(
            ANSWER.format(
                rules=answer_rules(),
                history=format_history(history),
                context=build_context(hits),
                question=question,
            )
        )
        print(f"\n{answer}\n")
        history.append((question, answer))


if __name__ == "__main__":
    main()
