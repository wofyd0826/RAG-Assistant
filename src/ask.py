"""Baseline RAG 질의.

질문 -> 검색(코사인 Top-K) -> qwen3-vl:8b 답변.
발표 요구 순서대로 출력한다: 질문 / 검색 청크 / 검색 점수 / 답변 / 출처.

사용법:  python src/ask.py "법인카드 신규발급 절차 알려줘"
"""
import json
import pathlib
import sys

import numpy as np
import requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
# qwen3-vl:8b는 thinking 전용 변종이라 사고를 끌 수 없어 답변 1건당 50초가 걸린다.
# 답변 생성에는 vision이 필요 없으므로 사고를 끌 수 있는 qwen3:8b를 쓴다 (약 7배 빠름).
# vision이 필요한 페이지 인제스트 단계에서만 qwen3-vl:8b를 쓴다.
EMBED_MODEL, LLM = "bge-m3", "qwen3:8b"
TOP_K = 5

PROMPT = """당신은 DGIST 행정 매뉴얼 안내 도우미다.

아래 [참고 문서]만 근거로 답하라. 문서에 없는 내용은 절대 지어내지 말고,
근거가 없으면 "제공된 매뉴얼에서 찾을 수 없습니다"라고만 답하라.
절차를 설명할 때는 각 단계 끝에 [문서명 p.페이지] 형식으로 출처를 붙여라.

[참고 문서]
{context}

[질문]
{question}"""


def embed(text):
    res = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": EMBED_MODEL, "input": [text]},
        timeout=120,
    )
    res.raise_for_status()
    vec = np.array(res.json()["embeddings"][0], dtype="float32")
    return vec / np.linalg.norm(vec)


def search(question, chunks, vectors):
    scores = vectors @ embed(question)
    top = np.argsort(-scores)[:TOP_K]
    return [(chunks[i], float(scores[i])) for i in top]


def generate(question, hits):
    context = "\n\n".join(
        f"[{c['doc']} p.{c['page']}]\n{c['text']}" for c, _ in hits
    )
    res = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": LLM,
            "messages": [
                {"role": "user", "content": PROMPT.format(context=context, question=question)}
            ],
            "think": False,  # 베이스라인은 사고 과정 없이 빠르게
            "stream": False,
            "options": {"num_ctx": 8192, "temperature": 0.2},
        },
        timeout=600,
    )
    res.raise_for_status()
    return res.json()["message"]["content"].strip()


def main():
    question = " ".join(sys.argv[1:])
    if not question:
        print('사용법: python src/ask.py "질문 내용"')
        return

    chunks = json.loads((DATA / "chunks.json").read_text(encoding="utf-8"))
    vectors = np.load(DATA / "vectors.npy")
    hits = search(question, chunks, vectors)

    print(f"\n[질문] {question}\n")
    print("[검색된 청크]")
    for rank, (c, score) in enumerate(hits, start=1):
        print(f"  #{rank} score={score:.3f}  {c['doc']} p.{c['page']}")
        print(f"      {c['text'][:80]}...")

    print(f"\n[답변]\n{generate(question, hits)}")
    print("\n[출처]")
    for c, score in hits:
        print(f"  - {c['doc']} p.{c['page']} (score {score:.3f})")


if __name__ == "__main__":
    main()
