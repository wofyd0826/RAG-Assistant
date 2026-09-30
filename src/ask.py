"""Baseline RAG 질의.

질문 -> 검색(코사인 Top-K) -> qwen3-vl:8b 답변.
발표 요구 순서대로 출력한다: 질문 / 검색 청크 / 검색 점수 / 답변 / 출처.

사용법:  python src/ask.py "법인카드 신규발급 절차 알려줘"
"""
import json
import os
import pathlib
import re
import sys

import numpy as np
import requests

# PDF 추출 텍스트에는 cp949 로 못 찍는 사설 영역 문자가 섞여 있다.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parent.parent

# 인덱스 선택. 기본은 베이스라인이고, 환경변수로 개선안을 쓴다.
#   베이스라인:  python src/ask.py "질문"
#   개선안:      RAG_INDEX=data_vlm python src/ask.py "질문"
DATA = ROOT / os.environ.get("RAG_INDEX", "data")

# 검색 방식. dense = 코사인 유사도만, hybrid = dense + BM25 를 RRF 로 결합.
#   RAG_INDEX=data_vlm RAG_RETRIEVAL=hybrid python src/ask.py "질문"
RETRIEVAL = os.environ.get("RAG_RETRIEVAL", "dense")

# 문서 메타데이터 활용. 작성일을 컨텍스트에 넣어 최신본을 고르게 하고,
# 질문과 다른 언어의 문서는 순위를 낮춘다 (국영 병렬본이 서로를 밀어내는 것 방지).
#   RAG_INDEX=data_vlm RAG_RETRIEVAL=hybrid RAG_META=1 python src/ask.py "질문"
META = os.environ.get("RAG_META", "") == "1"
LANG_PENALTY = 0.7
# qwen3-vl:8b는 thinking 전용 변종이라 사고를 끌 수 없어 답변 1건당 50초가 걸린다.
# 답변 생성에는 vision이 필요 없으므로 사고를 끌 수 있는 qwen3:8b를 쓴다 (약 7배 빠름).
# vision이 필요한 페이지 인제스트 단계에서만 qwen3-vl:8b를 쓴다.
EMBED_MODEL, LLM = "bge-m3", "qwen3:8b"
TOP_K = 5

DATE_RULE = """
참고 문서의 대괄호 안에는 작성일이 있다. 답을 쓰기 전에 아래를 순서대로 하라.

1. 작성일이 서로 다른 문서들이 같은 사안을 다루는지 확인하라. 문서명이 달라도
   같은 목록이나 같은 기준을 싣고 있으면 같은 사안으로 본다.
   같은 문서의 다른 페이지이거나 작성일이 같은 문서는 버전 관계가 아니다.
   이 규칙을 적용하지 말고, 페이지 제목(본문 첫 줄의 대괄호)을 보고 질문에 맞는 쪽을 골라라.
2. 1에 해당하면 작성일을 비교해 가장 늦은 문서 하나만 근거로 삼아라. 서로 다른 시점의
   목록·표·기준을 합치거나 "추가로 이런 것도 있습니다" 식으로 나란히 나열하지 마라.
3. 오래된 문서에는 있는데 최신 문서에는 없는 항목은 더 이상 유효하지 않은 것이다.
   답에 포함하지 말고, 예전에는 있었으나 지금은 빠졌다는 사실만 따로 알려라.
4. 1~3에 해당해서 여러 시점 중 하나를 골랐을 때만, 답 끝에 고른 문서의 작성일을
   "…기준입니다" 형식으로 밝혀라. 작성일에 적힌 만큼만 써라(연도만 있으면 연도만).
   그런 선택을 하지 않았다면 적지 마라. 답을 찾을 수 없다고 답할 때도 적지 마라.

주제가 실제로 다른 문서라면(신청 방법 안내와 점검 절차 안내 등) 오래됐다는 이유만으로
배제하지는 마라.
"""
# 출처는 참고 문서 번호로만 받는다. 예전에는 [문서명 p.페이지] 를 직접 쓰게 했는데,
# 같은 문서의 비슷한 페이지(법인카드 p.4 / 연구비카드 p.5)가 함께 들어오면
# 8B 모델이 내용은 p.4 로 쓰고 인용은 p.5 로 옮겨 적는 일이 잦았다.
# 짧은 번호는 정확히 옮기고, 번호를 문서·페이지로 바꾸는 일은 코드(cited)가 한다.
CITE = """각 단계나 문장 끝에 근거가 된 참고 문서의 번호를 [1], [2] 처럼 붙여라.
문서명이나 페이지를 직접 쓰지 말고 번호만 써라."""

PROMPT = """당신은 DGIST 행정 매뉴얼 안내 도우미다.

아래 [참고 문서]만 근거로 답하라. 문서에 없는 내용은 절대 지어내지 말고,
근거가 없으면 "제공된 매뉴얼에서 찾을 수 없습니다"라고만 답하라.
{cite}
{rules}
[참고 문서]
{context}

[질문]
{question}"""


def _meta(flag):
    """None 이면 환경변수 기본값을 쓴다. UI 에서는 토글 값을 직접 넘긴다."""
    return META if flag is None else flag


def answer_rules(meta=None, hits=None):
    """날짜 규칙은 검색된 청크의 작성일이 실제로 둘 이상일 때만 넣는다.

    8B 모델은 "같은 날짜면 적용하지 마라" 를 지키지 못했다. 한 문서의 p.4(법인카드)와
    p.5(연구비카드)만 검색됐는데도 "가장 늦은 문서" 를 고르려다 p.4 내용을 p.5 로
    인용했다. 규칙이 필요 없는 경우는 코드에서 걸러낸다.
    """
    if not _meta(meta):
        return ""
    if hits is not None:
        dates = {c.get("date", "미상") for c, _ in hits} - {"미상"}
        if len(dates) < 2:
            return ""
    return DATE_RULE


def build_context(hits, meta=None, ordered=True):
    """청크마다 [번호] 를 붙인다. 답변은 이 번호로 출처를 단다.

    번호는 검색 순위 그대로 두고(UI 의 검색 결과·cited() 와 맞추기 위해) 배치만
    문서별 → 페이지 순으로 바꾼다. 점수 순으로 넣으면 여러 쪽에 걸친 절차가
    p.8 → p.7 → p.9 → p.4 처럼 뒤섞여 들어가 모델이 단계 순서를 틀리게 답했다.
    문서끼리는 가장 높은 순위의 청크가 있는 문서부터 놓는다.

    meta 모드에서는 작성일을 같이 넣어 LLM 이 최신본을 고를 수 있게 한다.
    ordered=False 면 검색 점수 순서 그대로 넣는다 (평가의 Baseline 재현용).
    """
    def head(n, c):
        date = f" (작성일 {c.get('date', '미상')})" if _meta(meta) else ""
        return f"[{n}] {c['doc']} p.{c['page']}{date}"
    numbered = list(enumerate((c for c, _ in hits), start=1))
    if ordered:
        doc_rank = {}
        for n, c in numbered:
            doc_rank.setdefault(c["doc"], n)
        numbered.sort(key=lambda x: (doc_rank[x[1]["doc"]], x[1]["page"], x[0]))
    return "\n\n".join(f"{head(n, c)}\n{c['text']}" for n, c in numbered)


CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def cited(answer, hits):
    """답변의 [번호] 인용을 (문서, 페이지) 로 바꾼다. 처음 인용된 순서대로, 중복 없이.

    돌려주는 값은 {(문서, 페이지): [번호, ...]} 이다. 없는 번호는 버린다.
    """
    out = {}
    for m in CITE_RE.finditer(answer):
        for n in (int(x) for x in m.group(1).split(",")):
            if 1 <= n <= len(hits):
                c = hits[n - 1][0]
                nums = out.setdefault((c["doc"], c["page"]), [])
                if n not in nums:
                    nums.append(n)
    return out


def embed(text):
    res = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": EMBED_MODEL, "input": [text]},
        timeout=120,
    )
    res.raise_for_status()
    vec = np.array(res.json()["embeddings"][0], dtype="float32")
    return vec / np.linalg.norm(vec)


_bm25 = None


def _get_bm25(chunks):
    """BM25 인덱스는 첫 검색 때 한 번만 만든다 (청크 2천 개 기준 1~2초)."""
    global _bm25
    if _bm25 is None:
        from bm25 import BM25
        # 검색용 텍스트가 있으면 그것을 쓴다 (문서명·페이지가 붙어 있다)
        _bm25 = BM25([c.get("search_text", c["text"]) for c in chunks])
    return _bm25


def search(question, chunks, vectors, top_k=TOP_K, pool=50, hybrid=None, meta=None):
    hybrid = (RETRIEVAL == "hybrid") if hybrid is None else hybrid
    dense = vectors @ embed(question)

    # 질문과 다른 언어의 문서는 뒤로 민다 (배제하지는 않는다).
    # 영문 전용 문서는 한국어로 물어도 여전히 찾을 수 있어야 하므로 감점만 준다.
    penalty = None
    if _meta(meta):
        from docmeta import query_lang
        qlang = query_lang(question)
        penalty = np.array(
            [LANG_PENALTY if c.get("lang") in ("ko", "en") and c["lang"] != qlang else 1.0
             for c in chunks], dtype="float32")

    if not hybrid:
        scores = dense * penalty if penalty is not None else dense
        top = np.argsort(-scores)[:top_k]
        return [(chunks[i], float(scores[i])) for i in top]

    # 하이브리드: dense 와 BM25 의 순위를 RRF 로 합친다
    from bm25 import rrf
    sparse = _get_bm25(chunks).scores(question)
    fused = rrf([np.argsort(-dense)[:pool], np.argsort(-sparse)[:pool]])
    if penalty is not None:
        for i in fused:
            fused[i] *= float(penalty[i])

    best = sorted(fused, key=fused.get, reverse=True)[:top_k]
    ceiling = 2.0 / 61.0            # 양쪽 모두 1위일 때의 RRF 값
    return [(chunks[i], min(fused[i] / ceiling, 1.0)) for i in best]


def generate(question, hits):
    context = build_context(hits)
    res = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": LLM,
            "messages": [
                {"role": "user", "content": PROMPT.format(cite=CITE, rules=answer_rules(hits=hits),
                                                          context=context, question=question)}
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
        print(f"  [{rank}] score={score:.3f}  {c['doc']} p.{c['page']}")
        print(f"      {c['text'][:80]}...")

    answer = generate(question, hits)
    print(f"\n[답변]\n{answer}")
    print("\n[출처]")
    for (doc, page), nums in cited(answer, hits).items():
        print(f"  {''.join(f'[{n}]' for n in nums)} {doc} p.{page}")


if __name__ == "__main__":
    main()
