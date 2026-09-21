"""RAG 데모 UI.

실행:  python -m streamlit run src/app.py

발표에서 요구하는 순서를 화면에 그대로 배치한다:
질문 -> 검색된 청크 -> 검색 점수 -> LLM 답변 -> 출처

답변은 스트리밍으로 표시한다. 검색까지 끝낸 뒤 rerun 하므로,
답변이 흘러나오는 동안 오른쪽에는 이미 새 검색 결과가 떠 있다.
"""
import json
import pathlib

import numpy as np
import streamlit as st

import ask
from ask import answer_rules, build_context
from chat import ANSWER, call_stream, format_history, rewrite

ROOT = pathlib.Path(__file__).resolve().parent.parent

MODES = {
    "개선안 (이미지·표 VLM 인제스트)": "data_vlm",
    "Baseline (텍스트 추출만)": "data",
}


@st.cache_resource
def load_index(name):
    d = ROOT / name
    chunks = json.loads((d / "chunks.json").read_text(encoding="utf-8"))
    vectors = np.load(d / "vectors.npy")
    return chunks, vectors


def use_index(name):
    """인덱스를 바꾸면 BM25 도 다시 만들어야 한다 (ask 모듈이 캐싱하고 있다)."""
    if st.session_state.get("bm25_for") != name:
        ask._bm25 = None
        st.session_state["bm25_for"] = name



def source_line(hits, meta):
    def label(c):
        date = f", {c.get('date', '미상')}" if meta else ""
        return f"{c['doc']} p.{c['page']}{date}"
    return "출처: " + " · ".join(sorted({label(c) for c, _ in hits}))


def render_hits(turn):
    """검색된 청크를 점수와 함께 표시."""
    if turn["query"] != turn["question"]:
        st.info(f"검색어 재작성 → {turn['query']}")

    for rank, (chunk, score) in enumerate(turn["hits"], start=1):
        st.markdown(f"**#{rank}**  ·  **`{score:.4f}`**  ·  {chunk['doc']} p.{chunk['page']}")
        st.progress(min(max(score, 0.0), 1.0))
        with st.expander("청크 내용"):
            st.text(chunk["text"])


st.set_page_config(page_title="DGIST 매뉴얼 도우미", layout="wide")
st.title("DGIST 행정 매뉴얼 도우미")
st.caption("bge-m3 검색 + qwen3:8b 생성 · 사이드바에서 Baseline 과 개선안을 바꿔가며 비교할 수 있습니다")

st.session_state.setdefault("turns", [])
st.session_state.setdefault("pending", None)

with st.sidebar:
    st.header("설정")
    mode = st.radio("① 인덱스", list(MODES), index=0)
    chunks, vectors = load_index(MODES[mode])
    use_index(MODES[mode])
    st.divider()
    hybrid = st.toggle("② 하이브리드 검색 (dense + BM25)", value=True,
                       help="BM25 를 함께 써서 'LectureDeck' 같은 고유명사를 "
                            "정확 매칭으로 잡습니다")
    rw_mode = st.radio("③ 후속 질문 검색어 재작성",
                       ["개선 후 프롬프트", "개선 전 프롬프트", "사용 안 함"], index=0,
                       help="\"그건 누가 결재해?\" 같은 후속 질문의 대명사를 "
                            "앞 대화의 대상으로 바꿔 검색합니다. "
                            "개선 전에는 대명사를 놓치거나 문장을 늘여 검색이 흐려졌습니다")
    meta = st.toggle("④ 문서 메타데이터 (작성일·언어)", value=True,
                     help="작성일을 컨텍스트에 넣어 최신본을 고르게 하고, "
                          "질문과 다른 언어의 문서는 순위를 낮춥니다")
    st.divider()
    top_k = st.slider("검색 청크 수 (Top-K)", 1, 10, 5)
    if st.button("대화 초기화", use_container_width=True):
        st.session_state.turns = []
        st.session_state.pending = None
        st.rerun()
    st.divider()
    st.metric("인덱스 청크", f"{len(chunks):,}")
    st.caption(f"`{MODES[mode]}/`")

col_chat, col_ret = st.columns([3, 2])
pending = st.session_state.pending

with col_ret:
    st.subheader("검색 결과")
    latest = pending or (st.session_state.turns[-1] if st.session_state.turns else None)
    if latest is None:
        st.caption("질문을 입력하면 검색된 청크와 점수가 표시됩니다.")
    else:
        render_hits(latest)

with col_chat:
    for turn in st.session_state.turns:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.markdown(turn["answer"])
            st.caption(source_line(turn["hits"], turn.get("meta", False)))

    # 검색까지 끝난 질문이 있으면 여기서 답변을 스트리밍한다
    if pending:
        with st.chat_message("user"):
            st.write(pending["question"])
        with st.chat_message("assistant"):
            history = [(t["question"], t["answer"]) for t in st.session_state.turns]
            prompt = ANSWER.format(
                rules=answer_rules(pending["meta"]),
                history=format_history(history),
                context=build_context(pending["hits"], meta=pending["meta"]),
                question=pending["question"],
            )
            answer = st.write_stream(call_stream(prompt))
            st.caption(source_line(pending["hits"], pending["meta"]))

        st.session_state.turns.append({**pending, "answer": answer})
        st.session_state.pending = None

question = st.chat_input("질문을 입력하세요")
if question:
    history = [(t["question"], t["answer"]) for t in st.session_state.turns]

    query = question
    if rw_mode != "사용 안 함" and history:
        with st.spinner("검색어 재작성 중..."):
            query = rewrite(history, question, improved=(rw_mode == "개선 후 프롬프트"))

    with st.spinner("검색 중..."):
        hits = ask.search(query, chunks, vectors, top_k=top_k,
                          hybrid=hybrid, meta=meta)

    st.session_state.pending = {"question": question, "query": query,
                                "hits": hits, "meta": meta}
    st.rerun()
