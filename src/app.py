"""Baseline RAG 데모 UI.

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

from ask import embed
from chat import ANSWER, REWRITE, call, call_stream, format_history

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


@st.cache_resource
def load_index():
    chunks = json.loads((DATA / "chunks.json").read_text(encoding="utf-8"))
    vectors = np.load(DATA / "vectors.npy")
    return chunks, vectors


def search(question, chunks, vectors, k):
    scores = vectors @ embed(question)
    top = np.argsort(-scores)[:k]
    return [(chunks[i], float(scores[i])) for i in top]


def build_context(hits):
    return "\n\n".join(f"[{c['doc']} p.{c['page']}]\n{c['text']}" for c, _ in hits)


def source_line(hits):
    return "출처: " + " · ".join(sorted({f"{c['doc']} p.{c['page']}" for c, _ in hits}))


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
st.caption("Baseline RAG — bge-m3 dense 검색 + qwen3:8b")

chunks, vectors = load_index()
st.session_state.setdefault("turns", [])
st.session_state.setdefault("pending", None)

with st.sidebar:
    st.header("설정")
    top_k = st.slider("검색 청크 수 (Top-K)", 1, 10, 5)
    use_rewrite = st.checkbox("후속 질문 검색어 재작성", value=True)
    if st.button("대화 초기화", use_container_width=True):
        st.session_state.turns = []
        st.session_state.pending = None
        st.rerun()
    st.divider()
    st.metric("인덱스 청크", f"{len(chunks):,}")

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
            st.caption(source_line(turn["hits"]))

    # 검색까지 끝난 질문이 있으면 여기서 답변을 스트리밍한다
    if pending:
        with st.chat_message("user"):
            st.write(pending["question"])
        with st.chat_message("assistant"):
            history = [(t["question"], t["answer"]) for t in st.session_state.turns]
            prompt = ANSWER.format(
                history=format_history(history),
                context=build_context(pending["hits"]),
                question=pending["question"],
            )
            answer = st.write_stream(call_stream(prompt))
            st.caption(source_line(pending["hits"]))

        st.session_state.turns.append({**pending, "answer": answer})
        st.session_state.pending = None

question = st.chat_input("질문을 입력하세요")
if question:
    history = [(t["question"], t["answer"]) for t in st.session_state.turns]

    query = question
    if use_rewrite and history:
        with st.spinner("검색어 재작성 중..."):
            query = call(REWRITE.format(history=format_history(history), question=question))

    with st.spinner("검색 중..."):
        hits = search(query, chunks, vectors, top_k)

    st.session_state.pending = {"question": question, "query": query, "hits": hits}
    st.rerun()
