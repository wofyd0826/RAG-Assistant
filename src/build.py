"""Baseline RAG 인덱스 구축.

PDF -> PyMuPDF 텍스트 추출 -> 고정 길이 청킹 -> 임베딩 -> data/
개선안과의 비교 기준선이므로 의도적으로 가장 단순한 방식을 쓴다.
"""
import json
import pathlib

import numpy as np
import pymupdf
import requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
EMBED_MODEL = "bge-m3"
CHUNK, OVERLAP, MIN_LEN = 500, 100, 50
BATCH = 16


def chunks_from_pdf(path):
    """PDF 한 개에서 (문서, 페이지, 텍스트) 청크를 뽑는다."""
    doc = pymupdf.open(path)
    for pageno, page in enumerate(doc, start=1):
        text = " ".join(page.get_text("text").split())
        for i in range(0, len(text), CHUNK - OVERLAP):
            piece = text[i : i + CHUNK]
            if len(piece) >= MIN_LEN:
                yield {"doc": path.stem, "page": pageno, "text": piece}
    doc.close()


def embed(texts):
    """Ollama 임베딩 API 호출."""
    res = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=600,
    )
    res.raise_for_status()
    return np.array(res.json()["embeddings"], dtype="float32")


def main():
    pdfs = sorted(ROOT.glob("*/*.pdf"))
    chunks = [c for pdf in pdfs for c in chunks_from_pdf(pdf)]
    print(f"PDF {len(pdfs)}개 -> 청크 {len(chunks)}개")

    vectors = []
    for i in range(0, len(chunks), BATCH):
        vectors.append(embed([c["text"] for c in chunks[i : i + BATCH]]))
        print(f"\r임베딩 {min(i + BATCH, len(chunks))}/{len(chunks)}", end="")
    print()

    vectors = np.vstack(vectors)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)  # 코사인용 정규화

    DATA.mkdir(exist_ok=True)
    np.save(DATA / "vectors.npy", vectors)
    (DATA / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False), encoding="utf-8"
    )
    print(f"저장 완료: {DATA}  (vectors {vectors.shape})")


if __name__ == "__main__":
    main()
