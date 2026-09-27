"""개선 파이프라인 인덱스 구축.

베이스라인과 다른 점 두 가지.
  1. 텍스트가 거의 없는 페이지와 표가 있는 페이지는 VLM으로 다시 읽는다.
  2. 마크다운 표는 쪼개지 않고 통째로 한 청크로 둔다.

페이지 결과를 cache/vlm/ 에 저장하므로 중간에 멈춰도 이어서 돌릴 수 있다.
베이스라인 data/ 는 건드리지 않고 data_vlm/ 에 저장한다 (A/B 비교용).

실행:  python src/build_vlm.py          (청크 500자 -> data_vlm/)
       python src/build_vlm.py 1000     (청크 1000자 -> data_vlm_1000/, 크기 비교 실험용)
"""
import base64
import hashlib
import json
import pathlib
import sys
import time

import numpy as np
import pymupdf
import requests

from docmeta import doc_date, doc_lang

ROOT = pathlib.Path(__file__).resolve().parent.parent
CHUNK = int(sys.argv[1]) if len(sys.argv) > 1 else 500
OVERLAP = CHUNK // 5    # 텍스트 페이지 겹침. 베이스라인과 같은 20%
DATA = ROOT / ("data_vlm" if CHUNK == 500 else f"data_vlm_{CHUNK}")
CACHE = ROOT / "cache" / "vlm"
TITLES = ROOT / "cache" / "title"   # build_titles.py 가 만든다. 없으면 제목 없이 진행
NOT_TITLES = {"대구경북과학기술원", "DGIST"}   # 제목 띠가 없는 페이지에서 로고 글자를 읽은 경우

EMBED_MODEL, VLM = "bge-m3", "qwen2.5vl:7b"
LOW_TEXT = 200          # 이보다 짧은 페이지는 이미지로 본다
MAX_LINE = 20           # 가장 긴 줄이 이보다 짧으면 산문이 아니라 조각난 레이아웃이다
TAB_ROWS, TAB_COLS = 4, 3   # 이 크기 이상의 표가 있으면 VLM으로 다시 읽는다
MIN_LEN, BATCH = 40, 16

MAX_OUT = 2048      # 생성 토큰 상한. 표에서 같은 셀을 무한 반복하는 폭주를 막는다
MAX_CHARS = 6000    # 한 페이지 결과가 이보다 길면 반복 루프로 보고 버린다
MAX_LINE_LEN = 1500 # 한 줄이 이보다 길어도 마찬가지
UNIQUE_RATIO = 0.5  # 서로 다른 줄의 비율이 이보다 낮으면 같은 블록을 되풀이한 것이다
RETRY = 3

PROMPT = """이 문서 페이지의 내용을 빠짐없이 텍스트로 옮겨라.

규칙:
- 표는 반드시 마크다운 표로 변환하라. 행과 열의 대응을 절대 틀리지 마라.
- 화면 캡처나 도형 안에 있는 메뉴 경로, 버튼 이름, 번호 붙은 단계도 그대로 적어라.
- 장식용 그림은 무시하라.
- 설명하지 말고 페이지 내용만 출력하라."""


def needs_vlm(page):
    text = page.get_text("text").strip()
    if len(text) < LOW_TEXT:
        return True
    # 호실 목록처럼 글자는 있으나 한 줄에 몇 자씩만 떨어져 나오는 페이지.
    # 표나 도형이 잘게 쪼개져 추출된 경우라 원문 구조가 이미 사라져 있다.
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if max((len(ln) for ln in lines), default=0) < MAX_LINE:
        return True
    try:
        return any(
            t.row_count >= TAB_ROWS and t.col_count >= TAB_COLS
            for t in page.find_tables().tables
        )
    except Exception:
        return False


def is_degenerate(text):
    """같은 내용을 되풀이하는 폭주 출력인지 본다.

    한 줄 안에서 셀이 반복되는 경우와, 같은 블록을 통째로 여러 번 찍는 경우가
    모두 나타나므로 길이·줄길이·중복비율 세 가지를 본다.
    """
    if len(text) > MAX_CHARS:
        return True
    lines = [ln.strip() for ln in text.split("\n") if len(ln.strip()) > 10]
    if not lines:
        return False
    if max(len(ln) for ln in lines) > MAX_LINE_LEN:
        return True
    return len(set(lines)) / len(lines) < UNIQUE_RATIO


def call_vlm(png):
    """VLM 호출. 일시적 오류는 재시도하고, 끝내 실패하면 None."""
    for attempt in range(RETRY):
        try:
            res = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": VLM,
                    "stream": False,
                    "messages": [{"role": "user", "content": PROMPT,
                                  "images": [base64.b64encode(png).decode()]}],
                    "options": {"num_ctx": 16384, "temperature": 0.1,
                                "num_predict": MAX_OUT, "repeat_penalty": 1.15},
                },
                timeout=600,
            )
            res.raise_for_status()
            return res.json()["message"]["content"].strip()
        except Exception:
            if attempt == RETRY - 1:
                return None
            time.sleep(5 * (attempt + 1))
    return None


def read_page(page, cache_path):
    """VLM이 필요하면 호출하고, 결과를 캐시에 남긴다.

    호출이 실패하거나 폭주하면 추출 텍스트로 대신한다 (캐시하지 않으므로
    다시 실행하면 재시도한다). 한 페이지 때문에 전체가 멈추지 않도록 한다.
    """
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8"), "cache"

    raw = " ".join(page.get_text("text").split())
    if not needs_vlm(page):
        return raw, "text"

    text = call_vlm(page.get_pixmap(dpi=150).tobytes("png"))
    if not text or is_degenerate(text):
        return raw, "fallback"

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text, encoding="utf-8")
    return text, "vlm"


def split_page(text, doc, page):
    """빈 줄 단위로 나누되, 마크다운 표는 통째로 한 청크로 유지한다.

    표를 따로 떼면 표에 제목이 없어지는 문제가 있다. 실제로
    "◎ DGIST에서 제공하는 공용 S/W 목록" 이 앞 청크에 남고 표에는
    '프로그램명(S/W명)' 뿐이라 "소프트웨어 목록" 질문에 걸리지 않았다.
    그래서 표 앞의 마지막 줄을 제목으로 함께 넣는다.
    """
    out, buf = [], ""

    def add(body):
        if len(body.strip()) >= MIN_LEN:
            out.append({"doc": doc, "page": page, "text": body.strip()})

    for block in [b.strip() for b in text.split("\n\n") if b.strip()]:
        if "|" in block and "\n" in block:
            head = ""
            if buf.strip():
                head = buf.strip().split("\n")[-1].strip()[:100]
            add(buf)
            out.append({"doc": doc, "page": page,
                        "text": f"{head}\n{block}" if head else block})
            buf = ""
        elif len(buf) + len(block) > CHUNK:
            add(buf); buf = block
        else:
            buf = f"{buf}\n\n{block}" if buf else block
    add(buf)
    return out


def split_text(text, doc, page):
    """텍스트 추출 페이지는 베이스라인과 같이 글자 수로 자르고 20% 겹친다.

    추출 텍스트는 줄바꿈을 공백으로 합친 한 줄이라 split_page 의 빈 줄 기준으로는
    나뉘지 않는다. 예전에는 이 때문에 텍스트 페이지가 통째로(최대 4천 자) 한 청크였다.
    """
    return [{"doc": doc, "page": page, "text": text[i:i + CHUNK]}
            for i in range(0, len(text), CHUNK - OVERLAP)
            if len(text[i:i + CHUNK]) >= MIN_LEN]


def page_title(stem, pageno, text):
    """본문에 빠진 페이지 제목.

    본문에 이미 있거나, 머리말의 문서명을 제목으로 읽은 경우에는 붙이지 않는다.
    문서명은 search_text 에 이미 들어가 있어 새 정보가 아니다.
    """
    path = TITLES / stem / f"{pageno:03d}.txt"
    if not path.exists():
        return ""
    title = path.read_text(encoding="utf-8").strip()
    squash = lambda s: "".join(s.split())
    if (not title or title in NOT_TITLES
            or squash(title) in squash(text) or squash(title) in squash(stem)):
        return ""
    return title


def embed(texts):
    res = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=600,
    )
    res.raise_for_status()
    return np.array(res.json()["embeddings"], dtype="float32")


def main():
    pdfs = sorted(ROOT.glob("*/*.pdf"))
    chunks, stat = [], {"text": 0, "vlm": 0, "cache": 0, "fallback": 0, "dup": 0}
    started = time.perf_counter()

    seen = set()
    for n, pdf in enumerate(pdfs, start=1):
        digest = hashlib.md5(pdf.read_bytes()).hexdigest()
        if digest in seen:      # 같은 내용이 두 번 인덱싱되면 Top-K 자리를 낭비한다
            stat["dup"] += 1
            print(f"[{n}/{len(pdfs)}] 중복 건너뜀: {pdf.stem}", flush=True)
            continue
        seen.add(digest)

        doc = pymupdf.open(pdf)
        texts, hows = [], []
        for pageno, page in enumerate(doc, start=1):
            text, how = read_page(page, CACHE / pdf.stem / f"{pageno:03d}.txt")
            stat[how] += 1
            texts.append(text)
            hows.append(how)
            done = stat["vlm"] + stat["cache"] + stat["text"] + stat["fallback"]
            print(f"[{n}/{len(pdfs)}] {done}p  vlm={stat['vlm']} cache={stat['cache']} "
                  f"text={stat['text']} fallback={stat['fallback']} dup={stat['dup']}  "
                  f"청크 {len(chunks)}  {time.perf_counter()-started:.0f}s", flush=True)
        date = doc_date(pdf.stem, doc, texts[0] if texts else "")
        doc.close()

        # 언어는 VLM 추출 결과로 판정한다. 원본 텍스트가 0자인 문서가 많아
        # get_text() 기준으로는 국문 매뉴얼도 한글 비율 0 으로 나온다.
        lang = doc_lang(" ".join(texts[:6]))
        for pageno, (text, how) in enumerate(zip(texts, hows), start=1):
            # VLM 결과(문단·표 구조가 있음)는 문단 단위로, 추출 텍스트는 글자 수로 자른다
            split = split_text if how in ("text", "fallback") else split_page
            # VLM 본문이 제목 띠를 빠뜨린 페이지는 제목을 모든 청크 앞에 붙인다.
            # 제목만 다르고 본문이 거의 같은 페이지(법인카드 / 연구비카드 절차)를
            # 검색도 LLM 도 구분할 수 있게 된다.
            title = page_title(pdf.stem, pageno, text)
            for chunk in split(text, pdf.stem, pageno):
                if title:
                    chunk["text"] = f"[{title}]\n{chunk['text']}"
                chunk["date"], chunk["lang"] = date, lang
                # 검색에만 쓰는 텍스트. 문서명을 붙여 두지 않으면 본문에 없는
                # 단어("업무용", "모바일")로는 그 문서를 찾을 수 없다.
                # 표시·LLM 컨텍스트에는 text 를 쓰므로 프롬프트는 깨끗하게 남는다.
                chunk["search_text"] = f"{pdf.stem} {pageno}쪽\n{chunk['text']}"
                chunks.append(chunk)
    print()

    vectors = []
    for i in range(0, len(chunks), BATCH):
        vectors.append(embed([c["search_text"] for c in chunks[i:i + BATCH]]))
        print(f"\r임베딩 {min(i + BATCH, len(chunks))}/{len(chunks)}", end="")
    print()

    vectors = np.vstack(vectors)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    DATA.mkdir(exist_ok=True)
    np.save(DATA / "vectors.npy", vectors)
    (DATA / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False), encoding="utf-8"
    )
    print(f"저장 완료: {DATA}  vectors {vectors.shape}  "
          f"(VLM {stat['vlm']}p, 캐시 {stat['cache']}p, 텍스트 {stat['text']}p, "
          f"폴백 {stat['fallback']}p, 중복문서 {stat['dup']}개)")


if __name__ == "__main__":
    main()
