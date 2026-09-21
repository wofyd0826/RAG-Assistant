"""문서 단위 메타데이터 — 작성일과 언어.

작성일은 세 곳에서 찾되 우선순위가 있다.
  1. 파일명       저자가 직접 붙인 내용 시점이라 가장 믿을 만하다
  2. 본문 1페이지  문서에 인쇄된 날짜 ("2025. 7. 2. 안전보안팀")
  3. PDF 메타데이터 파일을 저장한 시각일 뿐이라 가장 약하다
     (예: 180220_연구수당 매뉴얼은 2018년 문서인데 메타데이터는 2026-09-08)
"""
import re

HANGUL = re.compile(r"[가-힣]")
ALPHA = re.compile(r"[A-Za-z]")


def _from_name(stem):
    m = re.search(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)", stem)          # 20251017
    if m:
        return f"{m[1]}-{m[2]}-{m[3]}"
    # 250702 / 260728 / 220317 — 앞뒤가 구분자여야 한다 (UDD02010 오탐 방지)
    m = re.search(r"(?:^|[_\-(\s])(\d{2})(\d{2})(\d{2})(?:$|[_\-)\s.])", stem)
    if m and 1 <= int(m[2]) <= 12 and 1 <= int(m[3]) <= 31:
        year = int(m[1])
        return f"{2000 + year if year < 70 else 1900 + year}-{m[2]}-{m[3]}"
    m = re.search(r"(?<!\d)(\d{2})\.(\d{2})\.(\d{2})(?!\d)", stem)        # 17.07.05
    if m and 1 <= int(m[2]) <= 12 and 1 <= int(m[3]) <= 31:
        return f"20{m[1]}-{m[2]}-{m[3]}"
    m = re.search(r"(20\d{2})\s*년", stem)                                # 2022년도
    if m:
        return m[1]
    return None


def _from_text(text):
    head = text[:1200]
    m = re.search(r"(20\d{2})\s*[.\-년]\s*(\d{1,2})\s*[.\-월]\s*(\d{1,2})", head)
    if m:
        return f"{m[1]}-{int(m[2]):02d}-{int(m[3]):02d}"
    m = re.search(r"(20\d{2})\s*[.\-년]\s*(\d{1,2})\s*[.월]", head)
    if m:
        return f"{m[1]}-{int(m[2]):02d}"
    return None


def _from_pdf(doc):
    meta = doc.metadata or {}
    for key in ("creationDate", "modDate"):
        m = re.match(r"D:(\d{4})(\d{2})(\d{2})", meta.get(key) or "")
        if m:
            return f"{m[1]}-{m[2]}-{m[3]}"
    return None


def doc_date(stem, doc, first_page_text):
    return _from_name(stem) or _from_text(first_page_text) or _from_pdf(doc) or "미상"


def doc_lang(text):
    """ko / en / mixed. 국영 병렬본이 서로를 밀어내는 것을 막는 데 쓴다."""
    ko, en = len(HANGUL.findall(text)), len(ALPHA.findall(text))
    if ko + en < 50:
        return "mixed"
    ratio = ko / (ko + en)
    return "ko" if ratio > 0.5 else "en" if ratio < 0.12 else "mixed"


def query_lang(question):
    return "ko" if HANGUL.search(question) else "en"
