"""VLM 으로 읽은 페이지의 제목을 따로 뽑는다.

본문을 옮기는 프롬프트는 페이지 맨 위 제목 띠를 장식으로 보고 건너뛰는 일이 있다.
법인(연구비)카드 매뉴얼 p.4 "법인카드 발급, 탈회 절차" 와 p.5 "연구비카드 발급,
탈회 절차" 는 제목만 다르고 본문 구조와 메뉴 경로가 거의 같아서, 제목이 빠지자
"법인카드 발급" 질문에 연구비카드 절차로 답했다.

본문 캐시(cache/vlm/)가 있는 페이지만 대상으로 하고, 결과는 cache/title/ 에 남긴다.
제목이 없는 페이지는 빈 파일로 저장해 다시 묻지 않는다. build_vlm.py 가 이 제목을
청크 앞에 붙인다.

실행:  python src/build_titles.py     (중간에 멈춰도 이어서 돈다)
"""
import base64
import pathlib
import time

import pymupdf
import requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
VLM_CACHE = ROOT / "cache" / "vlm"
TITLE_CACHE = ROOT / "cache" / "title"

VLM = "qwen2.5vl:7b"
MAX_TITLE = 80   # 이보다 길면 제목이 아니라 본문을 읽은 것으로 본다
RETRY = 3

# "가장 크게 쓰인 글자" 라고만 하면 본문의 큰 원형 도형("01 신규발급")을 제목으로 골랐다.
# 제목의 자리를 지정하고, 본문 도형은 제목이 아니라고 못 박아야 제목 띠를 읽는다.
PROMPT = """이 문서 페이지의 제목을 한 줄로 출력하라.
제목은 로고 아래, 페이지 맨 위 띠나 머리글 자리에 있는 큰 글자다.
본문 안의 도형·번호·표 제목은 제목이 아니다.
제목이 없으면 "없음" 이라고만 출력하라."""


def call_vlm(png):
    for attempt in range(RETRY):
        try:
            res = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": VLM,
                    "stream": False,
                    "messages": [{"role": "user", "content": PROMPT,
                                  "images": [base64.b64encode(png).decode()]}],
                    # 기본 4096 이면 큰 페이지 이미지가 넘친다
                    "options": {"num_ctx": 8192, "temperature": 0, "num_predict": 40},
                },
                timeout=300,
            )
            res.raise_for_status()
            return res.json()["message"]["content"].strip()
        except Exception:
            if attempt == RETRY - 1:
                return None
            time.sleep(5 * (attempt + 1))
    return None


def clean(out):
    """첫 줄만 쓰고, 따옴표·마크다운 기호를 벗긴다. 제목이 아니면 빈 문자열."""
    line = out.split("\n")[0].strip().strip("#*\"'` ").strip()
    if not line or "없음" in line or len(line) > MAX_TITLE:
        return ""
    return line


def main():
    pdfs = {p.stem: p for p in ROOT.glob("*/*.pdf")}
    todo = sorted(VLM_CACHE.glob("*/*.txt"))
    done = failed = 0
    started = time.perf_counter()

    for n, cached in enumerate(todo, start=1):
        stem, pageno = cached.parent.name, int(cached.stem)
        out = TITLE_CACHE / stem / cached.name
        if out.exists() or stem not in pdfs:
            continue

        with pymupdf.open(pdfs[stem]) as doc:
            png = doc[pageno - 1].get_pixmap(dpi=100).tobytes("png")
        raw = call_vlm(png)
        if raw is None:          # 캐시하지 않으므로 다시 돌리면 재시도한다
            failed += 1
            continue

        title = clean(raw)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(title, encoding="utf-8")
        done += 1
        print(f"[{n}/{len(todo)}] {time.perf_counter()-started:.0f}s  "
              f"{stem[:30]} p.{pageno} → {title or '(없음)'}", flush=True)

    print(f"\n완료: 새로 추출 {done}쪽, 실패 {failed}쪽 (다시 실행하면 재시도)")


if __name__ == "__main__":
    main()
