# DGIST 행정 매뉴얼 RAG 어시스턴트 (Baseline)

PDF 매뉴얼을 검색해 근거와 함께 답하는 RAG 시스템입니다. 모든 처리가 로컬에서 돌아가며
외부 API 키가 필요 없습니다.

구성은 `PDF → 텍스트 추출 → 고정 길이 청킹 → bge-m3 임베딩 → 코사인 검색 → qwen3:8b 답변` 입니다.
벡터 DB 없이 numpy 배열 하나로 검색하므로 별도 인프라가 필요 없습니다.

---

## 1. 요구사항

| 항목 | 버전 / 비고 |
|---|---|
| Python | 3.10 이상 |
| [Ollama](https://ollama.com/download) | 실행 중이어야 함 (기본 포트 11434) |
| GPU | 필수는 아니지만 없으면 답변이 수십 초 걸립니다 (VRAM 8GB 이상 권장) |
| 디스크 | 모델 약 6.5GB + 인덱스 약 10MB |

## 2. 설치

```bash
pip install pymupdf numpy requests streamlit
ollama pull bge-m3      # 임베딩 (1.2GB)
ollama pull qwen3:8b    # 답변 생성 (5.2GB)
```

설치 확인:

```bash
ollama list             # bge-m3, qwen3:8b 두 개가 보여야 합니다
```

## 3. 문서 준비

PDF는 **프로젝트 루트 바로 아래 폴더** 안에 넣습니다. 폴더 이름은 자유이고 개수 제한도 없습니다.

```
프로젝트루트/
├── 매뉴얼 취합/          ← 폴더 이름은 무엇이든 상관없음
│   ├── 수강신청 매뉴얼.pdf
│   └── TA 신청 매뉴얼.pdf
├── 매뉴얼 외/
│   └── 버스앱 안내서.pdf
└── src/
```

`src/build.py`가 `*/*.pdf` 패턴으로 찾기 때문에 **한 단계 하위 폴더까지만** 인식합니다.
루트에 바로 둔 PDF나 두 단계 아래 PDF는 무시됩니다.

**파일명이 곧 출처 표기가 됩니다.** 답변에 `[수강신청 매뉴얼 p.6]` 형태로 나오므로
알아볼 수 있는 이름을 쓰세요.

## 4. 인덱스 구축

```bash
python src/build.py
```

PDF 70개 / 1,298페이지 기준 수 분 걸립니다 (청크 1,854개, 진행률이 표시됩니다).
완료되면 `data/` 폴더에 두 파일이 생깁니다.

- `chunks.json` — 청크 원문과 출처(문서명, 페이지)
- `vectors.npy` — 청크별 1024차원 벡터

이 두 파일이 벡터 DB 역할을 합니다. 다시 돌려야 하는 경우는 세 가지뿐입니다.

1. PDF를 추가·삭제했을 때
2. 청킹 설정(`CHUNK`, `OVERLAP`)을 바꿨을 때
3. 임베딩 모델을 바꿨을 때

질문만 바꿀 때는 재실행이 필요 없습니다.

## 5. 사용법

### 웹 UI (권장)

```bash
python -m streamlit run src/app.py
```

`http://localhost:8501` 에서 열립니다. 답변이 스트리밍으로 나오고, 오른쪽에 검색된 청크와
유사도 점수가 표시됩니다. 사이드바에서 Top-K와 검색어 재작성을 켜고 끌 수 있습니다.

> `streamlit` 명령이 없다는 오류가 나면 `python -m` 을 붙이세요. Python Scripts 폴더가
> PATH에 없어서 생기는 문제이고, `python -m` 형태는 PATH와 무관하게 동작합니다.

### 단일 질문 (터미널)

```bash
python src/ask.py "TA 신청 절차 알려줘"
```

질문 → 검색된 청크 → 점수 → 답변 → 출처를 순서대로 출력합니다.

### 멀티턴 대화 (터미널)

```bash
python src/chat.py
```

이전 대화를 기억합니다. 핵심은 **검색어 재작성**입니다. `"그거 정정하려면?"` 같은 후속
질문은 그대로 검색하면 엉뚱한 문서가 나오므로, 검색 전에 이전 대화를 참고해
`"수강신청 오류 수정 방법은?"` 같은 독립 질문으로 바꾼 뒤 검색합니다.

종료는 `quit` 입니다.

## 6. 설정 바꾸기

| 상수 | 파일 | 기본값 | 설명 |
|---|---|---|---|
| `CHUNK` | `build.py` | 500 | 청크 길이(자). 바꾸면 재구축 필요 |
| `OVERLAP` | `build.py` | 100 | 청크 간 겹침(자). 바꾸면 재구축 필요 |
| `MIN_LEN` | `build.py` | 50 | 이보다 짧은 조각은 버림 |
| `BATCH` | `build.py` | 16 | 임베딩 배치 크기 |
| `EMBED_MODEL` | `ask.py` | `bge-m3` | 바꾸면 재구축 필요 |
| `LLM` | `ask.py` | `qwen3:8b` | 답변 생성 모델 |
| `TOP_K` | `ask.py` | 5 | 검색 청크 수 (UI는 슬라이더로 조절) |

`EMBED_MODEL`은 `build.py`와 `ask.py` 양쪽에 있습니다. 바꿀 때는 **둘 다** 같은 값으로
맞춰야 합니다. 다르면 질문과 문서가 서로 다른 벡터 공간에 놓여 검색이 무의미해집니다.

`LLM`은 `ask.py`에만 정의돼 있고 `chat.py`와 `app.py`가 가져다 쓰므로 한 곳만 고치면 됩니다.

## 7. 파일 구조

```
src/
├── build.py    인덱스 구축 (무겁다, 1회 실행)
├── ask.py      단일 질문 + 검색 함수·프롬프트 정의
├── chat.py     멀티턴 대화 + 검색어 재작성 + 스트리밍 호출
└── app.py      Streamlit UI
data/
├── chunks.json  청크 원문 + 출처
└── vectors.npy  임베딩 행렬
```

의존 방향은 `app.py → chat.py → ask.py` 한 방향입니다. `ask.py`가 모델 상수와 검색 함수를
갖고 있고 위쪽 모듈이 그것을 재사용합니다.

## 8. 문제 해결

**`ConnectionError` / `Connection refused`**
Ollama가 꺼져 있습니다. `ollama list`로 응답이 오는지 확인하세요.

**답변이 30초 이상 걸린다**
`ollama show <모델명>` 의 Capabilities에 `thinking`이 있는지 보세요. 있으면 사고 과정
때문에 느린 것이고, 대부분 끌 수 없습니다. `qwen3:8b`처럼 사고를 끌 수 있는 모델로 바꾸세요.

**답변이 계속 "찾을 수 없습니다"로 나온다**
해당 PDF에 추출 가능한 텍스트가 없을 수 있습니다. 스캔본이나 캡처 이미지로 만든 PDF는
텍스트 레이어가 없어 인덱스에 전혀 들어가지 않습니다. 확인 방법:

```bash
python -c "import pymupdf; d=pymupdf.open('파일.pdf'); print(sum(len(p.get_text()) for p in d))"
```

0이 나오면 이 Baseline으로는 검색할 수 없는 문서입니다.

**포트 8501을 이미 쓰고 있다**

```powershell
# Windows
$p = (Get-NetTCPConnection -LocalPort 8501 -State Listen).OwningProcess
Stop-Process -Id $p -Force
```
