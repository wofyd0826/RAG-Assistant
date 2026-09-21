# DGIST 행정 매뉴얼 RAG 어시스턴트

PDF 매뉴얼을 검색해 근거와 함께 답하는 RAG 시스템입니다. 모든 처리가 로컬에서 돌아가며
외부 API 키가 필요 없습니다.

인덱스가 두 벌이고, 둘을 바꿔가며 비교할 수 있습니다.

| | 인덱스 | 구성 |
|---|---|---|
| **Baseline** | `data/` | PDF → 텍스트 추출 → 고정 길이 청킹 |
| **개선안** | `data_vlm/` | 위 + 이미지·표 페이지를 VLM으로 재추출, 표는 통째 청킹 |

검색·생성은 두 인덱스가 공유합니다 — `bge-m3` 임베딩, 코사인 또는 BM25 하이브리드,
`qwen3:8b` 답변. 벡터 DB 없이 numpy 배열 하나로 검색하므로 별도 인프라가 필요 없습니다.

---

## 1. 요구사항

| 항목 | 버전 / 비고 |
|---|---|
| Python | 3.10 이상 |
| [Ollama](https://ollama.com/download) | 실행 중이어야 함 (기본 포트 11434) |
| GPU | 없으면 답변이 수십 초 걸립니다. 개선안 인덱스 구축에는 사실상 필수 (VRAM 8GB 이상 권장) |
| 디스크 | 모델 약 12.5GB + 인덱스·캐시 약 30MB |

## 2. 설치

```bash
pip install pymupdf numpy requests streamlit
ollama pull bge-m3        # 임베딩 (1.2GB)
ollama pull qwen3:8b      # 답변 생성 (5.2GB)
ollama pull qwen2.5vl:7b  # 페이지 이미지 판독 (6.0GB, 개선안 인덱스에만 필요)
```

설치 확인:

```bash
ollama list               # 위 세 개가 보여야 합니다
```

> **`qwen3-vl:8b`를 판독 모델로 쓰지 마세요.** 사고 과정을 끌 수 없는 변종이라 한 페이지에
> 10분을 생각하고도 본문을 출력하지 못합니다. `qwen2.5vl:7b`는 같은 페이지를 20초에 처리합니다.

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

두 인덱스를 각각 만듭니다. 비교 실험을 하지 않고 성능만 필요하면 개선안만 만들어도 됩니다.

### Baseline — `data/`

```bash
python src/build.py
```

PDF 70개 / 1,298페이지 기준 수 분 걸립니다 (청크 1,854개).

### 개선안 — `data_vlm/`

```bash
python src/build_vlm.py
```

**수 시간 걸립니다** (검증 환경 기준 약 3~4시간). 페이지를 골라 VLM에 넘기기 때문입니다.

- 텍스트가 200자 미만인 페이지 — 슬라이드·캡처로 만들어져 추출이 안 되는 경우
- 가장 긴 줄이 20자 미만인 페이지 — 표가 잘게 쪼개져 구조가 이미 깨진 경우
- 4행 3열 이상 표가 있는 페이지 — 텍스트는 있으나 행·열 대응이 무너지는 경우

70개 문서 중 **713페이지**가 대상이었고 나머지는 추출 텍스트를 그대로 씁니다.

**중간에 끊어도 됩니다.** 페이지 결과가 `cache/vlm/{문서명}/{페이지}.txt`에 저장되므로,
다시 실행하면 처리한 페이지는 0초에 넘어가고 남은 것부터 이어서 합니다. 청킹 방식만
바꿔 다시 돌릴 때도 VLM을 재호출하지 않고 임베딩만 다시 합니다.

`cache/vlm/` 안의 `.txt`를 열어보면 모델이 그 페이지를 어떻게 읽었는지 그대로 보입니다.
검색이 틀렸을 때 "추출이 잘못됐나, 검색이 잘못됐나"를 바로 가릴 수 있습니다.

### 산출물

두 스크립트 모두 같은 형태의 파일 두 개를 만듭니다.

- `chunks.json` — 청크 원문과 출처(문서명, 페이지). 개선안은 작성일·언어·검색용 텍스트도 포함
- `vectors.npy` — 청크별 1024차원 벡터

이 두 파일이 벡터 DB 역할을 합니다. 다시 돌려야 하는 경우는 셋뿐입니다.

1. PDF를 추가·삭제했을 때
2. 청킹 설정(`CHUNK`, `LOW_TEXT` 등)을 바꿨을 때
3. 임베딩 모델을 바꿨을 때

질문만 바꿀 때는 재실행이 필요 없습니다.

## 5. 사용법

### 웹 UI (권장)

```bash
python -m streamlit run src/app.py
```

`http://localhost:8501` 에서 열립니다. 답변이 스트리밍으로 나오고, 오른쪽에 검색된 청크와
점수가 표시됩니다. 사이드바에서 네 가지 개선을 하나씩 켜고 끌 수 있습니다.

```
① 인덱스                  Baseline(data/) ↔ 개선안(data_vlm/)
② 하이브리드 검색           dense 단독 ↔ dense + BM25
③ 후속 질문 검색어 재작성     개선 후 / 개선 전 / 사용 안 함
④ 문서 메타데이터           작성일 표시 + 질문과 다른 언어 문서 감점
```

③은 후속 질문에서만 동작합니다. 첫 질문에는 히스토리가 없어 아무 일도 일어나지 않습니다.

> `streamlit` 명령이 없다는 오류가 나면 `python -m` 을 붙이세요. Python Scripts 폴더가
> PATH에 없어서 생기는 문제이고, `python -m` 형태는 PATH와 무관하게 동작합니다.

> 서버를 띄워둔 채 인덱스를 다시 빌드하면 옛 데이터를 계속 씁니다. 재빌드 후에는
> 서버를 재시작하세요.

### 단일 질문 (터미널)

```bash
python src/ask.py "TA 신청 절차 알려줘"                      # Baseline

RAG_INDEX=data_vlm RAG_RETRIEVAL=hybrid RAG_META=1 \
  python src/ask.py "TA 신청 절차 알려줘"                    # 개선안 전체
```

질문 → 검색된 청크 → 점수 → 답변 → 출처를 순서대로 출력합니다.

터미널에서는 환경변수로 조합을 고릅니다. 기본값은 Baseline이라 아무것도 안 주면
개선 기능이 꺼진 채로 돕니다.

| 변수 | 값 | 기본 |
|---|---|---|
| `RAG_INDEX` | `data` / `data_vlm` | `data` |
| `RAG_RETRIEVAL` | `dense` / `hybrid` | `dense` |
| `RAG_META` | `1` 이면 켬 | 꺼짐 |

PowerShell에서는 이렇게 씁니다.

```powershell
$env:RAG_INDEX="data_vlm"; $env:RAG_RETRIEVAL="hybrid"; $env:RAG_META="1"
python src/ask.py "TA 신청 절차 알려줘"
```

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
| `CHUNK` | `build.py` / `build_vlm.py` | 500 | 청크 길이(자). 바꾸면 재구축 필요 |
| `OVERLAP` | `build.py` | 100 | 청크 간 겹침(자). Baseline 전용 |
| `MIN_LEN` | 둘 다 | 50 / 40 | 이보다 짧은 조각은 버림 |
| `LOW_TEXT` | `build_vlm.py` | 200 | 이보다 짧은 페이지는 VLM으로 다시 읽음 |
| `MAX_LINE` | `build_vlm.py` | 20 | 가장 긴 줄이 이보다 짧으면 조각난 레이아웃으로 봄 |
| `TAB_ROWS/COLS` | `build_vlm.py` | 4 / 3 | 이 크기 이상 표가 있으면 VLM으로 다시 읽음 |
| `MAX_OUT` | `build_vlm.py` | 2048 | VLM 생성 토큰 상한 (반복 폭주 방지) |
| `VLM` | `build_vlm.py` | `qwen2.5vl:7b` | 페이지 판독 모델 |
| `EMBED_MODEL` | `ask.py` / 빌드 스크립트 | `bge-m3` | 바꾸면 재구축 필요 |
| `LLM` | `ask.py` | `qwen3:8b` | 답변 생성 모델 |
| `TOP_K` | `ask.py` | 5 | 검색 청크 수 (UI는 슬라이더로 조절) |
| `LANG_PENALTY` | `ask.py` | 0.7 | 질문과 다른 언어 문서의 감점 배율 |

`EMBED_MODEL`은 빌드 스크립트와 `ask.py` 양쪽에 있습니다. 바꿀 때는 **둘 다** 같은 값으로
맞춰야 합니다. 다르면 질문과 문서가 서로 다른 벡터 공간에 놓여 검색이 무의미해집니다.

`LLM`은 `ask.py`에만 정의돼 있고 `chat.py`와 `app.py`가 가져다 쓰므로 한 곳만 고치면 됩니다.

## 7. 파일 구조

```
src/
├── build.py      Baseline 인덱스 구축
├── build_vlm.py  개선안 인덱스 구축 (VLM 판독 + 표 청킹)
├── docmeta.py    문서 작성일·언어 판정
├── bm25.py       BM25 희소 검색 + RRF 결합
├── ask.py        검색·프롬프트·단일 질문 (다른 모듈이 가져다 씀)
├── chat.py       멀티턴 대화 + 검색어 재작성 + 스트리밍 호출
└── app.py        Streamlit UI
data/       Baseline 인덱스
data_vlm/   개선안 인덱스
cache/vlm/  VLM 이 읽어낸 페이지 본문 (재실행 시 재사용)
```

의존 방향은 `app.py → chat.py → ask.py` 한 방향입니다. `ask.py`가 모델 상수·검색 함수·
프롬프트를 갖고 있고 위쪽 모듈이 그것을 재사용합니다. 검색 로직은 `ask.search` 한 곳에만
있으므로 CLI·멀티턴·UI가 항상 같은 동작을 합니다.

`data/`, `data_vlm/`, `cache/`는 문서 원문을 담고 있어 저장소에 올라가지 않습니다
(`.gitignore`). 다른 환경에서는 빌드 스크립트로 새로 만들어야 합니다.

## 8. 문제 해결

**`ConnectionError` / `Connection refused`**
Ollama가 꺼져 있습니다. `ollama list`로 응답이 오는지 확인하세요.

**답변이 30초 이상 걸린다**
`ollama show <모델명>` 의 Capabilities에 `thinking`이 있는지 보세요. 있으면 사고 과정
때문에 느린 것이고, 대부분 끌 수 없습니다. `qwen3:8b`처럼 사고를 끌 수 있는 모델로 바꾸세요.

**답변이 계속 "찾을 수 없습니다"로 나온다**
해당 PDF에 추출 가능한 텍스트가 없을 수 있습니다. 스캔본이나 캡처 이미지로 만든 PDF는
텍스트 레이어가 없어 Baseline 인덱스에 전혀 들어가지 않습니다. 확인 방법:

```bash
python -c "import pymupdf; d=pymupdf.open('파일.pdf'); print(sum(len(p.get_text()) for p in d))"
```

0이 나오면 Baseline으로는 검색할 수 없는 문서입니다. `build_vlm.py`로 만든 개선안
인덱스를 쓰면 해결됩니다.

**개선안 인덱스가 만들어지다 멈췄다**
그대로 다시 실행하면 됩니다. 처리한 페이지는 캐시에서 0초에 넘어갑니다. 진행 상황은
캐시 파일 개수로 확인합니다.

```powershell
(Get-ChildItem cache\vlm -Recurse -Filter *.txt).Count
```

**VLM 판독 결과가 이상하다**
`cache/vlm/{문서명}/{페이지}.txt`를 직접 열어보세요. 같은 문장이 반복되는 폭주 출력은
자동으로 걸러지지만(6,000자 초과, 한 줄 1,500자 초과, 중복 줄 비율 50% 초과), 놓친 것이
있으면 해당 `.txt`를 지우고 다시 실행하면 그 페이지만 재처리됩니다.

**포트 8501을 이미 쓰고 있다**

```powershell
# Windows
$p = (Get-NetTCPConnection -LocalPort 8501 -State Listen).OwningProcess
Stop-Process -Id $p -Force
```
