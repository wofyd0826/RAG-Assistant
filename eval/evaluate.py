"""개선 평가: No-RAG vs Baseline vs 개선안, 청크 크기(500/1000, 두 시스템 모두), Top-k(5/10).

    python eval/evaluate.py retrieval          검색 지표 (P, R, MRR, nDCG @5, @10)
    python eval/evaluate.py generate all      다섯 조건의 답변 생성 (조건 이름 하나만 줘도 됨)
    python eval/evaluate.py pack               블라인드 채점 묶음 만들기
    python eval/evaluate.py report             채점 결과 집계

정답은 페이지 단위다. 검색된 청크의 (문서, 페이지)가 questions.json 의 gold_refs 에
있으면 관련 있는 것으로 본다. 답변 생성은 온도 0 으로 고정한다.
"""
import json
import math
import pathlib
import random
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "eval" / "results"
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import ask  # noqa: E402
import chat  # noqa: E402
from ask import CITE, PROMPT, answer_rules, build_context, cited  # noqa: E402
from chat import ANSWER, format_history  # noqa: E402

chat.TEMPERATURE = 0

# 조건마다 인덱스·검색·재작성·컨텍스트 배치를 고정한다.
# Baseline 은 처음 만든 구성 그대로: 텍스트 추출 인덱스, dense, 개선 전 재작성, 점수순 컨텍스트.
CONFIGS = {
    "norag":          None,
    "baseline":       dict(index="data",          hybrid=False, meta=False, rewrite="v1", k=5,  ordered=False),
    "baseline_c1000": dict(index="data_1000",     hybrid=False, meta=False, rewrite="v1", k=5,  ordered=False),
    "improved":       dict(index="data_vlm",      hybrid=True,  meta=True,  rewrite="v2", k=5,  ordered=True),
    "improved_c1000": dict(index="data_vlm_1000", hybrid=True,  meta=True,  rewrite="v2", k=5,  ordered=True),
    "improved_k10":   dict(index="data_vlm",      hybrid=True,  meta=True,  rewrite="v2", k=10, ordered=True),
    # k=10 을 나머지 인덱스에도 (두 번째 채점 묶음 B)
    "baseline_k10":       dict(index="data",          hybrid=False, meta=False, rewrite="v1", k=10, ordered=False),
    "baseline_c1000_k10": dict(index="data_1000",     hybrid=False, meta=False, rewrite="v1", k=10, ordered=False),
    "improved_c1000_k10": dict(index="data_vlm_1000", hybrid=True,  meta=True,  rewrite="v2", k=10, ordered=True),
}
FIRST_ROUND = ["norag", "baseline", "baseline_c1000", "improved", "improved_c1000", "improved_k10"]

NORAG = """당신은 DGIST 행정 안내 도우미다. 아래 질문에 답하라.
모르는 내용이면 지어내지 말고 모른다고 답하라.

[이전 대화]
{history}

[질문]
{question}"""

_loaded = {}


def load(index):
    """인덱스를 바꿀 때는 ask 모듈이 캐싱한 BM25 도 비운다."""
    if index not in _loaded:
        d = ROOT / index
        _loaded[index] = (json.loads((d / "chunks.json").read_text(encoding="utf-8")),
                          np.load(d / "vectors.npy"))
    ask._bm25 = None
    return _loaded[index]


def questions():
    return json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))


def prior_question(q):
    return q["context"].split("직전 질문:", 1)[-1].strip() if q.get("context") else None


# ---------------------------------------------------------------- 검색 지표

def page_metrics(hits, gold, k):
    """페이지 단위 이진 관련도. 같은 페이지가 여러 번 나오면 nDCG 에서는 첫 번째만 점수를 준다."""
    top = [(c["doc"], c["page"]) for c, _ in hits[:k]]
    rel = [p in gold for p in top]
    found = {p for p in top if p in gold}
    first = next((i for i, r in enumerate(rel, 1) if r), None)
    seen, dcg = set(), 0.0
    for i, p in enumerate(top, 1):
        if p in gold and p not in seen:
            dcg += 1 / math.log2(i + 1)
            seen.add(p)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, min(k, len(gold)) + 1))
    return {"precision": sum(rel) / k, "recall": len(found) / len(gold),
            "mrr": 1 / first if first else 0.0, "ndcg": dcg / idcg}


def coverage_groups(qs):
    """정답 페이지가 Baseline(텍스트 추출) 인덱스에 얼마나 들어 있는지로 문항을 나눈다.

    '전부 없음' 은 정답이 이미지뿐인 페이지라 Baseline 이 원천적으로 찾을 수 없는 문항이다.
    """
    base = {(c["doc"], c["page"]) for c in load("data")[0]}
    groups = {}
    for q in qs:
        gold = {(r["doc"], r["page"]) for r in q["gold_refs"]}
        have = len(gold & base)
        groups[q["no"]] = "전부 없음" if have == 0 else "일부 없음" if have < len(gold) else "모두 있음"
    return groups


def retrieval():
    qs = [q for q in questions() if q["gold_refs"]]
    groups = coverage_groups(qs)
    table = {}
    for name in ("baseline", "baseline_c1000", "improved", "improved_c1000"):
        cfg = CONFIGS[name]
        chunks, vectors = load(cfg["index"])
        per_q = []
        for q in qs:
            gold = {(r["doc"], r["page"]) for r in q["gold_refs"]}
            hits = ask.search(q["question"], chunks, vectors, top_k=10,
                              hybrid=cfg["hybrid"], meta=cfg["meta"])
            per_q.append({"no": q["no"], "category": q["category"], "baseline_coverage": groups[q["no"]],
                          **{f"{m}@{k}": v for k in (5, 10) for m, v in page_metrics(hits, gold, k).items()},
                          "top10": [[c["doc"], c["page"]] for c, _ in hits]})
        def avg(rows):
            return {m: sum(r[m] for r in rows) / len(rows) for m in rows[0] if "@" in m}
        by_group = {g: avg([r for r in per_q if r["baseline_coverage"] == g])
                    for g in ("모두 있음", "일부 없음", "전부 없음")}
        table[name] = {"mean": avg(per_q), "by_baseline_coverage": by_group, "per_question": per_q}
        print(f"{name:15} " + "  ".join(f"{m} {v:.3f}" for m, v in table[name]["mean"].items()), flush=True)
    counts = {g: sum(v == g for v in groups.values()) for g in ("모두 있음", "일부 없음", "전부 없음")}
    table["_groups"] = {"counts": counts, "questions": groups}
    OUT.mkdir(exist_ok=True)
    (OUT / "retrieval.json").write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{len(qs)}문항 기준. 저장: {OUT / 'retrieval.json'}")


# ---------------------------------------------------------------- 답변 생성

def answer_one(question, history, cfg, chunks, vectors):
    """한 조건으로 검색·답변. 멀티턴이면 history 를 쓰고 검색어를 재작성한다."""
    if cfg is None:
        return {"query": question, "hits": [],
                "answer": chat.call(NORAG.format(history=format_history(history), question=question))}
    query = question
    if history and cfg["rewrite"]:
        query = chat.rewrite(history, question, improved=cfg["rewrite"] == "v2")
    hits = ask.search(query, chunks, vectors, top_k=cfg["k"], hybrid=cfg["hybrid"], meta=cfg["meta"])
    fill = dict(cite=CITE, rules=answer_rules(cfg["meta"], hits),
                context=build_context(hits, meta=cfg["meta"], ordered=cfg["ordered"]), question=question)
    prompt = ANSWER.format(history=format_history(history), **fill) if history else PROMPT.format(**fill)
    return {"query": query, "hits": hits, "answer": chat.call(prompt)}


def generate(name):
    cfg = CONFIGS[name]
    chunks, vectors = load(cfg["index"]) if cfg else (None, None)
    rows, started = [], time.perf_counter()
    for q in questions():
        t = time.perf_counter()
        history = []
        if prior_question(q):
            prior = prior_question(q)
            history = [(prior, answer_one(prior, [], cfg, chunks, vectors)["answer"])]
        r = answer_one(q["question"], history, cfg, chunks, vectors)
        rows.append({
            "no": q["no"], "category": q["category"], "question": q["question"],
            "history": history, "query": r["query"],
            "retrieved": [{"rank": n, "doc": c["doc"], "page": c["page"], "score": round(s, 4),
                           "date": c.get("date"), "text": c["text"]}
                          for n, (c, s) in enumerate(r["hits"], 1)],
            "answer": r["answer"],
            "cited": [{"doc": d, "page": p} for d, p in cited(r["answer"], r["hits"])],
            "elapsed": round(time.perf_counter() - t, 1),
        })
        print(f"[{name}] Q{q['no']:>2} {rows[-1]['elapsed']:5.1f}s  {r['answer'][:50]!r}", flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / f"gen_{name}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{name}] 완료 {time.perf_counter() - started:.0f}s\n", flush=True)


# ---------------------------------------------------------------- 블라인드 채점

JUDGE_BATCH = 40


def pack(names=None, prefix="A"):
    """답변을 섞고 임의 ID 를 붙인다. 채점자는 어느 조건의 답인지 모른다.

    나중에 조건을 추가할 때는 새 조건만 다른 prefix 로 묶는다 (기존 채점·대응표는 유지):
        python eval/evaluate.py pack B baseline_k10 baseline_c1000_k10 improved_c1000_k10
    """
    names = names or FIRST_ROUND
    qs = {q["no"]: q for q in questions()}
    items = []
    key_path = OUT / "judge_key.json"
    key = json.loads(key_path.read_text(encoding="utf-8")) if key_path.exists() and prefix != "A" else {}
    for name in names:
        for r in json.loads((OUT / f"gen_{name}.json").read_text(encoding="utf-8")):
            items.append((name, r))
    random.Random(0).shuffle(items)
    batches = []
    for i, (name, r) in enumerate(items):
        iid = f"{prefix}{i:03d}"
        key[iid] = {"config": name, "no": r["no"]}
        q = qs[r["no"]]
        batches.append({
            "id": iid, "category": q["category"], "question": r["question"],
            "history": [{"q": a, "a": b} for a, b in r["history"]],
            "gold_answer": q["gold_answer"],
            # No-RAG 는 근거 문서가 없으므로 faithfulness 를 매기지 않는다.
            # 모델이 본 그대로 보여줘야 한다. 메타데이터 조건은 작성일을 받았으므로 함께 넣는다
            # (빼면 "…기준입니다" 날짜가 근거 없는 내용으로 채점된다).
            "context": None if name == "norag" else
                [f"[{h['rank']}] {h['doc']} p.{h['page']}"
                 + (f" (작성일 {h.get('date') or '미상'})" if CONFIGS[name]["meta"] else "")
                 + f"\n{h['text']}" for h in r["retrieved"]],
            "answer": r["answer"],
        })
    (OUT / "judge").mkdir(exist_ok=True)
    tag = "" if prefix == "A" else prefix
    for b in range(0, len(batches), JUDGE_BATCH):
        (OUT / "judge" / f"batch_{tag}{b // JUDGE_BATCH + 1:02d}.json").write_text(
            json.dumps(batches[b:b + JUDGE_BATCH], ensure_ascii=False, indent=1), encoding="utf-8")
    key_path.write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"답변 {len(batches)}개 -> 묶음 {math.ceil(len(batches) / JUDGE_BATCH)}개 ({OUT / 'judge'})")


def report():
    """채점 결과(judge/score_XX.json: [{id, correctness, faithfulness, reason}])를 조건별로 모은다."""
    key = json.loads((OUT / "judge_key.json").read_text(encoding="utf-8"))
    scores = {}
    for f in sorted((OUT / "judge").glob("score_*.json")):
        for s in json.loads(f.read_text(encoding="utf-8")):
            scores[s["id"]] = s
    missing = sorted(set(key) - set(scores))
    if missing:
        print(f"채점 안 된 답변 {len(missing)}개: {missing[:10]}")
    qs = {q["no"]: q for q in questions()}
    names = [n for n in CONFIGS if (OUT / f"gen_{n}.json").exists()]
    answers = {name: {r["no"]: r["answer"] for r in
                      json.loads((OUT / f"gen_{name}.json").read_text(encoding="utf-8"))}
               for name in names}
    rel = json.loads((OUT / "retrieval.json").read_text(encoding="utf-8"))["_groups"]["questions"]

    def norm(vals):
        return round(sum(vals) / (2 * len(vals)), 3) if vals else None

    table = {}
    for name in names:
        mine = [(key[i]["no"], scores[i]) for i in key if key[i]["config"] == name and i in scores]
        corr = [s["correctness"] for _, s in mine]
        faith = [s["faithfulness"] for _, s in mine if s.get("faithfulness") is not None]
        answerable = [no for no, _ in mine if "찾을 수 없" not in qs[no]["gold_answer"]]
        refused = [no for no in answerable if "찾을 수 없" in answers[name][no]]
        table[name] = {
            "n": len(mine),
            "correctness": norm(corr),
            "faithfulness": norm(faith), "faith_n": len(faith),
            # 답이 있는 문항에서 "찾을 수 없습니다" 라고 한 수
            "wrong_refusals": len(refused), "answerable_n": len(answerable),
            "by_category": {c: norm([s["correctness"] for no, s in mine if qs[no]["category"] == c])
                            for c in sorted({q["category"] for q in qs.values()})},
            "by_baseline_coverage": {g: norm([s["correctness"] for no, s in mine if rel.get(str(no)) == g])
                                     for g in ("모두 있음", "일부 없음", "전부 없음")},
        }
        t = table[name]
        if t["correctness"] is None:
            print(f"{name:15} 채점 없음")
            continue
        print(f"{name:15} correctness {t['correctness']:.3f}  "
              + (f"faithfulness {t['faithfulness']:.3f} (n={t['faith_n']})" if t["faithfulness"] is not None
                 else "faithfulness -")
              + f"  틀린 거절 {t['wrong_refusals']}/{t['answerable_n']}")
    (OUT / "generation.json").write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "retrieval":
        retrieval()
    elif cmd == "generate":
        targets = list(CONFIGS) if sys.argv[2:] == ["all"] else sys.argv[2:]
        for t in targets:
            generate(t)
    elif cmd == "pack":
        pack(sys.argv[3:] or None, sys.argv[2]) if len(sys.argv) > 2 else pack()
    elif cmd == "report":
        report()
    else:
        print(__doc__)
