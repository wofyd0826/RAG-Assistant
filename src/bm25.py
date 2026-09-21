"""BM25 희소 검색.

한국어는 조사가 붙어 공백 분리만으로는 "안전교육을"과 "안전교육"이 다른 토큰이 된다.
형태소 분석기를 쓰지 않고, 한글은 문자 2-gram으로 쪼개 이 문제를 피한다.
영문·숫자는 그대로 두어 'LectureDeck', 'E7', 'sugang' 같은 고유명사를 정확히 잡는다.
"""
import math
import re
from collections import Counter, defaultdict

import numpy as np

WORD = re.compile(r"[a-z0-9]+|[가-힣]+")


def tokenize(text):
    tokens = []
    for w in WORD.findall(text.lower()):
        if w[0].isascii():
            tokens.append(w)                     # lecturedeck, e7, 2025
        elif len(w) == 1:
            tokens.append(w)
        else:
            tokens += [w[i:i + 2] for i in range(len(w) - 1)]   # 안전교육 → 안전,전교,교육
    return tokens


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        tokenized = [tokenize(d) for d in docs]
        self.n = len(docs)
        self.length = np.array([len(t) for t in tokenized], dtype="float32")
        self.avg = float(self.length.mean()) or 1.0

        self.postings = defaultdict(list)        # 단어 -> [(문서번호, 빈도)]
        df = Counter()
        for i, toks in enumerate(tokenized):
            for term, freq in Counter(toks).items():
                self.postings[term].append((i, freq))
                df[term] += 1
        self.idf = {
            t: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for t, c in df.items()
        }

    def scores(self, query):
        out = np.zeros(self.n, dtype="float32")
        for term in set(tokenize(query)):
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = self.idf[term]
            for i, freq in posting:
                norm = 1 - self.b + self.b * self.length[i] / self.avg
                out[i] += idf * freq * (self.k1 + 1) / (freq + self.k1 * norm)
        return out


def rrf(rank_lists, k=60):
    """Reciprocal Rank Fusion.

    점수 스케일이 다른 두 검색을 합칠 때는 점수를 정규화하는 대신 순위만 쓴다.
    같은 문서가 양쪽에서 모두 상위면 점수가 높아진다.
    """
    fused = defaultdict(float)
    for ranks in rank_lists:
        for rank, idx in enumerate(ranks):
            fused[idx] += 1.0 / (k + rank + 1)
    return fused
