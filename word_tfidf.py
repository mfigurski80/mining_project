import pandas as pd
import json
from collections import defaultdict
import itertools
from gensim import models, corpora

from to_tokens import tokenize


def read_tokens_from_data(fname):
    df_iter = pd.read_csv(FNAME, header=None, chunksize=1000, encoding="ISO-8859-1")
    return itertools.chain.from_iterable(tokenize(df[5]) for df in df_iter)


def build_frequency_dict(l):
    d = defaultdict(int)
    for toks in l:
        for tok in toks:
            d[tok] += 1
    return {tok: n for (tok, n) in d.items() if n > 1}


def build_translation_dictionary(toks_list):
    toks_list = list(toks_list)
    freqs = build_frequency_dict(toks_list)
    #  print(len(freqs))
    toks_list = [toks for toks in toks_list]
    #  print(len(toks_list))
    d = corpora.Dictionary(toks_list)
    return d
    #  corpus = [d.doc2bow(toks) for toks in toks_list]
    #  return corpus


def build_tfidf(corpus):
    print("building tfidf")
    m = models.TfidfModel(corpus)
    return m


if __name__ == "__main__":
    #  FNAME = "./data/training.1600000.processed.noemoticon.csv"
    FNAME = "./data/smol.csv"

    # everything is being done with generators to minimize
    # memory usage. Important for large dataset
    toks_gen = lambda: read_tokens_from_data(FNAME)  # iterator through FNAME tokens

    d = build_translation_dictionary(toks_gen())
    print(d)
    corpus = [d.doc2bow(toks) for toks in toks_gen()]
    m = build_tfidf(corpus)
    print(m)
    print(corpus[6])
    print(m[corpus[6]])

    #  print(len(freqs))
