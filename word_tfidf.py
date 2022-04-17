import pandas as pd
import json
from collections import defaultdict
import itertools

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


if __name__ == "__main__":
    FNAME = "./data/training.1600000.processed.noemoticon.csv"
    toks_list = read_tokens_from_data(FNAME)
    freqs = build_frequency_dict(toks_list)
    print(freqs)
    print(len(freqs))
