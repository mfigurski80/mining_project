import gensim.downloader
import pandas as pd
import numpy as np

from to_tokens import tokenize


CHUNK_SIZE = 1000


def vectorize(l):
    model = gensim.downloader.load("glove-twitter-25")
    return ([list(model[token]) for token in tweet if token in model] for tweet in l)


def make_even(items, n):
    return [(i + [None] * n)[:n] for i in items]


def get_lens(items):
    return list(map(lambda x: len(x), items))


FNAME = "./data/smol.csv"
OUT_FNAME = "./data/vectorized-smol.csv"
if __name__ == "__main__":
    n_done = 0
    df_iter = pd.read_csv(
        FNAME, header=None, chunksize=CHUNK_SIZE, encoding="ISO-8859-1"
    )
    mode = "w"
    for df in df_iter:
        vecs = vectorize(tokenize(df[5]))
        #  print(get_lens(vecs))
        vecs_df = pd.DataFrame(make_even(list(vecs), 30))
        vecs_df.insert(0, "Sentiment", df[0])
        #  print(vecs_df)
        vecs_df.to_csv(OUT_FNAME, header=None, index=False, mode=mode)
        n_done += CHUNK_SIZE
        mode = "a"
        print(f"{n_done} ({max(get_lens(vecs))} tokens max)...")
