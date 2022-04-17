import gensim.downloader
import pandas as pd
import numpy as np

from read import apply_transformation_to_token_dfs
from to_tokens import tokenize


def vectorize(l, w2v_model):
    return (
        [list(w2v_model[token]) for token in tweet if token in w2v_model] for tweet in l
    )


def make_even(items, n):
    return [(i + [None] * n)[:n] for i in items]


def get_lens(items):
    return list(map(lambda x: len(x), items))


if __name__ == "__main__":
    OUT_FNAME = "./data/vectorized-smol.csv"
    FNAME = "./data/smol.csv"
    w2v = gensim.downloader.load("glove-twitter-25")

    def transform(toks, df):
        vecs = vectorize(toks, w2v)
        #  print(df.iloc[:, 0])
        #  print(get_lens(vecs))
        vecs_df = pd.DataFrame(make_even(list(vecs), 30))
        #  vecs_df.insert(0, "Sentiment", df.iloc[:, 0])
        vecs_df = pd.concat([df.iloc[:, 0].reset_index(drop=True), vecs_df], axis=1)
        print(vecs_df)
        return vecs_df

    apply_transformation_to_token_dfs(FNAME, transform, OUT_FNAME)

    #  print(df[0])
    #  #  print(vecs_df)
    #  vecs_df.to_csv(OUT_FNAME, header=None, index=False, mode=mode)
    #  n_done += CHUNK_SIZE
    #  mode = "a"
    #  print(f"{n_done}...")
    #  #  print(f"{n_done} ({max(get_lens(vecs))} tokens max)...")
