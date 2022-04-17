import itertools
import pandas as pd

from to_tokens import tokenize


def read_tokens_rows(fname):
    # use on raw dataset
    df_iter = pd.read_csv(fname, header=None, chunksize=1000, encoding="ISO-8859-1")
    return itertools.chain.from_iterable(tokenize(df[5]) for df in df_iter)


def read_column_rows(fname, columns):
    # use on raw dataset
    df_iter = pd.read_csv(fname, header=None, chunksize=1000, encoding="ISO-8859-1")
    return itertools.chain.from_iterable(df[columns] for df in df_iter)


def apply_transformation_to_dfs(fname, gen, out_fname):
    # use on any dataset
    df_iter = pd.read_csv(fname, header=None, chunksize=1000, encoding="ISO-8859-1")
    mode = "w"
    n = 0
    for df in df_iter:
        gen(df).to_csv(out_fname, mode=mode, header=None, index=False)
        mode = "a"
        print(f"{n}...")
        n += 1000


def apply_transformation_to_token_dfs(fname, gen, out_fname):
    apply_transformation_to_dfs(fname, lambda df: gen(tokenize(df[5]), df), out_fname)


def read_word_vectorized_rows(fname):
    # for use on transformed word-vector dataset
    df_iter = pd.read_csv(fname, header=None, chunksize=1000)
    return itertools.chain.from_iterable(
        ((r[1][1:].dropna() for r in df.iterrows()) for df in df_iter)
    )
