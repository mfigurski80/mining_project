import itertools
import pandas as pd

from to_tokens import tokenize


def read_tokens(fname):
    df_iter = pd.read_csv(fname, header=None, chunksize=1000, encoding="ISO-8859-1")
    return itertools.chain.from_iterable(tokenize(df[5]) for df in df_iter)


def read_columns(fname, columns):
    df_iter = pd.read_csv(fname, header=None, chunksize=1000, encoding="ISO-8859-1")
    return itertools.chain.from_iterable(df[columns] for df in df_iter)
