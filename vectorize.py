import gensim.downloader
import pandas as pd
import string

STOPLIST = set("for a and of the in to so".split())
CHUNK_SIZE = 1000


def tokenize(l):

    trans = str.maketrans("", "", string.punctuation)
    return [
        [
            w.translate(trans)
            for w in tweet.lower().split()
            if (w not in STOPLIST and not w.startswith("@"))
        ]
        for tweet in l
    ]
    # TODO: implement bi-gram detector model


def vectorize(l):
    model = gensim.downloader.load("glove-twitter-25")
    return [[list(model[token]) for token in tweet if token in model] for tweet in l]


def make_even(items, n):
    return [(i + [None] * n)[:n] for i in items]


def get_lens(items):
    return list(map(lambda x: len(x), items))


if __name__ == "__main__":
    n_done = 0
    df_iter = pd.read_csv(
        "./data/smol.csv", header=None, chunksize=CHUNK_SIZE, encoding="ISO-8859-1"
    )
    mode = "w"
    for df in df_iter:
        vecs = vectorize(tokenize(df[5]))
        #  print(get_lens(vecs))
        vecs_df = pd.DataFrame(make_even(vecs, 30))
        vecs_df.to_csv(
            "./data/vectorized-smol.csv", header=None, index=False, mode=mode
        )
        n_done += CHUNK_SIZE
        mode = "a"
        print(f"{n_done} ({max(get_lens(vecs))} tokens max)...")
