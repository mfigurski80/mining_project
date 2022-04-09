import gensim.downloader
import pandas as pd
import string

STOPLIST = set("for a and of the in to so".split())
CHUNK_SIZE = 1000


def tokenize(l):
    trans = str.maketrans("", "", string.punctuation)
    return [
        [w.translate(trans) for w in tweet.lower().split() if w not in STOPLIST]
        for tweet in l
    ]
    # TODO: implement bi-gram detector model


def vectorize(l):
    model = gensim.downloader.load("glove-twitter-25")
    #  print(model)
    return [([list(model[token]) for token in tweet if token in model] + [None]*30)[:30] for tweet in l]


if __name__ == "__main__":
    n_done = 0
    df_iter = pd.read_csv(
        "./data/smol.csv", header=None, chunksize=CHUNK_SIZE, encoding="ISO-8859-1"
    )
    mode = "w"
    for df in df_iter:
        vecs = pd.DataFrame(vectorize(tokenize(df[5])))
        vecs.to_csv("./data/vectorized-smol.csv", header=None, index=False, mode=mode)
        n_done += CHUNK_SIZE
        mode = "a"
        print(f"{n_done}...")
