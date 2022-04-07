import gensim.downloader
import pandas as pd
import string

stoplist = set("for a and of the in to so".split())


def tokenize(l):
    trans = str.maketrans("", "", string.punctuation)
    return [
        [w.translate(trans) for w in tweet.lower().split() if w not in stoplist]
        for tweet in l
    ]
    # TODO: implement bi-gram detector model


def vectorize(l):
    model = gensim.downloader.load("glove-twitter-25")
    #  print(model)
    return [[list(model[token]) for token in tweet if token in model] for tweet in l]


if __name__ == "__main__":
    df_iter = pd.read_csv(
        "./data/smol.csv", header=None, chunksize=1000, encoding="ISO-8859-1"
    )
    mode = "w"
    for df in df_iter:
        vecs = pd.DataFrame(vectorize(tokenize(df[5])))
        vecs.to_csv("./data/vectorized-smol.csv", header=None, index=False, mode=mode)
        print("...")
        mode = "a"
