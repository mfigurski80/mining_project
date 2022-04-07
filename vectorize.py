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


if __name__ == "__main__":
    df_iter = pd.read_csv(
        "./data/smol.csv", header=None, chunksize=100, encoding="ISO-8859-1"
    )
    for df in df_iter:
        print(tokenize(df[5]))
        break
