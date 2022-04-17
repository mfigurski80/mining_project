import pandas as pd
from numpy.linalg import norm
import itertools
import gensim.downloader
import csv

from read import (
    apply_transformation_to_token_dfs,
)
from word_tfidf import read_translation_dictionary, read_tfidf, generate_bow_with
from vectorize import vectorize


def vectorize(toks_list, dictionary, tfidf_model, w2vec_model):

    print("performing tweet-vectorization")

    for bow in generate_bow_with(toks_list, dictionary):
        #  print(vecs)
        toks = [dictionary[w[0]] for w in bow]
        weights = tfidf_model[bow]
        weights = [w[1] for w in weights]
        #  print(toks)
        #  print(weights)

        vecs = [
            list(w2vec_model[tok] * w)
            for (tok, w) in zip(toks, weights)
            if tok in w2vec_model
        ]
        vec = [sum(i) for i in zip(*vecs)]
        vec = vec / norm(vec)
        yield vec


def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


if __name__ == "__main__":
    FNAME = "./data/smol.csv"
    OUT_FNAME = "./data/tweet-vectorized-smol.csv"
    #  OUT_FNAME = "./test.csv"
    #  tok_list_gen = lambda: read_tokens_rows(FNAME)
    d = read_translation_dictionary("./models/dictionary.model")
    tfidf = read_tfidf("./models/word-tfidf.model")
    w2v = gensim.downloader.load("glove-twitter-25")

    def transform(toks):
        #  print(df)
        res = pd.DataFrame(vectorize(toks, d, tfidf, w2v))
        #  print(res)
        return res

    apply_transformation_to_token_dfs(FNAME, transform, OUT_FNAME)

    #  tweet_vectorized_gen = vectorize(
    #  #  wvecs_gen(),
    #  tok_list_gen(),
    #  read_translation_dictionary("./models/dictionary.model"),
    #  read_tfidf("./models/word-tfidf.model"),
    #  )
    #  #  print(tweet_vectorized)
#
#  # write dem out
#  with open(OUT_FNAME, "w") as f:
#  wr = csv.writer(f)
#  wr.writerows(tweet_vectorized_gen)
