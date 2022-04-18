import pandas as pd

from read import apply_transformation_to_token_dfs
from word_tfidf import read_translation_dictionary, generate_bow_with, read_tfidf

if __name__ == "__main__":
    FNAME = "./data/smol.csv"
    OUT_FNAME = "./data/tweet-keyword-smol.csv"
    d = read_translation_dictionary("./models/dictionary.model")
    tf_model = read_tfidf("./models/word-tfidf.model")

    def transform(toks_list, df):
        corpus = generate_bow_with(toks_list, d)
        tf_idf = tf_model[corpus]

        keywords = [
            d[max(weights, key=lambda w: w[1])[0]] if (len(weights) > 0) else None
            for weights in tf_idf
        ]
        #  for weights in tf_idf:
        #  if len(weights) == 0:
        #  continue
        #  best = max(weights, key=lambda w: w[1])
        #
        #  print(d[best[0]])
        df["keywords"] = keywords
        df = df[[5, "keywords"]]
        print(df)

        return df

    apply_transformation_to_token_dfs(FNAME, transform, OUT_FNAME)
