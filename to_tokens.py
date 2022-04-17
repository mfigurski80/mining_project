import string

STOPLIST = set("for a and of the in to so".split())


def tokenize(l: list) -> list:
    """Takes list of tweets, returns list of token
    lists: [[str]]. Super useful because they
    get cleaned and are more ready for processing
    """
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
