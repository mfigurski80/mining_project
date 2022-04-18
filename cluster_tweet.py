import pandas as pd
from sklearn.cluster import KMeans
from joblib import dump, load


def train_kmeans(df, n, save_name=None):
    kmeans = KMeans(n)
    kmeans.fit(df)
    if save_name:
        dump(kmeans, save_name)
    return kmeans


def load_kmeans(save_name):
    return load(save_name)


def print_inertia(df, to=20):
    wcss = []
    for i in range(1, 20):
        m = train_kmeans(df, i)
        wcss_iter = m.inertia_
        wcss.append(wcss_iter)
        print(f"{i} : {round(wcss_iter,2)}")
    return wcss


def save_scatterplot(df, a, b):
    return (
        df.plot.scatter(a, b, c="cluster", cmap="rainbow")
        .get_figure()
        .savefig(f"images/c-{a}-{b}.png")
    )


if __name__ == "__main__":
    vecs_df = pd.read_csv(
        "./data/tweet-vectorized-smol.csv",
        header=None,
    ).fillna(0)
    df = pd.read_csv("./data/smol.csv", header=None, encoding="ISO-8859-1")
    #  print(df.corr())
    #  print_inertia(df)

    #  kmeans = train_kmeans(df, 10, "./models/tweet-10-kmeans.model")
    kmeans = load_kmeans("./models/tweet-10-kmeans.model")
    clusters = kmeans.fit_predict(vecs_df)
    del vecs_df
    df["cluster"] = clusters
    df = df[[0, 5, "cluster"]]
    df = df.sort_values("cluster")
    print(df)
    df.to_csv("./data/clustered.csv", header=None)
