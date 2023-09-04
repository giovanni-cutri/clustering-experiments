import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df_movies = pd.read_csv("data/movies_metadata.csv", usecols = ["id", "original_title", "revenue"])
df_rating = pd.read_csv("data/ratings_small.csv", usecols = ["movieId", "rating"])

df_rating.rename(columns = {"movieId": "id"}, inplace=True)
df_rating["id"] = df_rating["id"].astype(str)
df_average_ratings = df_rating.groupby("id").mean()

df = df_movies.merge(df_average_ratings, on = "id").dropna()

data = df[["rating", "revenue"]].values
inertias = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init=auto)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

plt.plot(range(1,11), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

kmeans = KMeans(n_clusters=3)
kmeans.fit(data)

plt.scatter(df["rating"], df["revenue"], c=kmeans.labels_)
plt.show()