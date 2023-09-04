import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv("data/books.csv", usecols = ["title", "average_rating", "num_pages"])

df["average_rating"] = pd.to_numeric(df["average_rating"], errors='coerce')
df["num_pages"] = pd.to_numeric(df["num_pages"], errors='coerce')
df = df.dropna()
data = df[["average_rating", "num_pages"]].values
inertias = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

plt.plot(range(1,11), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

kmeans = KMeans(n_clusters=4)
kmeans.fit(data)

plt.scatter(df["average_rating"], df["num_pages"], c=kmeans.labels_)
plt.show()
