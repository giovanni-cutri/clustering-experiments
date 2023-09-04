import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df_gdp = pd.read_csv("data/gdp.csv", skiprows = 4, usecols = ["Country Name", "Country Code", "2022"])
df_population = pd.read_csv("data/population.csv", skiprows = 4, usecols = ["Country Code", "2022"])

df_gdp.rename(columns = {"2022": "gdp_2022"}, inplace=True)
df_population.rename(columns = {"2022": "population_2022"}, inplace=True)

df = df_gdp.merge(df_population, on = "Country Code").dropna()

X = df[["gdp_2022", "population_2022"]].values
inertias = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.plot(range(1,11), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

plt.scatter(df["gdp_2022"], df["population_2022"], c=kmeans.labels_)
plt.show()