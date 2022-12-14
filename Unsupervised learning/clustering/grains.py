import pandas as pd
# Import KMeans
from sklearn.cluster import KMeans
# Import pyplot
from matplotlib import pyplot as plt

df = pd.read_csv('Unsupervised learning/clustering/grains.csv')
samples = df.iloc[:, 0:-1]
species = df.iloc[:, -1]
ks = range(1, 6)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)

    # Fit model to samples
    model.fit(samples)

    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)

# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()
