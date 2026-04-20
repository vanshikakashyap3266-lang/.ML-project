import pandas as pd
from sklearn.cluster import KMeans

# Dataset path
path = r"/content/heart.csv"   

# Load dataset
df = pd.read_csv(path)


X = df[['trestbps']]   

model = KMeans(n_clusters=2, random_state=0)


model.fit(X)


print(model.labels_)
