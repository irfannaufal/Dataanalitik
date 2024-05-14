import seaborn as sns
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df = pd.read_excel('Data.xlsx')
df1 = df[(df['Asuransi Fob'] != 0) | (df['Freight Fob'] != 0)]
#df1
dft=df1.head(1000)
df2 = df[(df['Netto Peb'] != 0)]
df2
X = dft[['Freight Fob', 'Nilai Dev Usd Brg']]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(X_scaled)

# Visualize the clustering
plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=dbscan.labels_, cmap='viridis', marker='o', s=50)
plt.title('DBSCAN Clustering ')
plt.xlabel('Netto (Standardized)')
plt.ylabel('Dev (Standardized)')
plt.colorbar(label='Cluster Label')
plt.show()
