import pandas as pd
import time
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, Birch, MiniBatchKMeans
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings('ignore')

def cluster_and_evaluate(algorithm, data, n_clusters, **kwargs):
    model = algorithm(n_clusters=n_clusters, **kwargs)
    labels = model.fit_predict(data)
    if len(set(labels)) > 1:
        score = silhouette_score(data, labels, metric='euclidean')
    else:
        score = None
    return score

data = pd.read_csv('dados_1000.csv')

# Substituído SpectralClustering por MiniBatchKMeans na lista de algoritmos
algorithms = [
    (KMeans, {"n_init": 10, "random_state": 0}),
    (MiniBatchKMeans, {"n_init": 10, "random_state": 0}),  # MiniBatchKMeans com parâmetros similares ao KMeans
    (AgglomerativeClustering, {"affinity": "euclidean", "linkage": "ward"}),
    (Birch, {})
]

results = []

total_start_time = time.time()

for algo, params in algorithms:
    algo_name = algo.__name__
    for n_clusters in range(3, 16):
        start_time = time.time()
        
        score = cluster_and_evaluate(algo, data, n_clusters, **params)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        results.append({
            "Algoritmo": algo_name,
            "Nº de Clusters": n_clusters,
            "Silhouette": score,
            "Tempo (s)": elapsed_time
        })

        print(f"Algoritmo: {algo_name}, Nº de Clusters: {n_clusters}, Silhouette: {score}, Tempo: {elapsed_time} segundos")

total_end_time = time.time()
print(f"Tempo total de execução: {total_end_time - total_start_time} segundos")

results_df = pd.DataFrame(results)
results_df.to_csv('resultados_clusterizacao.csv', index=False, float_format="%.10f")
