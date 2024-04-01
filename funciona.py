import pandas as pd
import time
from sklearn.cluster import KMeans, SpectralClustering, MiniBatchKMeans, DBSCAN
from sklearn.metrics import silhouette_score
from hdbscan import HDBSCAN
import warnings

warnings.filterwarnings('ignore')

def cluster_and_evaluate(algorithm, data, **kwargs):
    model = algorithm(**kwargs)
    labels = model.fit_predict(data)
    if len(set(labels)) > 1:
        score = silhouette_score(data, labels, metric='euclidean')
    else:
        score = None
    return score

data = pd.read_csv('dados_modificados.csv')

algorithms = [
    (KMeans, {"n_init": 10, "random_state": 0, "n_clusters": 8}),  # Exemplo de n_clusters para KMeans
    (MiniBatchKMeans, {"n_init": 10, "random_state": 0, "n_clusters": 8}),  # MiniBatchKMeans com parâmetros similares ao KMeans
    (DBSCAN, {"eps": 0.5, "min_samples": 5}),  # Parâmetros de DBSCAN
    (HDBSCAN, {"min_cluster_size": 5, "min_samples": 5})  # Parâmetros de HDBSCAN
]

results = []

total_start_time = time.time()

for algo, params in algorithms:
    algo_name = algo.__name__
    start_time = time.time()
    
    score = cluster_and_evaluate(algo, data, **params)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    results.append({
        "Algoritmo": algo_name,
        "Silhouette": score,
        "Tempo (s)": elapsed_time
    })

    print(f"Algoritmo: {algo_name}, Silhouette: {score}, Tempo: {elapsed_time} segundos")

total_end_time = time.time()
print(f"Tempo total de execução: {total_end_time - total_start_time} segundos")

results_df = pd.DataFrame(results)
results_df.to_csv('resultados_clusterizacao.csv', index=False, float_format="%.10f")
