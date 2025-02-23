import pandas as pd
import time
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
from sklearn.metrics import silhouette_score
import hdbscan
import warnings

# Ignorar avisos desnecessários
warnings.filterwarnings('ignore')

def cluster_and_evaluate(algorithm, data, n_clusters=None, **kwargs):
    """
    Executa o algoritmo de clustering e calcula o Silhouette Score se houver mais de um cluster válido.
    """
    # Inicializa o modelo de clusterização
    if n_clusters is not None:
        model = algorithm(n_clusters=n_clusters, **kwargs)
    else:
        model = algorithm(**kwargs)
    
    # Realiza o agrupamento
    labels = model.fit_predict(data)
    
    # Calcula o Silhouette Score se houver mais de um cluster válido
    if len(set(labels)) > 1:
        score = silhouette_score(data, labels, metric='euclidean')
    else:
        score = None
    
    return score

# Carregar os dados a serem clusterizados
data = pd.read_csv('dados_teste.csv')

# Lista de algoritmos de clustering a serem testados
algorithms = [
    (KMeans, {"n_init": 10, "random_state": 0}),
    (MiniBatchKMeans, {"n_init": 10, "random_state": 0}),
    (DBSCAN, {"eps": 0.5, "min_samples": 5}),
    (hdbscan.HDBSCAN, {"min_cluster_size": 5})
]

# Lista para armazenar os resultados
results = []

total_start_time = time.time()  # Inicia a contagem do tempo total

# Loop pelos algoritmos
for algo, params in algorithms:
    algo_name = algo.__name__
    
    # Definir intervalo de clusters apenas para KMeans e MiniBatchKMeans
    if algo in [KMeans, MiniBatchKMeans]:
        cluster_range = range(3, 16)
    else:
        cluster_range = [None]  # DBSCAN e HDBSCAN determinam os clusters automaticamente
    
    # Loop pelos números de clusters
    for n_clusters in cluster_range:
        start_time = time.time()  # Inicia a contagem do tempo de execução
        
        # Executa o algoritmo e avalia
        score = cluster_and_evaluate(algo, data, n_clusters, **params)
        
        end_time = time.time()  # Fim do tempo de execução
        elapsed_time = end_time - start_time  # Calcula o tempo decorrido
        
        # Armazena os resultados
        results.append({
            "Algoritmo": algo_name,
            "Nº de Clusters": n_clusters or 'Auto',
            "Silhouette": score,
            "Tempo (s)": elapsed_time
        })
        
        # Exibe o progresso no console
        print(f"Algoritmo: {algo_name}, Nº de Clusters: {n_clusters or 'Auto'}, Silhouette: {score}, Tempo: {elapsed_time} segundos")

total_end_time = time.time()
print(f"Tempo total de execução: {total_end_time - total_start_time} segundos")

# Salvar os resultados em um arquivo CSV
results_df = pd.DataFrame(results)
results_df.to_csv('resultados.csv', index=False, float_format="%.10f")
