import pandas as pd


data = pd.read_csv('dados_completos.csv')
sample_data = data.sample(n=5000, random_state=42)
sample_data.to_csv('dados_teste.csv', index=False)
