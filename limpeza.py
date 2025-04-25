import pandas as pd

# Carregar o arquivo de dados
df = pd.read_csv('./dados_enade.csv')

# Removendo as colunas 'NU_ANO'
df = df.drop(columns=['NU_ANO'])

# Salvando o DataFrame da amostra em um novo arquivo CSV
df.to_csv('./dados_enade_limpos.csv', index=False)
