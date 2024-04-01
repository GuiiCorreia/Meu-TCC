import pandas as pd

# Carregar o arquivo de dados
df = pd.read_csv('./dados_enade_limpos.csv')

# Removendo as colunas 'NU_ANO' e 'CO_CURSO'
df = df.drop(columns=['NU_ANO', 'CO_CURSO'])

# Salvando o DataFrame modificado de volta para um arquivo CSV, se necessário
df.sample(1000)
df.to_csv('./dados_1000.csv', index=False)