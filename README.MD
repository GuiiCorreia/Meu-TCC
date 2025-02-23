## 📊 Projeto de Clusterização de Dados

Este repositório contém o código e os dados utilizados no Trabalho de Conclusão de Curso (TCC) sobre **técnicas de clusterização**. O objetivo do projeto é analisar diferentes algoritmos de agrupamento e avaliar seus desempenhos com base no **Silhouette Score**.

### 💁️️ Estrutura do Repositório

```
📂 Meu-Repositorio
│-- 📄 README.md
│-- 📄 requirements.txt
│-- 📄 gerar_testes.py
│-- 📄 execucao_algoritmos.py
│-- 📄 dados_completos.csv
│-- 📄 dados_teste.csv
│-- 📄 resultados.csv
```

### 🚀 Tecnologias Utilizadas

- **Python** 🐍
- **Pandas** (para manipulação de dados)
- **Scikit-learn** (para algoritmos de clustering e métricas)
- **HDBSCAN** (para clusterização baseada em densidade)
- **Matplotlib** (para visualização de resultados)

### 👅 Instalação

Para executar o projeto, siga os passos abaixo:

1. Clone este repositório:

   ```bash
   git clone https://github.com/seu-usuario/seu-repositorio.git
   cd seu-repositorio
   ```

2. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

### 📊 Execução do Projeto

#### 1️⃣ Gerar Amostra de Teste

Como a clusterização demora, criei um arquivo que gera dados de teste, o arquivo `gerar_testes.py` seleciona uma amostra de 5.000 registros do conjunto de dados original (`dados_completos.csv`) e gera o arquivo `dados_teste.csv`. Fique a vontade para usar caso não queira esperar a clusterização a partir dos dados completos. Para executar:

```bash
python gerar_testes.py
```

#### 2️⃣ Executar Algoritmos de Clusterização

O script `execucao_algoritmos.py` executa diferentes algoritmos de clusterização e calcula o **Silhouette Score**. Para rodá-lo:

```bash
python execucao_algoritmos.py
```

Os resultados serão salvos no arquivo `resultados.csv`.

### 📊 Algoritmos Utilizados

Os seguintes algoritmos de clusterização foram testados:

1. **K-Means**
2. **MiniBatch K-Means**
3. **DBSCAN**
4. **HDBSCAN**

Cada algoritmo é avaliado pelo **Silhouette Score**, uma métrica que mede a qualidade do agrupamento.

### 🐝 Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para enviar um *pull request* ou abrir uma *issue* para sugestões.

### 📝 Licença

Este projeto está sob a licença **MIT**. Sinta-se à vontade para contribuir e modificar!

---

Caso precise de ajustes ou queira adicionar mais detalhes, me avise! 😊🚀

