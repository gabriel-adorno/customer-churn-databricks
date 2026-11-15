# Customer Churn Prediction

Projeto pessoal de análise e previsão de cancelamento de clientes (*Customer Churn*) utilizando **Databricks, Python, PySpark e Machine Learning**.

A ideia do projeto é trabalhar com dados reais de uma empresa de telecomunicações, explorar os dados, identificar padrões de cancelamento e desenvolver modelos capazes de prever quais clientes apresentam maior probabilidade de abandonar o serviço.

## Tecnologias utilizadas

- **Databricks:** ambiente de desenvolvimento, processamento de dados e execução do pipeline.
- **GitLab:** versionamento e organização do código-fonte.
- **Python:** desenvolvimento da lógica do projeto.
- **PySpark:** transformação e processamento dos dados.
- **Pandas:** manipulação dos dados.
- **Scikit-learn:** treinamento e avaliação dos modelos de Machine Learning.
- **MLflow:** acompanhamento e comparação dos experimentos.

## Dados utilizados

O projeto utiliza o Iranian Churn Dataset, disponibilizado pela UCI Machine Learning Repository.

- [Página do dataset](https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset)
- [Arquivo CSV](https://archive.ics.uci.edu/static/public/563/data.csv)

O conjunto contém informações históricas de clientes de telecomunicações e aproximadamente 3.150 registros.

## O que foi desenvolvido

- Ingestão dos dados originais.
- Organização dos dados em uma arquitetura Medallion.
- Criação das camadas Bronze, Silver e Gold.
- Limpeza e preparação dos dados para análise.
- Construção de um pipeline de Machine Learning.
- Treinamento e comparação de modelos.
- Registro de métricas e experimentos com MLflow.
- Geração de previsões e classificação de clientes por nível de risco de cancelamento.
- Criação de um dashboard no Databricks para visualizar os resultados.

## Arquitetura dos dados

O projeto utiliza a arquitetura Medallion, dividida em três camadas:

**Bronze:** armazena os dados inicialmente ingeridos.

**Silver:** contém os dados tratados e preparados para utilização.

**Gold:** armazena os resultados das previsões realizadas pelo modelo, incluindo a probabilidade de churn e o nível de risco de cada cliente.

Tabelas criadas no Unity Catalog:

- `churn_demo.churn_demo.bronze_customers`
- `churn_demo.churn_demo.silver_customers`
- `churn_demo.churn_demo.gold_churn_predictions`

## Machine Learning

Foram desenvolvidos e comparados dois modelos:

- Logistic Regression
- Random Forest

Os experimentos foram acompanhados pelo MLflow, permitindo comparar métricas como Accuracy, Precision, Recall, F1-score e ROC-AUC.

O Random Forest apresentou o melhor resultado geral considerando o F1-score.

## Dashboard

Também foi desenvolvido um dashboard no Databricks para visualizar os resultados do projeto, incluindo indicadores de clientes, distribuição por nível de risco e probabilidades de cancelamento.

## Organização do projeto

O código-fonte está versionado no GitLab, enquanto o processamento e a execução do pipeline são realizados no Databricks.

Este projeto foi desenvolvido como estudo prático de Engenharia de Dados e Machine Learning, com foco na integração dessas tecnologias em um cenário próximo de um problema real de negócio.
