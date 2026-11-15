# Customer Churn Prediction — Databricks + GitLab

Projeto de portfólio para demonstrar ingestão de dados, transformação, Machine Learning,
rastreamento de experimentos com MLflow e automação com GitLab CI/CD.

## Dataset
Iranian Churn Dataset (UCI): https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset
CSV: https://archive.ics.uci.edu/static/public/563/data.csv
Licença informada pelo UCI: CC BY 4.0.

O conjunto contém dados históricos de clientes de telecomunicações. É adequado para estudo,
não deve ser tratado como retrato do mercado atual. `customer_id` é um identificador técnico
criado pelo projeto, não um identificador original. A coluna `status` é excluída das features
para reduzir risco de vazamento de informação.

## Tabelas
- `workspace.churn_demo.bronze_customers`
- `workspace.churn_demo.silver_customers`
- `workspace.churn_demo.gold_churn_predictions`

## Execução
O pipeline está preparado para execução como tarefa Python em Databricks Jobs. No Free Edition,
confirme que o workspace permite Jobs serverless e acesso de rede ao CSV do UCI. Se a rede
não estiver disponível no compute, faça upload do CSV para um Volume do Unity Catalog e
altere `UCI_DATA_URL` em `src/churn_project/pipeline.py`.

## Desenvolvimento local
```bash
python -m pip install -r requirements-dev.txt
pytest -v
```

## Databricks Asset Bundles
Instale a Databricks CLI e autentique-se antes de executar:
```bash
databricks bundle validate -t dev
databricks bundle deploy -t dev
databricks bundle run churn_pipeline -t dev
```

O deploy publica/atualiza o Job; o comando `bundle run` executa-o explicitamente.
