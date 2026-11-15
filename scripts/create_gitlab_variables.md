No GitLab, abra Settings > CI/CD > Variables e crie:
- `DATABRICKS_HOST`: URL do workspace Databricks
- `DATABRICKS_CLIENT_ID`: OAuth service principal client ID
- `DATABRICKS_CLIENT_SECRET`: OAuth service principal secret

Marque o segredo como Masked e Protected quando aplicável. Não coloque credenciais no código.
A autenticação OAuth M2M e a criação de Service Principal dependem dos recursos/permissões
disponíveis no workspace; no Free Edition, valide essa disponibilidade antes de habilitar deploy.
