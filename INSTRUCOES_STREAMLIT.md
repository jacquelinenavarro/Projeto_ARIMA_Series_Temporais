# 📊 Aplicativo Streamlit - Modelagem ARIMA From Scratch

## 🚀 Como Executar

### 1. Ativar o Ambiente Virtual

No PowerShell:
```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Executar o Aplicativo Streamlit

```powershell
streamlit run app_arima_streamlit.py
```

O aplicativo será aberto automaticamente no navegador em `http://localhost:8501`

## 📋 Funcionalidades

### Seção 1: Carregamento de Dados
- Upload de arquivo CSV via interface web
- Visualização prévia dos dados
- Seleção da coluna da série temporal

### Seção 2: Análise Exploratória
- Visualização da série original
- Histograma da distribuição
- Transformação logarítmica (se aplicável)
- Gráficos de FAC e FACP

### Seção 3: Modelagem ARIMA
- Botão para iniciar o Grid Search
- Processamento de todos os modelos ARIMA(p,d,q) com p,q ∈ [0,3] e d ∈ [1,2]
- Exibição dos Top 5 modelos candidatos

### Seção 4: Resultados do Modelo Vencedor
- Métricas estatísticas (AIC, BIC, P-valores)
- Análise dos 5 critérios estatísticos
- Parâmetros do modelo (phi e theta)
- Gráficos de diagnóstico dos resíduos

### Seção 5: Relatório de Inferência
- Relatório acadêmico completo em texto
- Interpretação estatística dos resultados

### Seção 6: Validação de Acurácia
- Métricas RMSE e MAPE
- Gráfico comparativo: Real vs Ajustado

## ⚙️ Requisitos

Todas as bibliotecas necessárias estão no arquivo `requirements.txt`:
- numpy
- pandas
- matplotlib
- scipy
- streamlit

## 📝 Notas Importantes

- **Processamento Demorado**: O Grid Search pode levar alguns minutos dependendo do tamanho da série


## 🔧 Solução de Problemas

Se encontrar erros:
1. Certifique-se de que o ambiente virtual está ativado
2. Verifique se todas as bibliotecas foram instaladas: `pip install -r requirements.txt`
3. Certifique-se de que o arquivo CSV está no formato correto (coluna numérica válida)
