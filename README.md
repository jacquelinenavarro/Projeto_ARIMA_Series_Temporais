# Implementação do Modelo ARIMA
### Uma Abordagem Algébrica para Previsão de Séries Temporais

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Status](https://img.shields.io/badge/Status-Academic%20Project-success)
![License](https://img.shields.io/badge/Methodology-Box%20%26%20Jenkins-orange)

---

## 📄 Resumo

Este projeto apresenta o desenvolvimento de um sistema computacional em **Python** para modelagem e previsão de séries temporais estocásticas. Diferente das abordagens convencionais que utilizam bibliotecas prontas (como statsmodels), esta implementação prioriza a exigência da Disciplina de Séries Temporais, reproduzindo manualmente as etapas de identificação, estimação e diagnóstico conforme os fundamentos teóricos de **Box & Jenkins** (1976). 

O sistema automatiza a seleção do modelo **ARIMA $(p,d,q)$** ótimo por meio de testes estatísticos e critérios de informação.

---

## 1. Introdução e Objetivos

A modelagem de séries temporais é essencial para a tomada de decisão baseada em dados. O objetivo deste trabalho foi desenvolver uma ferramenta capaz de:

- Processar dados brutos e transformá-los em séries estacionárias.  
- Ajustar modelos ARIMA sem depender de otimizadores numéricos externos.  

---

## 2. Metodologia:

O código foi estruturado seguindo o ciclo iterativo clássico de Box & Jenkins, dividido em blocos lógicos de processamento.

**Identificação -> Estimação -> Diagnóstico**

### 2.1. Pré-processamento e Estacionariedade
- **Estabilização da Variância:** Aplicação da transformação logarítmica ($\ln(Z_t)$) para linearizar tendências exponenciais e reduzir a volatilidade.  
- **Estacionariedade:** Implementação de um operador de diferenças recursivo ($\nabla^d$), capaz de calcular primeira ($d=1$) e segunda ($d=2$) diferenças, removendo tendências estocásticas e garantindo a condição necessária para modelagem ARIMA.

### 2.2. Identificação (Cálculo Manual das Correlações)
- **FAC (Função de Autocorrelação):** Calculada via covariância normalizada para identificar a ordem do processo MA.  
- **FACP (Função de Autocorrelação Parcial):** Implementada pelo **Algoritmo de Durbin-Levinson**, permitindo calcular os coeficientes $\phi_{kk}$ de forma eficiente e identificar a ordem do processo AR sem recorrer a bibliotecas estatísticas prontas.

### 2.3. Estimação dos Parâmetros (Abordagem Algébrica)
- **Componente AR:** Estimativa dos coeficientes $\phi$ via **Equações de Yule-Walker**, resolvendo sistemas lineares com matriz de Toeplitz das autocorrelações.  
- **Componente MA:** Estimativa dos coeficientes $\theta$ pelo **Método dos Momentos**, invertendo a estrutura da FAC e aproximando os parâmetros com base nas propriedades teóricas de processos MA puros.

---

## 3. Seleção de Modelos e Diagnóstico

O algoritmo de *Grid Search* testa múltiplas combinações de $(p,d,q)$ e seleciona o modelo vencedor com base em critérios estatísticos e informacionais.

### Critérios de Diagnóstico Aplicados
1. **AIC e BIC:** Penalizam a complexidade do modelo, evitando *overfitting* e privilegiando a parcimônia.  
2. **Análise Visual:** Gráficos dos resíduos e suas correlações.  
3. **Teste de Box-Pierce:** Verifica a aleatoriedade global dos resíduos.  
4. **Teste de Ljung-Box:** Versão para amostras finitas, avaliando ausência de correlação serial.  
5. **Periodograma Acumulado (Teste Espectral):** Implementação do teste de Kolmogorov-Smirnov no domínio da frequência, garantindo ausência de sazonalidades ocultas.

> **Critério de Decisão:** Um modelo só é considerado válido se seus resíduos se comportarem como **Ruído Branco**. Entre os modelos aprovados, vence aquele com menor AIC.

---

## 4. Validação e Resultados

Na etapa final, o modelo é validado frente aos dados reais (amostra de teste). O sistema reconstrói a série original (revertendo diferenciação e log) e calcula métricas de acurácia:

- **RMSE (Raiz do Erro Quadrático Médio):** Mede a magnitude do erro na mesma unidade dos dados.  
- **MAPE (Erro Percentual Absoluto Médio):** Avalia o erro relativo em termos percentuais.  

---

## 5. Conclusão

O desenvolvimento realizado demonstra que é possível construir previsões de séries temporais utilizando apenas os fundamentos de álgebra linear e estatística, atendendo às exigências da disciplina "Série Temporais" de implementar os métodos sem recorrer a bibliotecas de modelagem prontas.  

Dessa forma, o modelo ARIMA selecionado não se limita a um ajuste numérico, mas representa uma construção estatisticamente fundamentada e validada, em conformidade com os critérios acadêmicos da disciplina. O ciclo Box & Jenkins foi seguido, assegurando que cada decisão sobre ordens, parâmetros e diagnósticos esteja alinhada às práticas teóricas exigidas.


## 🛠️ Ferramentas e Apoio ao Desenvolvimento

Este projeto foi construído com o auxílio de um conjunto de ferramentas que facilitaram a organização do código, a documentação dos procedimentos e a análise dos resultados. O foco esteve em garantir clareza, reprodutibilidade e qualidade na implementação, de forma alinhada às exigências acadêmicas da disciplina "Séries Temporais."

| Categoria | Ferramenta | Função |
| :--- | :--- | :--- |
| **Ambiente de Desenvolvimento** | **Google Colab / VS Code** | Prototipagem matemática e testes (Colab); desenvolvimento da interface web `Streamlit` (VS Code). |
| **Controle de Versão** | **GitHub** | Versionamento do código e publicação da documentação do projeto. |
| **Assistência de Código** | **Ferramentas de LLM** | Apoio na depuração de erros, otimização da complexidade e geração de *docstrings*. |
| **Recursos Visuais** | **YouTube** | Hospedagem da vídeo-demonstração do protótipo e apresentação do projeto. |

---

## 📌 Contribuição

Este é um projeto acadêmico de código aberto e contribuições são muito bem-vindas! 

**Como contribuir:**
1.  **Reportar Bugs:** Encontrou alguma inconsistência nos cálculos? Abra uma [Issue](https://github.com/seu-usuario/seu-repositorio/issues).
2.  **Sugestões de Código:** Sinta-se à vontade para enviar *Pull Requests* com otimizações para o projeto.
3.  **Novos Datasets:** Tem uma série temporal interessante? Adicione-a à pasta `/exemplos` para enriquecer os testes.

---
