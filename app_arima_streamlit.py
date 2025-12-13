# -*- coding: utf-8 -*-
"""
Aplicativo Streamlit para Modelagem ARIMA "From Scratch"
Transformado do código original do Google Colab
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2, norm
import warnings
import re
import textwrap

# Configurações
warnings.filterwarnings('ignore')
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        plt.style.use('default')

# ==============================================================================
# BLOCO 1: BIBLIOTECAS E FUNÇÕES MATEMÁTICAS (NÚCLEO DO SISTEMA)
# ==============================================================================
# NOTA: Este bloco permanece INTOCÁVEL conforme exigência acadêmica

# ------------------------------------------------------------------------------
# 1. FUNÇÕES DE TRANSFORMAÇÃO
# ------------------------------------------------------------------------------
def aplicar_log(serie):
    """
    Aplica transformação Logarítmica (ln) para estabilizar variância.
    Retorna: (série_transformada, sucesso_bool)
    """
    if np.any(serie <= 0):
        return serie, False
    return np.log(serie), True

def diferenciar_serie(serie, ordem=1):
    """
    Calcula diferenças sucessivas para estacionarizar a série.
    Retorna: série diferenciada
    """
    diff = np.array(serie)
    for _ in range(ordem):
        diff = np.diff(diff)
    return diff

# ------------------------------------------------------------------------------
# 2. FUNÇÕES DE AUTOCORRELAÇÃO (MANUAL)
# ------------------------------------------------------------------------------
def calcular_fac_manual(serie, lags=20):
    """
    Calcula Função de Autocorrelação (FAC) manualmente.
    Retorna: Array de correlações até o lag k.
    """
    n = len(serie)
    media = np.mean(serie)
    var = np.var(serie)
    acf = np.zeros(lags + 1)
    acf[0] = 1.0

    if var == 0: return acf

    for k in range(1, lags + 1):
        cov = np.sum((serie[:-k] - media) * (serie[k:] - media)) / n
        acf[k] = cov / var
    return acf

def calcular_facp_manual(serie, lags=20):
    """
    Calcula Função de Autocorrelação Parcial (FACP) usando Durbin-Levinson.
    Retorna: Array de correlações parciais.
    """
    acf = calcular_fac_manual(serie, lags=lags)
    pacf = np.zeros(lags + 1)
    pacf[0] = 1.0

    if lags >= 1: pacf[1] = acf[1]

    # Matriz phi para guardar coeficientes da recursão
    phi = np.zeros((lags + 1, lags + 1))
    phi[1, 1] = acf[1]

    for k in range(2, lags + 1):
        # Recursão de Durbin-Levinson
        num = acf[k] - np.sum(phi[k-1, 1:k] * acf[k-1:0:-1])
        den = 1 - np.sum(phi[k-1, 1:k] * acf[1:k])

        phi[k, k] = num / den if abs(den) > 1e-10 else 0
        pacf[k] = phi[k, k]

        for j in range(1, k):
            phi[k, j] = phi[k-1, j] - phi[k, k] * phi[k-1, k-j]

    return pacf

# ------------------------------------------------------------------------------
# 3. ESTIMAÇÃO DE PARÂMETROS
# ------------------------------------------------------------------------------
def estimar_ar_yule_walker(serie, p):
    """
    Estima coeficientes AR(p) resolvendo equações de Yule-Walker.
    Retorna: (phi, sigma2)
    """
    if p == 0: return np.array([]), np.var(serie)

    acf = calcular_fac_manual(serie, lags=p)
    R = np.zeros((p, p)) # Matriz de Toeplitz
    for i in range(p):
        for j in range(p):
            R[i, j] = acf[abs(i - j)]

    r = acf[1:p+1]
    try:
        phi = np.linalg.solve(R, r)
        sigma2 = np.var(serie) * (1 - np.dot(r, phi))
        return phi, sigma2
    except:
        return np.zeros(p), np.var(serie)

def estimar_ma_momentos(serie, q):
    """
    Estima coeficientes MA(q) via Método dos Momentos (Inversão da FAC).
    Retorna: (theta, sigma2)
    """
    if q == 0: return np.array([]), np.var(serie)
    acf = calcular_fac_manual(serie, lags=q)
    theta = -acf[1:q+1] # Aproximação simples
    sigma2 = np.var(serie) / (1 + np.sum(theta**2))
    return theta, sigma2

# ------------------------------------------------------------------------------
# 4. TESTES DE DIAGNÓSTICO
# ------------------------------------------------------------------------------
def teste_box_pierce_ljung_box(residuos, n_params, lags=20):
    """
    Testes de Portmanteau para Ruído Branco.
    Retorna: ((Q_bp, p_bp), (Q_lb, p_lb))
    """
    n = len(residuos)
    acf = calcular_fac_manual(residuos, lags=lags)
    acf_sq = acf[1:] ** 2

    # Box-Pierce
    Q_bp = n * np.sum(acf_sq)

    # Ljung-Box (Ponderado)
    pesos = np.array([(n + 2) / (n - k) for k in range(1, lags + 1)])
    Q_lb = n * np.sum(pesos * acf_sq)

    df = max(1, lags - n_params)
    p_bp = 1 - chi2.cdf(Q_bp, df)
    p_lb = 1 - chi2.cdf(Q_lb, df)

    return (Q_bp, p_bp), (Q_lb, p_lb)

def teste_periodograma_acumulado(residuos):
    """
    Teste Kolmogorov-Smirnov no Periodograma Acumulado.
    Verifica se a distribuição espectral é uniforme (Ruído Branco).
    Retorna: (D_stat, Valor_Critico, Periodograma_Acum)
    """
    n = len(residuos)
    if n == 0: return 0, 0, np.array([])

    fft_vals = np.fft.fft(residuos - np.mean(residuos))
    periodograma = np.abs(fft_vals[:n//2])**2

    if np.sum(periodograma) == 0: return 0, 0, np.zeros(n//2)

    P_acum = np.cumsum(periodograma) / np.sum(periodograma)
    freqs = np.arange(1, len(P_acum) + 1) / len(P_acum)

    D = np.max(np.abs(P_acum - freqs))
    critico_95 = 1.36 / np.sqrt(len(P_acum))

    return D, critico_95, P_acum

def calcular_aic_bic(residuos, n_params):
    """
    Calcula Critérios de Informação (AIC e BIC) para seleção de modelo.
    Retorna: (AIC, BIC)
    """
    n = len(residuos)
    sse = np.sum(residuos**2)
    sigma2 = sse / n

    if sigma2 <= 0: return np.inf, np.inf

    log_lik = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)

    aic = 2 * n_params - 2 * log_lik
    bic = n_params * np.log(n) - 2 * log_lik
    return aic, bic

# ==============================================================================
# FUNÇÕES AUXILIARES PARA STREAMLIT
# ==============================================================================

def processar_modelos(serie_trabalho, coluna_alvo):
    """
    Executa o Grid Search de modelos ARIMA (BLOCO 3).
    Retorna: DataFrame com resultados e melhor modelo.
    """
    resultados = []

    # GRID SEARCH (TESTE DE TODOS OS MODELOS)
    for d in [1, 2]:
        ts_diff = diferenciar_serie(serie_trabalho, ordem=d)

        for p in range(4): # p = 0,1,2,3
            for q in range(4): # q = 0,1,2,3
                if p==0 and q==0: continue

                try:
                    # 1. ESTIMAÇÃO
                    phi, _ = estimar_ar_yule_walker(ts_diff, p)
                    theta, _ = estimar_ma_momentos(ts_diff, q)

                    # 2. CÁLCULO DE RESÍDUOS
                    n = len(ts_diff)
                    residuos = np.zeros(n)
                    media = np.mean(ts_diff)

                    for t in range(max(p,q), n):
                        ar_c = np.dot(phi, (ts_diff[t-p:t]-media)[::-1]) if p>0 else 0
                        ma_c = np.dot(theta, residuos[t-q:t][::-1]) if q>0 else 0
                        pred = media + ar_c + ma_c
                        residuos[t] = ts_diff[t] - pred

                    res_val = residuos[max(p,q):]
                    if len(res_val) < 10: continue

                    # 3. OS 5 TESTES ESTATÍSTICOS (DIAGNÓSTICO)
                    n_par = p + q

                    # Critério 1 e 2: AIC e BIC
                    aic, bic = calcular_aic_bic(res_val, n_par)

                    # Critério 3 e 4: Autocorrelação
                    (Qbp, pbp), (Qlb, plb) = teste_box_pierce_ljung_box(res_val, n_par)

                    # Critério 5: Periodograma Acumulado
                    Dks, cks, _ = teste_periodograma_acumulado(res_val)

                    # LÓGICA DE PONTUAÇÃO
                    passou_bp = 1 if pbp > 0.05 else 0
                    passou_lb = 1 if plb > 0.05 else 0
                    passou_ks = 1 if Dks < cks else 0
                    score_diag = passou_bp + passou_lb + passou_ks

                    resultados.append({
                        'modelo': f'ARIMA({p},{d},{q})',
                        'p': p, 'd': d, 'q': q,
                        'aic': aic,
                        'bic': bic,
                        'p_bp': pbp,
                        'p_lb': plb,
                        'D_ks': Dks,
                        'crit_ks': cks,
                        'passou_bp': passou_bp,
                        'passou_lb': passou_lb,
                        'passou_ks': passou_ks,
                        'score': score_diag,
                        'phi': phi, 'theta': theta,
                        'residuos': res_val
                    })
                except:
                    continue

    if len(resultados) > 0:
        df_res = pd.DataFrame(resultados)
        df_res = df_res.sort_values(by=['score', 'aic'], ascending=[False, True]).reset_index(drop=True)
        melhor_modelo = df_res.iloc[0].to_dict()
        return df_res, melhor_modelo
    else:
        return None, None

def gerar_relatorio_inferencia(melhor_modelo, nome_arquivo, coluna_alvo):
    """
    Gera o relatório de inferência estatística (BLOCO 4).
    """
    m = melhor_modelo
    
    try:
        p, d, q = m['p'], m['d'], m['q']
    except KeyError:
        numeros = re.findall(r'\d+', m['modelo'])
        p, d, q = int(numeros[0]), int(numeros[1]), int(numeros[2])

    # Texto para AR
    if p == 0:
        texto_ar = "ausência de termos autorregressivos (AR)"
    elif p == 1:
        texto_ar = "um termo autorregressivo (AR), associado à memória de curto prazo"
    else:
        texto_ar = f"{p} termos autorregressivos (AR), indicando dependência serial de ordem superior"

    # Texto para MA
    if q == 0:
        texto_ma = "ausência de termos de média móvel (MA), o que sugere que os choques aleatórios não possuem persistência relevante"
    elif q == 1:
        texto_ma = "um termo de média móvel (MA), indicando que choques passados afetam o presente imediato"
    else:
        texto_ma = f"{q} termos de média móvel (MA), indicando persistência prolongada dos choques"

    # Lógica para Diagnóstico
    if m['p_lb'] > 0.05 and m['D_ks'] < m['crit_ks']:
        texto_diag = (
            "Os testes de diagnóstico corroboraram o ajuste do modelo: o teste de Ljung-Box não apresentou evidências "
            "de autocorrelação nos resíduos (p-valor > 0,05), enquanto o teste de Kolmogorov-Smirnov aplicado ao "
            "periodograma acumulado indicou que os erros se comportam como ruído branco. Dessa forma, o modelo pode "
            "ser considerado estatisticamente adequado para fins de previsão."
        )
    else:
        texto_diag = (
            "No entanto, os testes de diagnóstico indicaram ressalvas: embora o modelo tenha o melhor critério de informação (AIC), "
            "o teste de Ljung-Box ou o teste espectral (KS) rejeitaram a hipótese de total independência dos resíduos. "
            "Isso sugere que, apesar do ajuste, o modelo pode não ter capturado toda a dinâmica dos dados ou haver volatilidade condicional presente."
        )

    texto_final = (
        f"A análise foi conduzida com base na série temporal extraída do arquivo '{nome_arquivo}', utilizando os valores da coluna "
        f"'{coluna_alvo}' como variável de interesse. Os resultados do processo de modelagem indicaram que o comportamento da série "
        f"é mais adequadamente representado por um modelo ARIMA({p},{d},{q}). "
        f"Verificou-se que a série é integrada de ordem {d} (I({d})), demandando diferenciação para atingir estacionariedade. "
        f"A estrutura de dependência temporal é explicada por {texto_ar} e {texto_ma}. "
        f"\n\n{texto_diag}"
    )

    return texto_final

# ==============================================================================
# FUNÇÃO PRINCIPAL STREAMLIT
# ==============================================================================

def main():
    st.set_page_config(
        page_title="Modelagem ARIMA From Scratch",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 Modelagem ARIMA From Scratch")
    st.markdown("**Aplicativo Acadêmico para Análise de Séries Temporais**")
    st.markdown("---")

    # Inicializar session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'coluna_alvo' not in st.session_state:
        st.session_state.coluna_alvo = None
    if 'serie_trabalho' not in st.session_state:
        st.session_state.serie_trabalho = None
    if 'sucesso_log' not in st.session_state:
        st.session_state.sucesso_log = False
    if 'melhor_modelo' not in st.session_state:
        st.session_state.melhor_modelo = None
    if 'nome_arquivo' not in st.session_state:
        st.session_state.nome_arquivo = None

    # ==========================================================================
    # SEÇÃO 1: UPLOAD E SELEÇÃO DE DADOS
    # ==========================================================================
    st.header("📂 1. Carregamento de Dados")

    uploaded_file = st.file_uploader(
        "Faça upload de um arquivo CSV",
        type=['csv'],
        help="Selecione um arquivo CSV contendo sua série temporal"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.nome_arquivo = uploaded_file.name
            
            st.success(f"✅ Arquivo '{uploaded_file.name}' carregado com sucesso!")
            st.dataframe(df.head(), use_container_width=True)
            
            # Seleção de coluna
            colunas = list(df.columns)
            coluna_selecionada = st.selectbox(
                "Selecione a coluna da série temporal:",
                colunas,
                help="Escolha a coluna que contém os valores da série temporal"
            )
            
            if coluna_selecionada:
                st.session_state.coluna_alvo = coluna_selecionada
                
                # Pré-processamento
                ts = pd.to_numeric(df[coluna_selecionada], errors='coerce').dropna().values
                
                if len(ts) == 0:
                    st.error("❌ A coluna selecionada não contém valores numéricos válidos.")
                else:
                    # Aplicar transformação log
                    ts_log, sucesso_log = aplicar_log(ts)
                    serie_trabalho = ts_log if sucesso_log else ts
                    
                    st.session_state.serie_trabalho = serie_trabalho
                    st.session_state.sucesso_log = sucesso_log
                    
                    # ==========================================================
                    # SEÇÃO 2: ANÁLISE EXPLORATÓRIA
                    # ==========================================================
                    st.header("🔍 2. Análise Exploratória")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Série Original")
                        fig1, ax1 = plt.subplots(figsize=(10, 4))
                        ax1.plot(ts, color='#1f77b4')
                        ax1.set_title(f"Série Original: {coluna_selecionada}")
                        ax1.set_xlabel("Tempo")
                        ax1.set_ylabel("Valor")
                        ax1.grid(True, alpha=0.3)
                        st.pyplot(fig1)
                    
                    with col2:
                        st.subheader("Histograma Original")
                        fig2, ax2 = plt.subplots(figsize=(10, 4))
                        ax2.hist(ts, bins=30, color='#1f77b4', alpha=0.7)
                        ax2.set_title("Distribuição dos Valores")
                        ax2.set_xlabel("Valor")
                        ax2.set_ylabel("Frequência")
                        ax2.grid(True, alpha=0.3)
                        st.pyplot(fig2)
                    
                    if sucesso_log:
                        st.info("✅ Transformação Log aplicada para estabilizar a variância.")
                        col3, col4 = st.columns(2)
                        
                        with col3:
                            st.subheader("Série Log (ln)")
                            fig3, ax3 = plt.subplots(figsize=(10, 4))
                            ax3.plot(ts_log, color='#2ca02c')
                            ax3.set_title("Série após Transformação Log")
                            ax3.set_xlabel("Tempo")
                            ax3.set_ylabel("ln(Valor)")
                            ax3.grid(True, alpha=0.3)
                            st.pyplot(fig3)
                        
                        with col4:
                            st.subheader("Histograma (ln)")
                            fig4, ax4 = plt.subplots(figsize=(10, 4))
                            ax4.hist(ts_log, bins=30, color='#2ca02c', alpha=0.7)
                            ax4.set_title("Distribuição após Log")
                            ax4.set_xlabel("ln(Valor)")
                            ax4.set_ylabel("Frequência")
                            ax4.grid(True, alpha=0.3)
                            st.pyplot(fig4)
                    else:
                        st.warning("⚠️ Transformação Log não aplicada (valores <= 0 detectados).")
                    
                    # FAC / FACP da Diferença
                    st.subheader("Análise de Identificação (Série Diferenciada d=1)")
                    ts_diff = diferenciar_serie(serie_trabalho, 1)
                    
                    col5, col6 = st.columns(2)
                    
                    with col5:
                        fac = calcular_fac_manual(ts_diff)
                        fig5, ax5 = plt.subplots(figsize=(10, 4))
                        ax5.stem(fac)
                        lim = 1.96/np.sqrt(len(ts_diff))
                        ax5.axhline(lim, c='r', ls='--', label='Limite 95%')
                        ax5.axhline(-lim, c='r', ls='--')
                        ax5.set_title("FAC (Autocorrelação)")
                        ax5.set_xlabel("Lag")
                        ax5.set_ylabel("Autocorrelação")
                        ax5.legend()
                        ax5.grid(True, alpha=0.3)
                        st.pyplot(fig5)
                    
                    with col6:
                        facp = calcular_facp_manual(ts_diff)
                        fig6, ax6 = plt.subplots(figsize=(10, 4))
                        ax6.stem(facp)
                        lim = 1.96/np.sqrt(len(ts_diff))
                        ax6.axhline(lim, c='r', ls='--', label='Limite 95%')
                        ax6.axhline(-lim, c='r', ls='--')
                        ax6.set_title("FACP (Autocorrelação Parcial)")
                        ax6.set_xlabel("Lag")
                        ax6.set_ylabel("Autocorrelação Parcial")
                        ax6.legend()
                        ax6.grid(True, alpha=0.3)
                        st.pyplot(fig6)
                    
                    # ==========================================================
                    # SEÇÃO 3: MODELAGEM ARIMA
                    # ==========================================================
                    st.header("⚙️ 3. Modelagem ARIMA")
                    st.markdown("**Critério de Escolha:** 1º Aprovação nos Testes Estatísticos -> 2º Menor AIC")
                    
                    if st.button("🔬 Calcular Modelos", type="primary", use_container_width=True):
                        with st.spinner("🔄 Processando modelos ARIMA... Isso pode levar alguns minutos."):
                            df_res, melhor_modelo = processar_modelos(serie_trabalho, coluna_selecionada)
                            
                            if df_res is not None and melhor_modelo is not None:
                                st.session_state.melhor_modelo = melhor_modelo
                                
                                st.success("✅ Modelagem concluída com sucesso!")
                                
                                # Top 5 Modelos
                                st.subheader("📋 Top 5 Modelos Candidatos")
                                display_cols = ['modelo', 'score', 'aic', 'p_lb', 'p_bp', 'passou_ks']
                                st.dataframe(
                                    df_res[display_cols].head(5),
                                    use_container_width=True,
                                    hide_index=True
                                )
                                
                                st.caption("**Legenda:** score = Quantos testes estatísticos o modelo passou (máx 3); p_lb/p_bp = P-valor (>0.05 é aprovado); passou_ks = 1 = Aprovado no Periodograma")
                                
                                # ==============================================
                                # SEÇÃO 4: RESULTADOS DO VENCEDOR
                                # ==============================================
                                st.header("🏆 4. Diagnóstico do Modelo Vencedor")
                                
                                m = melhor_modelo
                                st.markdown(f"### Modelo Selecionado: **{m['modelo']}**")
                                
                                # Métricas em colunas
                                col_met1, col_met2, col_met3, col_met4, col_met5 = st.columns(5)
                                
                                with col_met1:
                                    st.metric("AIC", f"{m['aic']:.4f}")
                                
                                with col_met2:
                                    st.metric("BIC", f"{m['bic']:.4f}")
                                
                                with col_met3:
                                    status_bp = "✅" if m['passou_bp'] else "❌"
                                    st.metric("Box-Pierce", f"{m['p_bp']:.4f}", help=f"P-valor: {m['p_bp']:.4f}")
                                    st.caption(f"{status_bp} {'Aprovado' if m['passou_bp'] else 'Rejeitado'}")
                                
                                with col_met4:
                                    status_lb = "✅" if m['passou_lb'] else "❌"
                                    st.metric("Ljung-Box", f"{m['p_lb']:.4f}", help=f"P-valor: {m['p_lb']:.4f}")
                                    st.caption(f"{status_lb} {'Aprovado' if m['passou_lb'] else 'Rejeitado'}")
                                
                                with col_met5:
                                    status_ks = "✅" if m['passou_ks'] else "❌"
                                    st.metric("KS Test", f"{m['D_ks']:.4f}", help=f"D = {m['D_ks']:.4f} (Crítico: {m['crit_ks']:.4f})")
                                    st.caption(f"{status_ks} {'Aprovado' if m['passou_ks'] else 'Rejeitado'}")
                                
                                # Análise dos 5 Critérios
                                st.subheader("📊 Análise dos 5 Critérios Estatísticos")
                                
                                criterios = {
                                    "1. AIC (Akaike)": {
                                        "valor": f"{m['aic']:.4f}",
                                        "conclusao": "Excelente" if m['aic'] == df_res['aic'].min() else "Bom (Balanceado)"
                                    },
                                    "2. BIC (Bayesiano)": {
                                        "valor": f"{m['bic']:.4f}",
                                        "conclusao": "Confirma a penalização por complexidade"
                                    },
                                    "3. Teste Box-Pierce": {
                                        "valor": f"p-valor = {m['p_bp']:.4f}",
                                        "conclusao": "✅ APROVADO (Resíduos Independentes)" if m['passou_bp'] else "❌ REJEITADO (Falha)"
                                    },
                                    "4. Teste Ljung-Box": {
                                        "valor": f"p-valor = {m['p_lb']:.4f}",
                                        "conclusao": "✅ APROVADO (Resíduos Independentes)" if m['passou_lb'] else "❌ REJEITADO (Autocorrelação detectada)"
                                    },
                                    "5. Periodograma Acumulado (KS)": {
                                        "valor": f"D = {m['D_ks']:.4f} (Crítico: {m['crit_ks']:.4f})",
                                        "conclusao": "✅ APROVADO (Ruído Branco)" if m['passou_ks'] else "❌ REJEITADO (Padrão espectral detectado)"
                                    }
                                }
                                
                                for criterio, info in criterios.items():
                                    with st.expander(criterio):
                                        st.write(f"**Valor:** {info['valor']}")
                                        st.write(f"**Conclusão:** {info['conclusao']}")
                                
                                # Veredito Final
                                if m['score'] == 3:
                                    st.success(f"✅ **VEREDITO FINAL:** O modelo {m['modelo']} é estatisticamente adequado e robusto.")
                                else:
                                    st.warning(f"⚠️ **VEREDITO FINAL:** O modelo {m['modelo']} é o melhor disponível, mas não passou em todos os testes. Sugestão: A série pode precisar de tratamentos adicionais (ex: GARCH) ou remoção de outliers.")
                                
                                # Parâmetros
                                st.subheader("📐 Parâmetros do Modelo")
                                col_param1, col_param2 = st.columns(2)
                                
                                with col_param1:
                                    if len(m['phi']) > 0:
                                        st.write(f"**AR (phi):** {m['phi']}")
                                    else:
                                        st.write("**AR (phi):** Nenhum")
                                
                                with col_param2:
                                    if len(m['theta']) > 0:
                                        st.write(f"**MA (theta):** {m['theta']}")
                                    else:
                                        st.write("**MA (theta):** Nenhum")
                                
                                # Gráficos de Diagnóstico
                                st.subheader("📈 Gráficos de Diagnóstico")
                                res = m['residuos']
                                
                                fig_diag, axes = plt.subplots(2, 2, figsize=(14, 9))
                                fig_diag.suptitle(f"Gráficos de Diagnóstico: {m['modelo']}")
                                
                                # Resíduos Tempo
                                axes[0,0].plot(res, c='purple', alpha=0.8)
                                axes[0,0].axhline(0, c='k', ls='--')
                                axes[0,0].set_title("Resíduos")
                                axes[0,0].set_xlabel("Tempo")
                                axes[0,0].set_ylabel("Resíduo")
                                axes[0,0].grid(True, alpha=0.3)
                                
                                # Histograma
                                axes[0,1].hist(res, density=True, bins=30, color='orange', alpha=0.6)
                                x = np.linspace(min(res), max(res), 100)
                                axes[0,1].plot(x, norm.pdf(x, res.mean(), res.std()), 'r-', label='Normal')
                                axes[0,1].set_title("Distribuição dos Erros")
                                axes[0,1].set_xlabel("Resíduo")
                                axes[0,1].set_ylabel("Densidade")
                                axes[0,1].legend()
                                axes[0,1].grid(True, alpha=0.3)
                                
                                # FAC Resíduos
                                facr = calcular_fac_manual(res)
                                axes[1,0].stem(facr)
                                lm = 1.96/np.sqrt(len(res))
                                axes[1,0].axhline(lm, c='r', ls='--', label='Limite 95%')
                                axes[1,0].axhline(-lm, c='r', ls='--')
                                axes[1,0].set_title("FAC dos Resíduos")
                                axes[1,0].set_xlabel("Lag")
                                axes[1,0].set_ylabel("Autocorrelação")
                                axes[1,0].legend()
                                axes[1,0].grid(True, alpha=0.3)
                                
                                # Periodograma
                                _, _, pacum = teste_periodograma_acumulado(res)
                                freqs = np.linspace(0, 1, len(pacum))
                                axes[1,1].plot(freqs, pacum, label='Modelo')
                                axes[1,1].plot([0,1], [0,1], 'r--', label='Ideal (Ruído Branco)')
                                axes[1,1].set_title("Periodograma Acumulado")
                                axes[1,1].set_xlabel("Frequência Normalizada")
                                axes[1,1].set_ylabel("Periodograma Acumulado")
                                axes[1,1].legend()
                                axes[1,1].grid(True, alpha=0.3)
                                
                                plt.tight_layout()
                                st.pyplot(fig_diag)
                                
                                # ==============================================
                                # SEÇÃO 5: INFERÊNCIA ESTATÍSTICA
                                # ==============================================
                                st.header("📑 5. Relatório de Inferência Estatística")
                                
                                texto_relatorio = gerar_relatorio_inferencia(
                                    melhor_modelo,
                                    st.session_state.nome_arquivo,
                                    coluna_selecionada
                                )
                                
                                st.markdown(textwrap.fill(texto_relatorio, width=100))
                                
                                # ==============================================
                                # SEÇÃO 6: VALIDAÇÃO DE ACURÁCIA
                                # ==============================================
                                st.header("🧪 6. Validação de Acurácia")
                                
                                # Recuperar dados
                                ts_diff = diferenciar_serie(serie_trabalho, ordem=m['d'])
                                residuos = m['residuos']
                                tamanho_validacao = len(residuos)
                                ts_diff_recorte = ts_diff[-tamanho_validacao:]
                                diff_previsto = ts_diff_recorte - residuos
                                inicio_validacao = len(serie_trabalho) - tamanho_validacao
                                original_log_recorte = serie_trabalho[inicio_validacao:]
                                
                                # Reconstrução
                                if m['d'] == 1:
                                    base = serie_trabalho[inicio_validacao - 1]
                                    previsto_log = base + np.cumsum(diff_previsto)
                                elif m['d'] == 2:
                                    st.warning("⚠️ Para d=2, exibindo a comparação na escala diferenciada.")
                                    original_log_recorte = ts_diff_recorte
                                    previsto_log = diff_previsto
                                else:
                                    previsto_log = diff_previsto
                                
                                # Reverter Log
                                if st.session_state.sucesso_log:
                                    valores_reais = np.exp(original_log_recorte)
                                    valores_previstos = np.exp(previsto_log)
                                    label_y = "Valores Originais (Destransformados)"
                                else:
                                    valores_reais = original_log_recorte
                                    valores_previstos = previsto_log
                                    label_y = "Valores Originais"
                                
                                # Métricas
                                rmse = np.sqrt(np.mean((valores_reais - valores_previstos)**2))
                                mask = valores_reais != 0
                                mape = np.mean(np.abs((valores_reais[mask] - valores_previstos[mask]) / valores_reais[mask])) * 100
                                
                                col_rmse, col_mape = st.columns(2)
                                
                                with col_rmse:
                                    st.metric("RMSE", f"{rmse:.4f}", help="Raiz do Erro Quadrático Médio")
                                
                                with col_mape:
                                    if mape < 10:
                                        classificacao = "⭐ Excelente"
                                    elif mape < 20:
                                        classificacao = "✅ Bom"
                                    elif mape < 50:
                                        classificacao = "⚠️ Razoável"
                                    else:
                                        classificacao = "❌ Ruim"
                                    st.metric("MAPE", f"{mape:.2f}%", help="Erro Percentual Médio Absoluto")
                                    st.caption(classificacao)
                                
                                # Gráfico de Validação
                                fig_val, ax_val = plt.subplots(figsize=(14, 6))
                                ax_val.plot(valores_reais, label='Real (Observado)', color='blue', alpha=0.6, linewidth=2)
                                ax_val.plot(valores_previstos, label='Modelo (Ajustado)', color='red', linestyle='--', linewidth=2)
                                ax_val.set_title(f"Teste de Acurácia: {m['modelo']} (MAPE: {mape:.2f}%)")
                                ax_val.set_xlabel("Passos de Tempo (Amostra de Validação)")
                                ax_val.set_ylabel(label_y)
                                ax_val.legend()
                                ax_val.grid(True, alpha=0.3)
                                st.pyplot(fig_val)
                            else:
                                st.error("❌ Nenhum modelo foi gerado. Verifique os dados de entrada.")
        except Exception as e:
            st.error(f"❌ Erro ao processar arquivo: {e}")
    else:
        st.info("👆 Por favor, faça upload de um arquivo CSV para começar.")

if __name__ == "__main__":
    main()
