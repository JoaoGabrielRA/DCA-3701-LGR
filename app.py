"""Ferramenta interativa de Lugar Geometrico das Raizes (LGR)."""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import sympy

st.set_page_config(page_title="Lugar Geometrico das Raizes (LGR)", layout="wide", page_icon="📈")

PALETA = {
    "polo": "#e63946",
    "zero": "#2a9d8f",
    "lgr": "#264653",
    "assintota": "#f4a261",
    "destaque": "#8338ec",
    "aceito": "#2a9d8f",
    "rejeitado": "#e63946",
}

EXEMPLOS = {
    "Personalizado": None,
    "Sistema de 2a ordem simples": ("1", "1 2 0", "1", "1"),
    "Sistema de 3a ordem com breakaway": ("1", "1 5 6 0", "1", "1"),
    "Sistema com par complexo": ("1", "1 4 8 0", "1", "1"),
    "Sistema com zero finito": ("1 3", "1 2 5 0", "1", "1"),
}


# ------------------------------------------------------------------
# Formatacao para LaTeX
# ------------------------------------------------------------------

def latex_polinomio(coefs, variavel="s"):
    grau = len(coefs) - 1
    termos = []
    for indice, coef in enumerate(coefs):
        potencia = grau - indice
        if abs(coef) < 1e-12:
            continue
        modulo = abs(coef)
        if potencia == 0:
            corpo = f"{modulo:g}"
        elif abs(modulo - 1) < 1e-12:
            corpo = ""
        else:
            corpo = f"{modulo:g}"
        if potencia == 0:
            texto = corpo
        elif potencia == 1:
            texto = f"{corpo}{variavel}" if corpo else variavel
        else:
            texto = f"{corpo}{variavel}^{{{potencia}}}" if corpo else f"{variavel}^{{{potencia}}}"
        sinal = "-" if coef < 0 else ("+" if termos else "")
        termos.append(f"{sinal}{texto}" if not termos else f" {sinal} {texto}")
    return "".join(termos) if termos else "0"


def latex_complexo(valor, casas=4):
    parte_r = round(valor.real, casas)
    parte_i = round(valor.imag, casas)
    if abs(parte_i) < 1e-10:
        return f"{parte_r:g}"
    if abs(parte_r) < 1e-10:
        if abs(abs(parte_i) - 1) < 1e-10:
            return "j" if parte_i > 0 else "-j"
        return f"{parte_i:g}j"
    sinal = "+" if parte_i >= 0 else "-"
    return f"{parte_r:g} {sinal} {abs(parte_i):g}j"


def latex_fatorado(raizes, variavel="s"):
    if len(raizes) == 0:
        return "1"
    fatores = []
    for raiz in raizes:
        if abs(raiz.imag) < 1e-8:
            valor = round(raiz.real, 4)
            if abs(valor) < 1e-8:
                fatores.append(variavel)
            elif valor > 0:
                fatores.append(f"({variavel} - {valor:g})")
            else:
                fatores.append(f"({variavel} + {abs(valor):g})")
        else:
            fatores.append(rf"\left({variavel} - ({latex_complexo(raiz)})\right)")
    agrupados = {}
    ordem = []
    for fator in fatores:
        if fator not in agrupados:
            agrupados[fator] = 0
            ordem.append(fator)
        agrupados[fator] += 1
    return "".join(f if agrupados[f] == 1 else f"{f}^{{{agrupados[f]}}}" for f in ordem)


# ------------------------------------------------------------------
# Nucleo de calculo
# ------------------------------------------------------------------

def ler_coeficientes(texto):
    try:
        valores = [float(x) for x in texto.strip().split()]
        return np.array(valores) if valores else None
    except Exception:
        return None


def decompor_jw(coefs):
    grau = len(coefs) - 1
    reais, imags = {}, {}
    for indice, coef in enumerate(coefs):
        potencia = grau - indice
        resto = potencia % 4
        if resto == 0:
            reais[potencia] = reais.get(potencia, 0) + coef
        elif resto == 1:
            imags[potencia] = imags.get(potencia, 0) + coef
        elif resto == 2:
            reais[potencia] = reais.get(potencia, 0) - coef
        else:
            imags[potencia] = imags.get(potencia, 0) - coef

    def montar(dicionario):
        if not dicionario:
            return np.array([0.0])
        grau_max = max(dicionario.keys())
        vetor = np.zeros(grau_max + 1)
        for potencia, valor in dicionario.items():
            vetor[grau_max - potencia] = valor
        return vetor

    return montar(reais), montar(imags)


def malha_aberta(num_g, den_g, num_h, den_h):
    numerador = np.convolve(num_g, num_h)
    denominador = np.convolve(den_g, den_h)
    tamanho = max(len(numerador), len(denominador))
    numerador = np.pad(numerador, (tamanho - len(numerador), 0))
    denominador = np.pad(denominador, (tamanho - len(denominador), 0))
    return numerador, denominador


def segmentos_eixo_real(zeros, polos):
    pontos_reais = [p.real for p in polos if abs(p.imag) < 1e-8]
    pontos_reais += [z.real for z in zeros if abs(z.imag) < 1e-8]
    if not pontos_reais:
        return []
    marcos = sorted(set(round(v, 8) for v in pontos_reais), reverse=True)
    segmentos = []
    for i in range(len(marcos) - 1):
        ponto_medio = (marcos[i] + marcos[i + 1]) / 2
        quantidade_direita = sum(1 for v in pontos_reais if v > ponto_medio + 1e-10)
        if quantidade_direita % 2 == 1:
            segmentos.append((marcos[i + 1], marcos[i]))
    if len(pontos_reais) % 2 == 1:
        segmentos.append((-np.inf, marcos[-1]))
    return segmentos


def geometria_assintotas(zeros, polos):
    n_polos, n_zeros = len(polos), len(zeros)
    diferenca = n_polos - n_zeros
    if diferenca == 0:
        return None, []
    centroide = (np.sum(polos).real - np.sum(zeros).real) / diferenca
    angulos = [(2 * q + 1) * 180.0 / diferenca for q in range(diferenca)]
    return centroide, angulos


def pontos_breakaway(num, den, polos, zeros):
    derivada_num = np.polyder(num)
    derivada_den = np.polyder(den)
    equacao = np.polysub(np.convolve(num, derivada_den), np.convolve(den, derivada_num))
    candidatos = np.roots(equacao)
    referencias = [p.real for p in polos if abs(p.imag) < 1e-8]
    referencias += [z.real for z in zeros if abs(z.imag) < 1e-8]
    validos = []
    for raiz in candidatos:
        valor_num = np.polyval(num, raiz)
        ganho = -np.polyval(den, raiz) / valor_num if abs(valor_num) > 1e-12 else None
        if abs(raiz.imag) < 1e-6:
            posicao = raiz.real
            a_direita = sum(1 for v in referencias if v > posicao + 1e-10)
            pertence = a_direita % 2 == 1
            ganho_real = ganho.real if ganho is not None else np.inf
            if pertence and ganho_real > 0:
                validos.append((posicao, ganho_real))
        elif ganho is not None and abs(ganho.imag) < 1e-6 and ganho.real > 0:
            validos.append((complex(raiz), ganho.real))
    return validos


def cruzamentos_eixo_imaginario(den, num):
    re_d, im_d = decompor_jw(den)
    re_n, im_n = decompor_jw(num)
    equacao_cruzamento = np.polysub(np.convolve(re_d, im_n), np.convolve(im_d, re_n))
    while len(equacao_cruzamento) > 1 and abs(equacao_cruzamento[0]) < 1e-12:
        equacao_cruzamento = equacao_cruzamento[1:]
    detalhes = {"re_d": re_d, "im_d": im_d, "re_n": re_n, "im_n": im_n, "cruzamento": equacao_cruzamento}
    if len(equacao_cruzamento) <= 1:
        return [], detalhes
    frequencias = np.roots(equacao_cruzamento)
    achados = []
    for freq in frequencias:
        if abs(freq.imag) > 1e-6 or freq.real < 1e-8:
            continue
        omega = freq.real
        im_n_v, im_d_v = np.polyval(im_n, omega), np.polyval(im_d, omega)
        re_n_v, re_d_v = np.polyval(re_n, omega), np.polyval(re_d, omega)
        if abs(im_n_v) > 1e-12:
            ganho = -im_d_v / im_n_v
        elif abs(re_n_v) > 1e-12:
            ganho = -re_d_v / re_n_v
        else:
            continue
        if ganho > 1e-10 and not any(
            abs(ganho - k) < 1e-4 and abs(omega - w) < 1e-4 for k, w in achados
        ):
            achados.append((ganho, omega))
    return achados, detalhes


def tabela_routh_hurwitz(den, num):
    simbolo_k = sympy.Symbol("K", positive=True)
    tamanho = max(len(den), len(num))
    den_pad = np.pad(den, (tamanho - len(den), 0))
    num_pad = np.pad(num, (tamanho - len(num), 0))

    def para_racional(valor):
        arredondado = round(valor)
        return sympy.Integer(arredondado) if abs(valor - arredondado) < 1e-10 else sympy.nsimplify(valor, rational=True)

    linha_coefs = [para_racional(d) + simbolo_k * para_racional(n) for d, n in zip(den_pad, num_pad)]
    grau = len(linha_coefs) - 1
    colunas = (grau + 2) // 2

    tabela = [[sympy.S.Zero] * colunas for _ in range(grau + 1)]
    for j in range(colunas):
        if 2 * j < len(linha_coefs):
            tabela[0][j] = linha_coefs[2 * j]
        if 2 * j + 1 < len(linha_coefs):
            tabela[1][j] = linha_coefs[2 * j + 1]

    for i in range(2, grau + 1):
        pivo = tabela[i - 1][0]
        if pivo == 0:
            break
        for j in range(colunas - 1):
            numerador = tabela[i - 1][0] * tabela[i - 2][j + 1] - tabela[i - 2][0] * tabela[i - 1][j + 1]
            tabela[i][j] = sympy.simplify(numerador / pivo)

    condicoes = []
    ganhos_criticos = set()
    for i in range(grau + 1):
        expressao = sympy.simplify(tabela[i][0])
        if expressao.has(simbolo_k):
            try:
                solucao = sympy.solve(expressao > 0, simbolo_k)
                condicao_latex = r"\forall\; K > 0" if solucao in (True, sympy.S.true) else sympy.latex(solucao)
            except Exception:
                condicao_latex = None
            condicoes.append((grau - i, expressao, condicao_latex))
            for raiz in sympy.solve(sympy.Eq(expressao, 0), simbolo_k):
                if raiz.is_real and raiz > 0:
                    ganhos_criticos.add(float(raiz))

    return {
        "tabela": tabela, "grau": grau, "colunas": colunas,
        "condicoes": condicoes, "ganhos_criticos": sorted(ganhos_criticos), "simbolo_k": simbolo_k,
    }


def reordenar_ramos(raizes_anteriores, raizes_atuais):
    n = len(raizes_atuais)
    if n == 0:
        return raizes_atuais
    saida = np.zeros(n, dtype=complex)
    usados = set()
    for i in range(n):
        melhor_indice, menor_distancia = None, np.inf
        for j in range(n):
            if j in usados:
                continue
            distancia = abs(raizes_anteriores[i] - raizes_atuais[j])
            if distancia < menor_distancia:
                menor_distancia, melhor_indice = distancia, j
        saida[i] = raizes_atuais[melhor_indice]
        usados.add(melhor_indice)
    return saida


def varrer_lgr(num, den, ganho_max=None):
    ordem = len(den) - 1
    if ganho_max is None:
        ganho_max = 1000.0
        for tentativa in (100, 500, 1000, 5000):
            polinomio = np.polyadd(den, tentativa * num)
            if np.max(np.abs(np.roots(polinomio))) > 50:
                ganho_max = tentativa
                break
    faixa_fina = np.linspace(0, 0.1, 200)
    faixa_log = np.logspace(-1, np.log10(ganho_max), 4800)
    ganhos = np.unique(np.concatenate([faixa_fina, faixa_log]))
    ganhos.sort()
    ramos = np.zeros((len(ganhos), ordem), dtype=complex)
    for i, k in enumerate(ganhos):
        raizes = np.roots(np.polyadd(den, k * num))
        ramos[i] = reordenar_ramos(ramos[i - 1], raizes) if i > 0 else raizes
    return ganhos, ramos


def avaliar_criterio_angulo(ponto, zeros, polos):
    soma_polos = sum(np.degrees(np.angle(ponto - p)) for p in polos)
    soma_zeros = sum(np.degrees(np.angle(ponto - z)) for z in zeros)
    diferenca = soma_zeros - soma_polos
    diferenca_normalizada = ((diferenca + 180) % 360) - 180
    pertence = abs(abs(diferenca_normalizada) - 180) < 5.0
    ganho = None
    if pertence:
        produto_polos = np.prod([abs(ponto - p) for p in polos]) if len(polos) else 1.0
        produto_zeros = np.prod([abs(ponto - z) for z in zeros]) if len(zeros) else 1.0
        ganho = produto_polos / produto_zeros if produto_zeros > 1e-12 else np.inf
    return diferenca, diferenca_normalizada, pertence, ganho


def ganho_no_ponto(ponto, zeros, polos):
    produto_polos = np.prod([abs(ponto - p) for p in polos]) if len(polos) else 1.0
    produto_zeros = np.prod([abs(ponto - z) for z in zeros]) if len(zeros) else 1.0
    return produto_polos / produto_zeros if produto_zeros > 1e-12 else np.inf


# ------------------------------------------------------------------
# Plotagem
# ------------------------------------------------------------------

def calcular_limites(polos, zeros, pontos_extra=None):
    conjunto = np.concatenate([polos, zeros]) if len(zeros) > 0 else polos
    amplitude = max(np.ptp(conjunto.real), np.ptp(conjunto.imag), 1.0)
    margem = max(1.0, 0.5 * amplitude)
    eixo_x = list(conjunto.real)
    if pontos_extra:
        eixo_x.extend(pontos_extra)
    limite_x = (min(eixo_x) - margem, max(eixo_x) + margem)
    limite_y = max(abs(conjunto.imag).max() + margem * 0.5, margem)
    return limite_x, limite_y, margem


def plotar_polos_zeros(eixo, polos, zeros):
    eixo.plot(polos.real, polos.imag, "x", color=PALETA["polo"], ms=10, mew=2, label="Polos", zorder=5)
    if len(zeros) > 0:
        eixo.plot(zeros.real, zeros.imag, "o", color=PALETA["zero"], ms=8, mew=2,
                   fillstyle="none", label="Zeros", zorder=5)


def finalizar_eixo(eixo, limite_x, limite_y, titulo="", limites_usuario=None):
    if limites_usuario is not None:
        eixo.set_xlim(limites_usuario[0], limites_usuario[1])
        eixo.set_ylim(limites_usuario[2], limites_usuario[3])
    else:
        eixo.set_xlim(limite_x)
        eixo.set_ylim(-limite_y, limite_y)
    eixo.axhline(0, color="k", lw=0.5, alpha=0.3)
    eixo.axvline(0, color="k", lw=0.5, alpha=0.3)
    eixo.grid(True, alpha=0.25)
    eixo.set_xlabel(r"Real ($\sigma$)")
    eixo.set_ylabel(r"Imaginario ($j\omega$)")
    eixo.set_title(titulo)
    eixo.legend(fontsize=9)


def plotar_segmentos(eixo, segmentos, limite_x):
    for i, (a, b) in enumerate(segmentos):
        a_visivel = max(a, limite_x[0] - 5) if np.isfinite(a) else limite_x[0] - 5
        b_visivel = min(b, limite_x[1] + 5) if np.isfinite(b) else limite_x[1] + 5
        rotulo = "Segmento LGR" if i == 0 else ""
        eixo.plot([a_visivel, b_visivel], [0, 0], color=PALETA["lgr"], linewidth=4,
                   alpha=0.6, solid_capstyle="round", label=rotulo)


def plotar_assintotas(eixo, centroide, angulos, limite_x, limite_y):
    comprimento = max(abs(limite_x[0]), abs(limite_x[1]), limite_y) * 2
    for i, angulo_graus in enumerate(angulos):
        angulo_rad = np.radians(angulo_graus)
        dx, dy = comprimento * np.cos(angulo_rad), comprimento * np.sin(angulo_rad)
        rotulo = "Assintotas" if i == 0 else ""
        eixo.plot([centroide, centroide + dx], [0, dy], "--", color=PALETA["assintota"],
                   linewidth=1.5, alpha=0.8, label=rotulo)


def plotar_ramos_fundo(eixo, ramos):
    for coluna in range(ramos.shape[1]):
        ramo = ramos[:, coluna]
        eixo.plot(ramo.real, ramo.imag, "-", color="gray", linewidth=1.4, alpha=0.35)


# ------------------------------------------------------------------
# Interface
# ------------------------------------------------------------------

st.title("Lugar Geometrico das Raizes (LGR)")
st.caption("Analise passo a passo do Lugar Geometrico das Raizes — João Gabriel R. de Azevedo · DCA-3701 UFRN")

st.markdown("""
<style>
    .katex-display { text-align: left !important; }
    div[data-testid="stMetric"] { background-color: rgba(38,70,83,0.06); border-radius: 8px; padding: 8px; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Configuracao")
    exemplo_escolhido = st.selectbox("Exemplo pronto", list(EXEMPLOS.keys()))
    valores_padrao = EXEMPLOS[exemplo_escolhido] or ("1", "1 4 0", "1", "1")

    with st.form("form_config"):
        st.subheader("G(s) = K · Ng(s) / Dg(s)")
        txt_num_g = st.text_input("Numerador de G(s)", value=valores_padrao[0])
        txt_den_g = st.text_input("Denominador de G(s)", value=valores_padrao[1])

        st.subheader("H(s) = Nh(s) / Dh(s)")
        txt_num_h = st.text_input("Numerador de H(s)", value=valores_padrao[2])
        txt_den_h = st.text_input("Denominador de H(s)", value=valores_padrao[3])

        st.subheader("Ponto de teste")
        parte_real = st.number_input("Parte real", value=0.0, format="%.4f")
        parte_imag = st.number_input("Parte imaginaria", value=0.0, format="%.4f")

        st.subheader("Limites do grafico")
        limites_manuais = st.checkbox("Definir manualmente", value=False)
        if limites_manuais:
            x_min = st.number_input("x min", value=-10.0, format="%.2f")
            x_max = st.number_input("x max", value=2.0, format="%.2f")
            y_min = st.number_input("y min", value=-10.0, format="%.2f")
            y_max = st.number_input("y max", value=10.0, format="%.2f")

        calcular = st.form_submit_button("Calcular LGR", type="primary", use_container_width=True)

if calcular:
    st.session_state["pronto"] = True

if not st.session_state.get("pronto", False):
    st.info("Preencha os coeficientes na barra lateral e clique em **Calcular LGR** para comecar.")
    st.stop()

nG, dG = ler_coeficientes(txt_num_g), ler_coeficientes(txt_den_g)
nH, dH = ler_coeficientes(txt_num_h), ler_coeficientes(txt_den_h)

if any(v is None for v in (nG, dG, nH, dH)):
    st.error("Confira os coeficientes — algo nao pode ser interpretado como numero.")
    st.stop()

num, den = malha_aberta(nG, dG, nH, dH)
zeros, polos = np.roots(num), np.roots(den)
segmentos = segmentos_eixo_real(zeros, polos)
n_lugares = max(len(polos), len(zeros))
centroide, angulos_asc = geometria_assintotas(zeros, polos)
breakaways = pontos_breakaway(num, den, polos, zeros)
routh = tabela_routh_hurwitz(den, num)
cruzamentos, info_jw = cruzamentos_eixo_imaginario(den, num)
ganhos_lgr, ramos_lgr = varrer_lgr(num, den)

limite_x, limite_y, margem = calcular_limites(polos, zeros, [centroide] if centroide is not None else None)
limites_usuario = (x_min, x_max, y_min, y_max) if limites_manuais else None

ponto_teste = complex(parte_real, parte_imag)
delta_angulo, delta_normalizado, pertence, _ = avaliar_criterio_angulo(ponto_teste, zeros, polos)
ganho_calculado = ganho_no_ponto(ponto_teste, zeros, polos)

st.markdown("---")


def _fs(largura, altura, escala):
    return (largura * escala, altura * escala)


def passo_1_equacao(escala=1.0):
    nG_l, dG_l = latex_polinomio(nG), latex_polinomio(dG)
    nH_l, dH_l = latex_polinomio(nH), latex_polinomio(dH)
    num_l, den_l = latex_polinomio(num), latex_polinomio(den)
    st.latex(rf"G(s) = K \cdot \frac{{{nG_l}}}{{{dG_l}}} \qquad H(s) = \frac{{{nH_l}}}{{{dH_l}}}")
    st.markdown("Funcao de transferencia de malha aberta:")
    st.latex(rf"G(s)H(s) = K \cdot \frac{{{num_l}}}{{{den_l}}} = K \cdot P(s)")
    st.markdown("Equacao caracteristica:")
    st.latex(rf"1 + K \cdot P(s) = 0 \;\Longrightarrow\; {den_l} + K\left({num_l}\right) = 0")


def passo_2_fatorada(escala=1.0):
    st.latex(rf"P(s) = \frac{{{latex_fatorado(zeros)}}}{{{latex_fatorado(polos)}}}")


def passo_3_polos_zeros(escala=1.0):
    fig, eixo = plt.subplots(figsize=_fs(10, 6, escala))
    plotar_polos_zeros(eixo, polos, zeros)
    for i, p in enumerate(polos):
        eixo.annotate(rf"$p_{{{i+1}}}$", (p.real, p.imag), xytext=(8, 8),
                       textcoords="offset points", fontsize=9, color=PALETA["polo"])
    for i, z in enumerate(zeros):
        eixo.annotate(rf"$z_{{{i+1}}}$", (z.real, z.imag), xytext=(8, 8),
                       textcoords="offset points", fontsize=9, color=PALETA["zero"])
    finalizar_eixo(eixo, limite_x, limite_y, "Polos e zeros no plano s", limites_usuario)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    coluna_p, coluna_z = st.columns(2)
    with coluna_p:
        st.markdown(f"**Polos** (np = {len(polos)})")
        for i, p in enumerate(polos):
            st.latex(rf"p_{{{i+1}}} = {latex_complexo(p)}")
    with coluna_z:
        st.markdown(f"**Zeros** (nz = {len(zeros)})")
        if len(zeros) > 0:
            for i, z in enumerate(zeros):
                st.latex(rf"z_{{{i+1}}} = {latex_complexo(z)}")
        else:
            st.markdown("*Nenhum zero finito*")


def passo_4_eixo_real(escala=1.0):
    fig, eixo = plt.subplots(figsize=_fs(10, 6, escala))
    plotar_polos_zeros(eixo, polos, zeros)
    plotar_segmentos(eixo, segmentos, limite_x)
    finalizar_eixo(eixo, limite_x, limite_y, "Segmentos do eixo real no LGR", limites_usuario)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("**Definição:** pertencem ao LGR os pontos do eixo real a esquerda de um numero impar de polos e zeros reais.")
    if segmentos:
        for a, b in segmentos:
            texto_a = f"{a:.4f}" if np.isfinite(a) else r"-\infty"
            texto_b = f"{b:.4f}" if np.isfinite(b) else r"+\infty"
            st.latex(rf"\left[{texto_a}\;,\;{texto_b}\right]")
    else:
        st.info("Nenhum segmento do eixo real pertence ao LGR.")


def passo_5_lugares(escala=1.0):
    st.latex(rf"n_p = {len(polos)}, \quad n_z = {len(zeros)}")
    st.latex(rf"L_s = \max(n_p, n_z) = {n_lugares}")


def passo_6_simetria(escala=1.0):
    st.markdown("O LGR e **simetrico em relacao ao eixo real**: como os coeficientes do polinomio "
                "sao reais, raizes complexas sempre aparecem em pares conjugados.")


def passo_7_assintotas(escala=1.0):
    if centroide is not None:
        n_asc = len(polos) - len(zeros)
        st.latex(rf"n_a = n_p - n_z = {n_asc}")
        st.latex(r"\sigma_a = \frac{\sum \operatorname{Re}(p_i) - \sum \operatorname{Re}(z_j)}{n_p - n_z}")
        soma_p = np.sum(polos).real
        soma_z = np.sum(zeros).real if len(zeros) else 0.0
        st.latex(rf"\sigma_a = \frac{{{soma_p:.4f} - {soma_z:.4f}}}{{{n_asc}}} = {centroide:.4f}")
        for q, ang in enumerate(angulos_asc):
            st.latex(rf"q={q}: \;\; \phi_a = \frac{{(2\cdot{q}+1)\cdot180^\circ}}{{{n_asc}}} = {ang:.1f}^\circ")

        fig, eixo = plt.subplots(figsize=_fs(10, 6, escala))
        plotar_polos_zeros(eixo, polos, zeros)
        plotar_segmentos(eixo, segmentos, limite_x)
        plotar_assintotas(eixo, centroide, angulos_asc, limite_x, limite_y)
        eixo.plot(centroide, 0, "+", ms=12, mew=2, color=PALETA["destaque"],
                   label=rf"Centroide ({centroide:.2f})")
        finalizar_eixo(eixo, limite_x, limite_y, "Assintotas do LGR", limites_usuario)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.latex(r"n_p = n_z \;\Rightarrow\; \text{sem assintotas}")


def passo_8_breakaway(escala=1.0):
    derivada_num, derivada_den = np.polyder(num), np.polyder(den)
    equacao_display = np.polysub(np.convolve(num, derivada_den), np.convolve(den, derivada_num))
    st.latex(r"K = -\frac{D(s)}{N(s)} \qquad \frac{dK}{ds}=0 \;\Rightarrow\; D'(s)N(s) - D(s)N'(s) = 0")
    st.latex(rf"{latex_polinomio(equacao_display)} = 0")

    referencias = [p.real for p in polos if abs(p.imag) < 1e-8] + [z.real for z in zeros if abs(z.imag) < 1e-8]
    for raiz in np.roots(equacao_display):
        valor_num = np.polyval(num, raiz)
        ganho = -np.polyval(den, raiz) / valor_num if abs(valor_num) > 1e-12 else None
        if abs(raiz.imag) < 1e-6:
            posicao = raiz.real
            a_direita = sum(1 for v in referencias if v > posicao + 1e-10)
            no_lgr = a_direita % 2 == 1
            k_real = ganho.real if ganho is not None else np.inf
            status = r"\text{no LGR}" if no_lgr else r"\text{fora do LGR}"
            st.latex(rf"s = {posicao:.4f}, \; K = {k_real:.4f} \;\; [{status}]")
        elif ganho is not None and abs(ganho.imag) < 1e-6 and ganho.real > 0:
            st.latex(rf"s = {latex_complexo(raiz)}, \; K = {ganho.real:.4f} \;\; [\text{{no LGR}}]")

    fig, eixo = plt.subplots(figsize=_fs(10, 6, escala))
    plotar_polos_zeros(eixo, polos, zeros)
    plotar_segmentos(eixo, segmentos, limite_x)
    if centroide is not None:
        plotar_assintotas(eixo, centroide, angulos_asc, limite_x, limite_y)
    validos = [(s, k) for s, k in breakaways if not isinstance(s, complex)]
    if validos:
        eixo.plot([s for s, _ in validos], [0] * len(validos), "d", ms=10, mew=2,
                   color=PALETA["destaque"], label="Breakaway/Break-in", zorder=6)
        for s, k in validos:
            eixo.annotate(rf"$s={s:.2f}, K={k:.2f}$", (s, 0), xytext=(5, 5),
                           textcoords="offset points", fontsize=8, color=PALETA["destaque"])
    finalizar_eixo(eixo, limite_x, limite_y, "Pontos de descolamento", limites_usuario)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def passo_9_eixo_imaginario(escala=1.0):
    tabela, grau, colunas = routh["tabela"], routh["grau"], routh["colunas"]
    linhas_latex = []
    for i in range(grau + 1):
        celulas = [sympy.latex(sympy.simplify(tabela[i][j])) for j in range(colunas)]
        linhas_latex.append(f"s^{{{grau - i}}} & " + " & ".join(celulas))
    corpo_tabela = (r"\begin{array}{c|" + "c" * colunas + "}\n \\hline\n"
                     + (r" \\" + "\n").join(linhas_latex) + r" \\ \hline" + "\n\\end{array}")
    st.latex(corpo_tabela)

    if routh["condicoes"]:
        st.markdown("**Condicoes de estabilidade** (primeira coluna > 0):")
        for exp, expr, cond in routh["condicoes"]:
            if cond is not None:
                st.latex(rf"s^{{{exp}}}: {sympy.latex(expr)} > 0 \;\Rightarrow\; {cond}")
    if routh["ganhos_criticos"]:
        st.markdown("**Ganhos criticos:**")
        for k_crit in routh["ganhos_criticos"]:
            st.latex(rf"K_{{crit}} = {k_crit:.4f}")

    st.markdown("---")
    st.markdown("**Metodo alternativo** ($s=j\\omega$):")
    st.latex(rf"{latex_polinomio(info_jw['cruzamento'], variavel=r'\omega')} = 0")
    if cruzamentos:
        for k_val, w_val in cruzamentos:
            st.latex(rf"\omega = {w_val:.4f} \;\Rightarrow\; K = {k_val:.4f} \quad (s = \pm {w_val:.4f}j)")
    else:
        st.info("O LGR nao cruza o eixo imaginario para K > 0.")

    fig, eixo = plt.subplots(figsize=_fs(10, 6, escala))
    plotar_ramos_fundo(eixo, ramos_lgr)
    plotar_polos_zeros(eixo, polos, zeros)
    for k_val, w_val in cruzamentos:
        eixo.plot(0, w_val, "s", ms=10, color="cyan", markeredgecolor="navy", mew=2, zorder=6,
                   label=rf"$j\omega={w_val:.2f}$ (K={k_val:.2f})")
        eixo.plot(0, -w_val, "s", ms=10, color="cyan", markeredgecolor="navy", mew=2, zorder=6)
    finalizar_eixo(eixo, limite_x, limite_y, "Cruzamento com o eixo imaginario", limites_usuario)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def passo_10_angulos(escala=1.0):
    polos_cx = [p for p in polos if p.imag > 1e-8]
    zeros_cx = [z for z in zeros if z.imag > 1e-8]
    angulos_partida, angulos_chegada = {}, {}

    if polos_cx or zeros_cx:
        if polos_cx:
            st.markdown("### Angulos de partida")
            for pk in polos_cx:
                soma_p = sum(np.degrees(np.angle(pk - pj)) for pj in polos if abs(pj - pk) > 1e-10)
                soma_z = sum(np.degrees(np.angle(pk - zj)) for zj in zeros)
                theta = ((180.0 - soma_p + soma_z + 180) % 360) - 180
                st.latex(rf"p_k = {latex_complexo(pk)}: \;\; \theta_d = {theta % 360:.2f}^\circ")
                angulos_partida[pk] = theta

        if zeros_cx:
            st.markdown("### Angulos de chegada")
            for zk in zeros_cx:
                soma_z = sum(np.degrees(np.angle(zk - zj)) for zj in zeros if abs(zj - zk) > 1e-10)
                soma_p = sum(np.degrees(np.angle(zk - pj)) for pj in polos)
                theta = ((180.0 - soma_z + soma_p + 180) % 360) - 180
                st.latex(rf"z_k = {latex_complexo(zk)}: \;\; \theta_a = {theta % 360:.2f}^\circ")
                angulos_chegada[zk] = theta

        fig, eixo = plt.subplots(figsize=_fs(10, 7, escala))
        plotar_ramos_fundo(eixo, ramos_lgr)
        plotar_polos_zeros(eixo, polos, zeros)
        conjunto = np.concatenate([polos, zeros]) if len(zeros) > 0 else polos
        comprimento_seta = max(np.ptp(conjunto.real), np.ptp(conjunto.imag), 1.0) * 0.3
        for pk, theta in angulos_partida.items():
            dx, dy = comprimento_seta * np.cos(np.radians(theta)), comprimento_seta * np.sin(np.radians(theta))
            eixo.annotate("", xy=(pk.real + dx, pk.imag + dy), xytext=(pk.real, pk.imag),
                           arrowprops=dict(arrowstyle="->", color=PALETA["polo"], lw=2))
        for zk, theta in angulos_chegada.items():
            dx, dy = comprimento_seta * np.cos(np.radians(theta)), comprimento_seta * np.sin(np.radians(theta))
            eixo.annotate("", xy=(zk.real + dx, zk.imag + dy), xytext=(zk.real, zk.imag),
                           arrowprops=dict(arrowstyle="->", color=PALETA["zero"], lw=2))
        finalizar_eixo(eixo, limite_x, limite_y, "Angulos de partida/chegada", limites_usuario)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Sem polos/zeros complexos — este passo nao se aplica.")


def passo_11_criterio_angulo(escala=1.0):
    st.latex(r"\sum \angle(s_0 - z_j) - \sum \angle(s_0 - p_i) = \pm 180^\circ(2q+1)")
    st.markdown(f"**Ponto de teste:** $s_0 = {latex_complexo(ponto_teste)}$")
    st.latex(rf"\Delta\theta = {delta_angulo:.2f}^\circ \;\to\; \text{{normalizado}} = {delta_normalizado % 360:.2f}^\circ")
    if pertence:
        st.success(f"O ponto **pertence** ao LGR.")
    else:
        desvio = abs(abs(delta_normalizado) - 180)
        st.warning(
            f"O ponto **nao pertence** ao LGR: pelo criterio do angulo, a soma "
            f"$\\Delta\\theta$ precisa ser $\\pm180^\\circ$ (multiplo impar), mas aqui "
            f"$\\Delta\\theta = {delta_normalizado:.2f}^\\circ$, um desvio de "
            f"**{desvio:.2f}°** em relacao a 180°."
        )

    fig, eixo = plt.subplots(figsize=_fs(10, 6, escala))
    plotar_polos_zeros(eixo, polos, zeros)
    cor = PALETA["aceito"] if pertence else PALETA["rejeitado"]
    marcador = "*" if pertence else "X"
    eixo.plot(ponto_teste.real, ponto_teste.imag, marcador, ms=14, color=cor,
               markeredgecolor="black", label=f"s0 = {latex_complexo(ponto_teste)}", zorder=6)
    for p in polos:
        eixo.plot([p.real, ponto_teste.real], [p.imag, ponto_teste.imag], ":", color=PALETA["polo"], alpha=0.4)
    for z in zeros:
        eixo.plot([z.real, ponto_teste.real], [z.imag, ponto_teste.imag], ":", color=PALETA["zero"], alpha=0.4)
    todos_x = list(polos.real) + [ponto_teste.real] + (list(zeros.real) if len(zeros) else [])
    todos_y = list(abs(polos.imag)) + [abs(ponto_teste.imag)] + (list(abs(zeros.imag)) if len(zeros) else [])
    finalizar_eixo(eixo, (min(todos_x) - 2, max(todos_x) + 2), max(todos_y) + 2, "Criterio de angulo", limites_usuario)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def passo_12_calculo_k(escala=1.0):
    st.latex(r"K = \frac{\prod_i |s_0 - p_i|}{\prod_j |s_0 - z_j|}")
    produto_polos = np.prod([abs(ponto_teste - p) for p in polos]) if len(polos) else 1.0
    produto_zeros = np.prod([abs(ponto_teste - z) for z in zeros]) if len(zeros) else 1.0
    st.latex(rf"K = \frac{{{produto_polos:.4f}}}{{{produto_zeros:.4f}}} = {ganho_calculado:.4f}")

    coluna_a, coluna_b = st.columns(2)
    coluna_a.metric("K neste ponto", f"{ganho_calculado:.4f}")
    coluna_b.metric("Pertence ao LGR?", "Sim" if pertence else "Nao")

    if pertence:
        st.success(rf"O ponto pertence ao LGR. $K = {ganho_calculado:.6f}$")
    else:
        st.warning(rf"O ponto nao pertence ao LGR. $K = {ganho_calculado:.6f}$ (valor de referencia)")


def grafico_lgr_completo(escala=1.0):
    fig, eixo = plt.subplots(figsize=_fs(6, 4.2, escala))
    for coluna in range(ramos_lgr.shape[1]):
        ramo = ramos_lgr[:, coluna]
        eixo.plot(ramo.real, ramo.imag, "-", color=PALETA["lgr"], linewidth=2, alpha=0.85)
    plotar_polos_zeros(eixo, polos, zeros)
    if centroide is not None:
        eixo.plot(centroide, 0, "+", ms=10, mew=2, color=PALETA["destaque"], label=f"Centroide ({centroide:.2f})")
    finalizar_eixo(eixo, limite_x, limite_y, "Lugar Geometrico das Raizes", limites_usuario)
    fig.tight_layout()
    return fig


PASSOS = [
    ("1. Equacao Caracteristica", passo_1_equacao),
    ("2. Forma Fatorada", passo_2_fatorada),
    ("3. Polos e Zeros", passo_3_polos_zeros),
    ("4. Eixo Real", passo_4_eixo_real),
    ("5. Numero de Lugares", passo_5_lugares),
    ("6. Simetria", passo_6_simetria),
    ("7. Assintotas", passo_7_assintotas),
    ("8. Pontos de Saída/Entrada", passo_8_breakaway),
    ("9. Eixo Imaginario", passo_9_eixo_imaginario),
    ("10. Angulos de Partida/Chegada", passo_10_angulos),
    ("11. Criterio de Angulo", passo_11_criterio_angulo),
    ("12. Calculo de K", passo_12_calculo_k),
]

titulos_abas = [
    "1. Eq. Caracteristica", "2. Forma Fatorada", "3. Polos/Zeros", "4. Eixo Real",
    "5. Lugares", "6. Simetria", "7. Assintotas", "8. P. de Saída",
    "9. Eixo Imaginario", "10. Angulos P/C", "11. Criterio Angulo", "12. Calculo de K",
    "📄 Relatório Completo",
]
abas = st.tabs(titulos_abas)

for aba, (_, funcao) in zip(abas[:12], PASSOS):
    with aba:
        funcao()

with abas[12]:
    st.caption("Use Ctrl+P / Cmd+P (ou Arquivo > Imprimir) para gerar um PDF ou imprimir esta pagina inteira.")
    for titulo, funcao in PASSOS:
        st.markdown(f"#### {titulo}")
        funcao(escala=1.0)
        st.markdown("---")

st.markdown("---")
st.subheader("Grafico completo do LGR")
coluna_grafico, _ = st.columns([2, 1])
with coluna_grafico:
    fig_completo = grafico_lgr_completo(escala=1.0)
    st.pyplot(fig_completo, width="content")
    plt.close(fig_completo)
