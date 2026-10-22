"""
Taller final Minería de Datos - Segmentación de clientes con K-Means
Aplicación interactiva en Streamlit.

Ejecutar desde Anaconda Prompt, dentro de la carpeta del taller:
    streamlit run app.py --theme.base light
"""

import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Segmentación de clientes", page_icon="🛍️", layout="wide")

# --------------------------------------------------------------------------
# Identidad visual
# --------------------------------------------------------------------------
TINTA = "#1F2547"
GRIS = "#6A7092"
ACENTO = "#4C5BD4"
FUENTE = "Manrope, sans-serif"

COLORES = {
    "Clientes de alto valor": "#FF6FA5",
    "Clientes con potencial": "#9B7BFF",
    "Clientes promedio": "#6FA0FF",
    "Clientes impulsivos": "#3FBF86",
    "Clientes de bajo valor": "#F2AE2E",
}
ICONOS = {
    "Clientes de alto valor": "💎",
    "Clientes con potencial": "🚀",
    "Clientes promedio": "⚖️",
    "Clientes impulsivos": "⚡",
    "Clientes de bajo valor": "🌱",
}
DESCRIPCION = {
    "Clientes de alto valor": "Ganan mucho y gastan mucho. Son los clientes más rentables del centro comercial.",
    "Clientes con potencial": "Ganan mucho pero gastan poco. Tienen capacidad de compra que hoy no se aprovecha.",
    "Clientes promedio": "Ingreso y gasto medios. Es el grupo más grande y el más estable.",
    "Clientes impulsivos": "Ganan poco pero gastan mucho. Suelen ser jóvenes y siguen las tendencias.",
    "Clientes de bajo valor": "Ganan poco y gastan poco. Compran con mucha cautela.",
}
ESTRATEGIAS = {
    "Clientes de alto valor": ["Programa VIP de fidelización", "Acceso anticipado a lanzamientos",
                               "Atención personalizada y productos premium"],
    "Clientes con potencial": ["Experiencias exclusivas y personalizadas", "Productos de mayor gama",
                               "Encuestas para entender por qué compran poco"],
    "Clientes promedio": ["Promociones por volumen (2x1, combos)", "Programa de puntos",
                          "Venta cruzada para subir el ticket promedio"],
    "Clientes impulsivos": ["Campañas de tendencia en redes sociales", "Pago a cuotas",
                            "Productos de moda a precio accesible"],
    "Clientes de bajo valor": ["Campañas de bajo costo por correo y redes", "Descuentos en productos básicos",
                               "Ofertas de temporada"],
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800&family=Manrope:wght@400;500;600;700&display=swap');

html, body, .stApp, .stApp p, .stApp label, .stApp li, .stApp span, .stApp div {{
    font-family: 'Manrope', sans-serif;
}}
.stApp {{ background: #F4F6FB; color: {TINTA}; }}
.stApp p, .stApp label, .stApp li {{ color: {TINTA}; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px; }}
h1, h2, h3, .titulo {{ font-family: 'Bricolage Grotesque', sans-serif !important; color: {TINTA} !important; }}

/* Encabezado */
.hero {{ display: flex; gap: 2.5rem; align-items: stretch; flex-wrap: wrap; margin-bottom: 1.8rem; }}
.hero-texto {{ flex: 1 1 380px; padding-top: .4rem; }}
.hero-texto h1 {{ font-size: 3.1rem; line-height: 1.02; font-weight: 800; margin: 0 0 1rem 0; letter-spacing: -0.02em; }}
.hero-texto p {{ font-size: 1.05rem; color: {GRIS} !important; max-width: 46ch; line-height: 1.6; margin: 0; }}
.hero-datos {{ display: flex; gap: 1.6rem; margin-top: 1.6rem; flex-wrap: wrap; }}
.dato b {{ display: block; font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.7rem; color: {TINTA}; }}
.dato span {{ color: {GRIS}; font-size: .88rem; }}

/* Directorio de segmentos, como el tablero de un centro comercial */
.directorio {{ flex: 0 1 400px; background: {TINTA}; border-radius: 22px; padding: 1.3rem 1.4rem; }}
.directorio-titulo {{ color: #C9CDF0; font-size: .9rem; font-weight: 600; margin-bottom: .7rem; }}
.fila {{ display: flex; align-items: center; gap: .8rem; padding: .62rem 0; border-top: 1px solid rgba(255,255,255,.08); }}
.fila:first-of-type {{ border-top: none; }}
.punto {{ width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }}
.fila-nombre {{ color: #FFFFFF; font-weight: 600; flex: 1; font-size: .95rem; }}
.fila-num {{ color: #C9CDF0; font-size: .88rem; }}
.barra {{ height: 6px; border-radius: 6px; background: rgba(255,255,255,.08); width: 70px; overflow: hidden; }}
.barra div {{ height: 100%; border-radius: 6px; }}

/* Pestañas */
.stTabs [data-baseweb="tab-list"] {{ gap: .3rem; background: #FFFFFF; padding: .35rem; border-radius: 14px; border: 1px solid #E3E6F3; width: fit-content; }}
.stTabs [data-baseweb="tab"] {{ border-radius: 10px; padding: .55rem 1.1rem; background: transparent; height: auto; }}
.stTabs [data-baseweb="tab"] p {{ font-weight: 600; color: {GRIS}; font-size: .95rem; }}
.stTabs [aria-selected="true"] {{ background: #ECEEFC; }}
.stTabs [aria-selected="true"] p {{ color: {ACENTO}; }}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display: none; }}

/* Contenedores con borde */
div[data-testid="stVerticalBlockBorderWrapper"] {{ background: #FFFFFF; border-radius: 18px !important; border: 1px solid #E3E6F3 !important; }}

.nota {{ background: #ECEEFC; border-radius: 14px; padding: 1rem 1.2rem; color: {TINTA}; line-height: 1.6; margin-top: .6rem; }}

/* Tarjetas de perfil */
.perfil {{ background: #FFFFFF; border-radius: 18px; padding: 1.2rem 1.1rem; border: 1px solid #E3E6F3; height: 100%; position: relative; overflow: hidden; }}
.perfil::before {{ content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 5px; background: var(--c); }}
.perfil-icono {{ font-size: 1.5rem; margin-bottom: .5rem; }}
.perfil-nombre {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 600; font-size: 1.08rem; color: {TINTA}; margin-bottom: .35rem; }}
.perfil-desc {{ color: {GRIS}; font-size: .86rem; line-height: 1.5; min-height: 4.2em; }}
.perfil-cifras {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: .3rem; margin-top: .9rem; padding-top: .8rem; border-top: 1px dashed #E3E6F3; }}
.perfil-cifras b {{ display: block; font-size: 1.05rem; color: {TINTA}; }}
.perfil-cifras span {{ font-size: .75rem; color: {GRIS}; }}

/* Formulario y botón */
.stButton > button {{ width: 100%; background: {ACENTO}; color: #FFFFFF; border: none; border-radius: 12px; padding: .75rem 1rem; font-weight: 700; font-size: 1rem; }}
.stButton > button:hover {{ background: #3C4AC0; color: #FFFFFF; }}
.stButton > button:focus-visible {{ outline: 3px solid #A9B1F5; outline-offset: 2px; }}

/* Resultado de la clasificación */
.resultado {{ border-radius: 18px; padding: 1.3rem 1.3rem 1.1rem; background: #FFFFFF; border: 2px solid var(--c); margin-top: 1rem; }}
.res-cabeza {{ display: flex; gap: .9rem; align-items: center; }}
.res-icono {{ font-size: 1.7rem; width: 3rem; height: 3rem; border-radius: 14px; display: flex; align-items: center; justify-content: center; }}
.res-sub {{ color: {GRIS}; font-size: .88rem; }}
.res-perfil {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: 1.45rem; color: {TINTA}; line-height: 1.15; }}
.res-desc {{ color: {TINTA}; margin: .9rem 0 .7rem; line-height: 1.55; font-size: .95rem; }}
.res-lista {{ margin: 0; padding-left: 1.1rem; }}
.res-lista li {{ margin: .2rem 0; font-size: .92rem; }}
.vacio {{ border: 2px dashed #D5D9EE; border-radius: 18px; padding: 1.4rem; color: {GRIS}; text-align: center; margin-top: 1rem; line-height: 1.6; }}

@media (max-width: 700px) {{ .hero-texto h1 {{ font-size: 2.3rem; }} }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Carga del modelo y de los datos generados en el notebook
# --------------------------------------------------------------------------
VARS = ["Ingreso_Anual", "Puntaje_Gasto"]


@st.cache_resource
def cargar_modelo():
    return (joblib.load("modelo_kmeans.pkl"),
            joblib.load("scaler.pkl"),
            joblib.load("perfiles.pkl"))


@st.cache_data
def cargar_datos():
    return pd.read_csv("clientes_segmentados.csv")


@st.cache_data
def calcular_metricas(X_scaled):
    ks = list(range(1, 11))
    inercias, siluetas = [], {}
    for k in ks:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42).fit(X_scaled)
        inercias.append(km.inertia_)
        if k >= 2:
            siluetas[k] = silhouette_score(X_scaled, km.labels_)
    return ks, inercias, siluetas


try:
    kmeans, scaler, perfiles = cargar_modelo()
    df = cargar_datos()
except FileNotFoundError as e:
    st.error(f"Falta el archivo **{e.filename}**. Ejecuta el notebook completo y guarda app.py "
             "en la misma carpeta que los archivos .pkl y .csv.")
    st.stop()

X_scaled = scaler.transform(df[VARS])
K_OPT = kmeans.n_clusters
centroides = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=VARS)
centroides["Perfil"] = [perfiles[i] for i in range(K_OPT)]
silueta_modelo = silhouette_score(X_scaled, df["Cluster"])
conteo = df["Perfil"].value_counts()


def estilo(fig, alto=420):
    fig.update_layout(
        template="plotly_white", height=alto,
        font=dict(family=FUENTE, color=TINTA, size=13),
        title_font=dict(family="Bricolage Grotesque, sans-serif", size=18),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=55, b=10),
        hoverlabel=dict(font_family=FUENTE, bgcolor="white"),
        legend=dict(orientation="h", yanchor="top", y=-0.15, x=0),
    )
    fig.update_xaxes(gridcolor="#EEF0F7", zeroline=False, linecolor="#E3E6F3")
    fig.update_yaxes(gridcolor="#EEF0F7", zeroline=False, linecolor="#E3E6F3")
    return fig


def grafico_clusters(nuevo=None):
    fig = go.Figure()
    for perfil, color in COLORES.items():
        g = df[df["Perfil"] == perfil]
        fig.add_trace(go.Scatter(
            x=g["Ingreso_Anual"], y=g["Puntaje_Gasto"], mode="markers", name=perfil,
            marker=dict(size=11, color=color, opacity=.85, line=dict(width=1.5, color="white")),
            hovertemplate="Ingreso: %{x}k$<br>Gasto: %{y}<extra>" + perfil + "</extra>"))
    fig.add_trace(go.Scatter(
        x=centroides["Ingreso_Anual"], y=centroides["Puntaje_Gasto"], mode="markers", name="Centroides",
        text=centroides["Perfil"],
        marker=dict(symbol="x-thin", size=18, color=TINTA, line=dict(width=4, color=TINTA)),
        hovertemplate="<b>Centroide</b><br>%{text}<br>Ingreso: %{x:.1f}k$<br>Gasto: %{y:.1f}<extra></extra>"))
    if nuevo is not None:
        fig.add_trace(go.Scatter(
            x=[nuevo[0]], y=[nuevo[1]], mode="markers", name="Nuevo cliente",
            marker=dict(symbol="star", size=28, color="#FF4D6D", line=dict(width=2, color="white")),
            hovertemplate="<b>Nuevo cliente</b><br>Ingreso: %{x}k$<br>Gasto: %{y}<extra></extra>"))
    fig.update_layout(xaxis_title="Ingreso anual (k$)", yaxis_title="Puntaje de gasto (1-100)")
    return estilo(fig, 540)


# --------------------------------------------------------------------------
# Encabezado
# --------------------------------------------------------------------------
filas = ""
for perfil, color in COLORES.items():
    n = int(conteo.get(perfil, 0))
    ancho = round(n / conteo.max() * 100)
    filas += (f'<div class="fila"><div class="punto" style="background:{color}"></div>'
              f'<div class="fila-nombre">{perfil}</div>'
              f'<div class="barra"><div style="width:{ancho}%;background:{color}"></div></div>'
              f'<div class="fila-num">{n}</div></div>')

st.markdown(
    '<div class="hero">'
    '<div class="hero-texto">'
    '<h1>¿Quién compra en el centro comercial?</h1>'
    f'<p>El modelo K-Means agrupó a {len(df)} clientes según su ingreso anual y su puntaje de gasto. '
    'Explora los grupos y clasifica a un cliente nuevo en segundos.</p>'
    '<div class="hero-datos">'
    f'<div class="dato"><b>{len(df)}</b><span>clientes analizados</span></div>'
    f'<div class="dato"><b>{K_OPT}</b><span>segmentos encontrados</span></div>'
    f'<div class="dato"><b>{silueta_modelo:.3f}</b><span>coeficiente de silueta</span></div>'
    '</div></div>'
    f'<div class="directorio"><div class="directorio-titulo">Directorio de segmentos</div>{filas}</div>'
    '</div>',
    unsafe_allow_html=True)

tab_codo, tab_clusters, tab_clasificar = st.tabs(
    ["📈 Método del codo", "🎯 Clusters", "🧭 Clasificar cliente"])

# --------------------------------------------------------------------------
# Pestaña 1: método del codo y silueta
# --------------------------------------------------------------------------
with tab_codo:
    ks, inercias, siluetas = calcular_metricas(X_scaled)
    col_a, col_b = st.columns(2, gap="medium")

    with col_a:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ks, y=inercias, mode="lines+markers",
            line=dict(color=ACENTO, width=3),
            marker=dict(size=10, color="white", line=dict(width=3, color=ACENTO)),
            hovertemplate="K = %{x}<br>Inercia = %{y:.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=[K_OPT], y=[inercias[K_OPT - 1]], mode="markers", hoverinfo="skip",
            marker=dict(size=16, color="#FF4D6D", line=dict(width=3, color="white"))))
        fig.add_annotation(x=K_OPT, y=inercias[K_OPT - 1], text=f"Codo en K = {K_OPT}",
                           showarrow=True, arrowhead=2, ax=70, ay=-70, arrowcolor="#FF4D6D",
                           font=dict(color="#FF4D6D", size=14))
        fig.update_layout(title="Método del codo", xaxis_title="Número de clusters (K)",
                          yaxis_title="Inercia (WCSS)", showlegend=False)
        fig.update_xaxes(dtick=1)
        with st.container(border=True):
            st.plotly_chart(estilo(fig), config={"displayModeBar": False}, key="graf_codo")

    with col_b:
        k_sil, valores = list(siluetas.keys()), list(siluetas.values())
        mejor = max(siluetas, key=siluetas.get)
        fig2 = go.Figure(go.Bar(
            x=k_sil, y=valores, text=[f"{v:.3f}" for v in valores], textposition="outside",
            marker=dict(color=["#FF4D6D" if k == mejor else "#C9D2FB" for k in k_sil],
                        cornerradius=8),
            hovertemplate="K = %{x}<br>Silueta = %{y:.4f}<extra></extra>"))
        fig2.update_layout(title=f"Coeficiente de silueta (mejor K = {mejor})",
                           xaxis_title="Número de clusters (K)", yaxis_title="Silueta",
                           yaxis_range=[0, max(valores) + 0.1], showlegend=False)
        fig2.update_xaxes(dtick=1)
        with st.container(border=True):
            st.plotly_chart(estilo(fig2), config={"displayModeBar": False}, key="graf_silueta")

    reduccion_antes = inercias[K_OPT - 2] - inercias[K_OPT - 1]
    reduccion_despues = inercias[K_OPT - 1] - inercias[K_OPT]
    st.markdown(
        f'<div class="nota">💡 Hasta K = {K_OPT} la inercia cae con fuerza '
        f'(de K = {K_OPT - 1} a K = {K_OPT} baja {reduccion_antes:.1f}), y después casi se estanca '
        f'(de K = {K_OPT} a K = {K_OPT + 1} solo baja {reduccion_despues:.1f}). '
        f'La silueta más alta también aparece en K = {mejor}, así que ambos métodos apuntan '
        f'a {K_OPT} segmentos.</div>',
        unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Pestaña 2: clusters y perfiles
# --------------------------------------------------------------------------
with tab_clusters:
    with st.container(border=True):
        st.plotly_chart(grafico_clusters(), config={"displayModeBar": False}, key="graf_clusters")

    st.markdown('<h3 class="titulo">Perfiles de clientes</h3>', unsafe_allow_html=True)
    resumen = df.groupby("Perfil").agg(
        Clientes=("Cluster", "count"), Edad=("Edad", "mean"),
        Ingreso=("Ingreso_Anual", "mean"), Gasto=("Puntaje_Gasto", "mean"))

    columnas = st.columns(len(COLORES), gap="small")
    for col, (perfil, color) in zip(columnas, COLORES.items()):
        if perfil not in resumen.index:
            continue
        r = resumen.loc[perfil]
        col.markdown(
            f'<div class="perfil" style="--c:{color}">'
            f'<div class="perfil-icono">{ICONOS[perfil]}</div>'
            f'<div class="perfil-nombre">{perfil}</div>'
            f'<div class="perfil-desc">{DESCRIPCION[perfil]}</div>'
            '<div class="perfil-cifras">'
            f'<div><b>{int(r.Clientes)}</b><span>clientes</span></div>'
            f'<div><b>{r.Ingreso:.0f}k</b><span>ingreso</span></div>'
            f'<div><b>{r.Gasto:.0f}</b><span>gasto</span></div>'
            '</div></div>',
            unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Pestaña 3: clasificar un cliente nuevo
# --------------------------------------------------------------------------
with tab_clasificar:
    izq, der = st.columns([1, 1.7], gap="large")

    with izq:
        with st.container(border=True):
            st.markdown('<h3 class="titulo" style="margin-top:0">Datos del cliente</h3>',
                        unsafe_allow_html=True)
            ingreso = st.number_input("Ingreso anual (miles de dólares)", min_value=1,
                                      max_value=300, value=60, step=1)
            gasto = st.slider("Puntaje de gasto", min_value=1, max_value=100, value=50,
                              help="Puntaje que asigna el centro comercial según el "
                                   "comportamiento de compra (1 = gasta muy poco, 100 = gasta mucho).")
            if st.button("Clasificar cliente"):
                nuevo = pd.DataFrame([[ingreso, gasto]], columns=VARS)
                cluster = int(kmeans.predict(scaler.transform(nuevo))[0])
                st.session_state["resultado"] = (ingreso, gasto, cluster)

        resultado = st.session_state.get("resultado")
        if resultado:
            ing, gas, cluster = resultado
            perfil = perfiles[cluster]
            color = COLORES[perfil]
            lista = "".join(f"<li>{e}</li>" for e in ESTRATEGIAS[perfil])
            st.markdown(
                f'<div class="resultado" style="--c:{color}">'
                '<div class="res-cabeza">'
                f'<div class="res-icono" style="background:{color}2E">{ICONOS[perfil]}</div>'
                f'<div><div class="res-sub">Cliente clasificado en el cluster {cluster}</div>'
                f'<div class="res-perfil">{perfil}</div></div></div>'
                f'<p class="res-desc">{DESCRIPCION[perfil]}</p>'
                '<div class="res-sub">Estrategias recomendadas</div>'
                f'<ul class="res-lista">{lista}</ul></div>',
                unsafe_allow_html=True)
            ing_min, ing_max = df["Ingreso_Anual"].min(), df["Ingreso_Anual"].max()
            if not ing_min <= ing <= ing_max:
                st.warning(f"El ingreso ingresado está fuera del rango de los datos de entrenamiento "
                           f"({ing_min}k a {ing_max}k). Toma la clasificación con precaución.")
        else:
            st.markdown('<div class="vacio">Ingresa el ingreso y el puntaje de gasto, y presiona '
                        '<b>Clasificar cliente</b> para ver a qué segmento pertenece.</div>',
                        unsafe_allow_html=True)

    with der:
        with st.container(border=True):
            punto = (resultado[0], resultado[1]) if resultado else None
            st.plotly_chart(grafico_clusters(punto), config={"displayModeBar": False}, key="graf_nuevo")
