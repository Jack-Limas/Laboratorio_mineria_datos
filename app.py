from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Segmentación de Clientes - K-Means", page_icon="📊", layout="wide")

st.title("📊 Segmentación de Clientes con K-Means")
st.markdown("Aplicación interactiva para analizar grupos de clientes y clasificar nuevos perfiles de negocio.")

FEATURES = ['Annual Income (k$)', 'Spending Score (1-100)']
DATA_PATH = Path(__file__).parent / 'Mall_Customers.csv'

ESTRATEGIAS = {
    'Alto valor (VIP)': "Fidelización: programa de lealtad premium, preventas, eventos exclusivos y atención personalizada.",
    'Con potencial (alto ingreso, bajo gasto)': "Activación: ofertas personalizadas basadas en calidad, marcas premium y servicios de conveniencia.",
    'Promedio': "Aumentar el ticket: venta cruzada, combos y programa de puntos para subir de nivel.",
    'Impulsivos (bajo ingreso, alto gasto)': "Mantener el gasto: promociones flash, redes sociales, pagos a plazos y marcas asequibles.",
    'Bajo valor': "Eficiencia: comunicación de bajo costo, descuentos por volumen y temporadas de liquidación.",
}


def asignar_perfil(ingreso_z, gasto_z, umbral=0.5):
    """Asigna un perfil de negocio a partir del centroide escalado (z-scores)."""
    alto_ing, bajo_ing = ingreso_z > umbral, ingreso_z < -umbral
    alto_gas, bajo_gas = gasto_z > umbral, gasto_z < -umbral
    if alto_ing and alto_gas:
        return 'Alto valor (VIP)'
    if alto_ing and bajo_gas:
        return 'Con potencial (alto ingreso, bajo gasto)'
    if bajo_ing and alto_gas:
        return 'Impulsivos (bajo ingreso, alto gasto)'
    if bajo_ing and bajo_gas:
        return 'Bajo valor'
    return 'Promedio'


# 1. Carga de datos
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def evaluar_k(X_scaled):
    wcss, sil = [], []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10).fit(X_scaled)
        wcss.append(km.inertia_)
        if k >= 2:
            sil.append(silhouette_score(X_scaled, km.labels_))
    return wcss, sil


df = load_data()

# Preparación y escalado
X = df[FEATURES].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
wcss, sil_scores = evaluar_k(X_scaled)
k_sugerido = int(np.argmax(sil_scores)) + 2

# Menú lateral
st.sidebar.header("Opciones de Configuración")
k_selected = st.sidebar.slider("Seleccione el valor de K (Clusters):", min_value=2, max_value=10, value=k_sugerido)
st.sidebar.info(f"K óptimo sugerido (máxima silueta): **{k_sugerido}**")
st.sidebar.metric("Silueta con el K elegido", f"{sil_scores[k_selected - 2]:.3f}")
st.sidebar.metric("Inercia (WCSS)", f"{wcss[k_selected - 1]:.1f}")

# Modelo K-Means
kmeans = KMeans(n_clusters=k_selected, init='k-means++', random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)
centroids_scaled = kmeans.cluster_centers_
centroids_original = scaler.inverse_transform(centroids_scaled)
perfiles = {i: asignar_perfil(*c) for i, c in enumerate(centroids_scaled)}
df['Perfil'] = df['Cluster'].map(perfiles)

# Pestañas principales
tab1, tab2, tab3 = st.tabs(["📉 Evaluación (Codo y Silueta)", "📌 Visualización de Clusters", "🔮 Clasificar Nuevo Cliente"])

with tab1:
    st.subheader("Evaluación del Número Óptimo de Clusters")
    col1, col2 = st.columns(2)

    with col1:
        # Gráfica del codo
        fig_elbow, ax_elbow = plt.subplots(figsize=(5, 3.5))
        ax_elbow.plot(range(1, 11), wcss, marker='o', linestyle='--', color='b')
        ax_elbow.axvline(k_selected, color='red', linestyle=':', label=f'K elegido = {k_selected}')
        ax_elbow.set_title("Método del Codo")
        ax_elbow.set_xlabel("K")
        ax_elbow.set_ylabel("WCSS")
        ax_elbow.set_xticks(range(1, 11))
        ax_elbow.legend()
        st.pyplot(fig_elbow)

    with col2:
        # Gráfica de Silueta
        fig_sil, ax_sil = plt.subplots(figsize=(5, 3.5))
        ax_sil.plot(range(2, 11), sil_scores, marker='s', color='green')
        ax_sil.axvline(k_selected, color='red', linestyle=':', label=f'K elegido = {k_selected}')
        ax_sil.set_title("Coeficiente de Silueta")
        ax_sil.set_xlabel("K")
        ax_sil.set_ylabel("Score")
        ax_sil.set_xticks(range(2, 11))
        ax_sil.legend()
        st.pyplot(fig_sil)

    st.markdown(
        f"El **codo** de la curva y el **máximo de la silueta** coinciden en **K = {k_sugerido}**: "
        "a partir de ese valor, agregar clusters casi no reduce la inercia y la silueta empieza a bajar."
    )

with tab2:
    st.subheader(f"Visualización de Clusters para K = {k_selected}")
    fig_cluster, ax_cluster = plt.subplots(figsize=(9, 5))
    sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='Set1', s=70, ax=ax_cluster)
    ax_cluster.scatter(centroids_original[:, 0], centroids_original[:, 1], s=200, c='black', marker='X', label='Centroides')
    ax_cluster.set_title("Segmentación del Mercado")
    ax_cluster.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    st.pyplot(fig_cluster)

    st.markdown("#### Resumen por cluster")
    resumen = df.groupby('Cluster').agg(
        Perfil=('Perfil', 'first'),
        Clientes=('CustomerID', 'count'),
        Edad_media=('Age', 'mean'),
        Ingreso_medio=('Annual Income (k$)', 'mean'),
        Gasto_medio=('Spending Score (1-100)', 'mean'),
    ).round(1)
    st.dataframe(resumen, width='stretch')

    with st.expander("Ver datos con el cluster asignado"):
        st.dataframe(df, width='stretch')

with tab3:
    st.subheader("Ingresar Datos para Clasificar Nuevo Cliente")
    col_input1, col_input2 = st.columns(2)

    with col_input1:
        nuevo_ingreso = st.number_input("Ingreso Anual (k$):", min_value=10.0, max_value=200.0, value=60.0)
    with col_input2:
        nuevo_gasto = st.number_input("Puntaje de Gasto (1-100):", min_value=1.0, max_value=100.0, value=50.0)

    if st.button("Clasificar Cliente"):
        # Escalar la entrada del nuevo cliente con el mismo scaler del entrenamiento
        nuevo_dato_scaled = scaler.transform([[nuevo_ingreso, nuevo_gasto]])
        cluster_predicho = int(kmeans.predict(nuevo_dato_scaled)[0])
        perfil = perfiles[cluster_predicho]

        st.success(f"El cliente ingresado pertenece al **Cluster #{cluster_predicho}**: **{perfil}**")
        st.info(f"Estrategia recomendada: {ESTRATEGIAS[perfil]}")

        fig_new, ax_new = plt.subplots(figsize=(9, 5))
        sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='Set1', s=60, alpha=0.6, ax=ax_new)
        ax_new.scatter(centroids_original[:, 0], centroids_original[:, 1], s=200, c='black', marker='X', label='Centroides')
        ax_new.scatter(nuevo_ingreso, nuevo_gasto, s=350, c='gold', marker='*', edgecolors='black', label='Nuevo cliente')
        ax_new.set_title("Ubicación del nuevo cliente")
        ax_new.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        st.pyplot(fig_new)
