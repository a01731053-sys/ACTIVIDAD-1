#Creamos el archivo de la APP en el interprete principal (Phyton)
#####################################################
#Importamos librerias
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
import plotly.express as px

# =========================================================
# CONFIGURACIÓN DEL DASHBOARD
# =========================================================

st.set_page_config(
    page_title="Análisis Univariado GAC",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# CARGA DE DATOS
# =========================================================

bitacora = pd.read_csv("BITACORA DE PISO.csv")


# =========================================================
# LIMPIEZA DE NOMBRES DE COLUMNAS
# =========================================================

# Quitar espacios al inicio o al final de los nombres
bitacora.columns = bitacora.columns.str.strip()


# =========================================================
# PREPROCESAMIENTO
# =========================================================

# Eliminar columnas completamente vacías
bitacora = bitacora.dropna(axis=1, how="all")

# Eliminar registros duplicados
bitacora = bitacora.drop_duplicates()


# =========================================================
# TRATAMIENTO DE NULOS
# =========================================================

# En Asesor, los registros sin asesor se conservan como categoría
bitacora["Asesor"] = bitacora["Asesor"].fillna(
    "Sin asesor"
)

# En SDC, los nulos no se consideran automáticamente como "No"
bitacora["SDC"] = bitacora["SDC"].fillna(
    "Sin información"
)


# =========================================================
# TRANSFORMACIÓN DE SDC
# =========================================================

bitacora["SDC"] = bitacora["SDC"].replace({
    True: "Sí solicitó crédito",
    False: "No solicitó crédito"
})


# =========================================================
# TRANSFORMACIÓN DE FECHA
# =========================================================

# Convertir Fecha a formato de fecha
bitacora["Fecha"] = pd.to_datetime(
    bitacora["Fecha"],
    format="mixed",
    errors="coerce"
)

# Para el análisis mensual solo utilizamos fechas válidas
bitacora_fechas = bitacora.dropna(
    subset=["Fecha"]
).copy()


# Crear número de mes
bitacora_fechas["Mes_Numero"] = (
    bitacora_fechas["Fecha"].dt.month
)


# Diccionario para poner los meses en español
meses = {
    1: "Enero",
    2: "Febrero",
    3: "Marzo",
    4: "Abril",
    5: "Mayo",
    6: "Junio",
    7: "Julio",
    8: "Agosto",
    9: "Septiembre",
    10: "Octubre",
    11: "Noviembre",
    12: "Diciembre"
}

bitacora_fechas["Mes"] = (
    bitacora_fechas["Mes_Numero"]
    .map(meses)
)


# =========================================================
# TOP 5 ASESORES
# =========================================================

# Seleccionar los 5 asesores con mayor número de registros
top_asesores = (
    bitacora.loc[
        bitacora["Asesor"] != "Sin asesor",
        "Asesor"
    ]
    .value_counts()
    .head(5)
    .index
)


# Filtrar únicamente esos asesores
sdc_asesor = bitacora[
    bitacora["Asesor"].isin(top_asesores)
].copy()


# Contar SDC por asesor
sdc_por_asesor = (
    sdc_asesor
    .groupby(
        ["Asesor", "SDC"]
    )
    .size()
    .reset_index(
        name="Clientes"
    )
)


# =========================================================
# VISITAS POR MES
# =========================================================

visitas_mes = (
    bitacora_fechas
    .groupby(
        ["Mes_Numero", "Mes"],
        as_index=False
    )
    .size()
)

visitas_mes.columns = [
    "Mes_Numero",
    "Mes",
    "Visitas"
]

# Ordenar cronológicamente
visitas_mes = visitas_mes.sort_values(
    "Mes_Numero"
)


# =========================================================
# DASHBOARD
# =========================================================

st.title("📊 Análisis Univariado GAC")

st.write(
    "Análisis de solicitudes de crédito de los principales asesores "
    "y comportamiento mensual de las visitas registradas."
)


# =========================================================
# CREAR COLUMNAS PARA MOSTRAR GRÁFICAS LADO A LADO
# =========================================================

col1, col2 = st.columns(2)


# =========================================================
# GRÁFICA 1
# SDC POR TOP 5 ASESORES
# =========================================================

with col1:

    st.subheader("💳 Solicitudes de crédito por asesor")

    fig_sdc = px.bar(
        sdc_por_asesor,
        x="Asesor",
        y="Clientes",
        color="SDC",
        barmode="group",
        text="Clientes",
        title="Top 5 asesores - Solicitudes de crédito"
    )

    fig_sdc.update_traces(
        textposition="outside"
    )

    fig_sdc.update_layout(
        title_x=0.5,
        xaxis_title="Asesor",
        yaxis_title="Número de clientes",
        legend_title="SDC",
        height=450
    )

    st.plotly_chart(
        fig_sdc,
        use_container_width=True
    )


# =========================================================
# GRÁFICA 2
# VISITAS POR MES
# =========================================================

with col2:

    st.subheader("📅 Visitas por mes")

    fig_mes = px.bar(
        visitas_mes,
        x="Mes",
        y="Visitas",
        text="Visitas",
        title="Visitas registradas por mes"
    )

    fig_mes.update_traces(
        textposition="outside"
    )

    fig_mes.update_layout(
        title_x=0.5,
        xaxis_title="Mes",
        yaxis_title="Número de visitas",
        showlegend=False,
        height=450
    )

    st.plotly_chart(
        fig_mes,
        use_container_width=True
    )