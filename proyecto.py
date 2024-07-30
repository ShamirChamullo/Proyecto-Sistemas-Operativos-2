import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.ticker as mticker
from io import BytesIO
import requests

# Título
st.title('Análisis de Youtubers')

# Descargar el archivo CSV desde Google Drive (ajustar URL si es necesario)
url = 'https://drive.google.com/uc?id=1vyt2Uh_4u5riWmBmtlHYYx7v8anKwIDB'
try:
    response = requests.get(url)
    data = pd.read_csv(BytesIO(response.content))
except Exception as e:
    st.error(f"Error al cargar los datos: {e}")
    st.stop()

# Mostrar las primeras filas del DataFrame
st.write("Primeras filas del archivo:")
st.write(data.head())

# Limpiar datos (remover filas con valores nulos en Suscribers, Visits, Likes o Comments)
data = data.dropna(subset=['Suscribers', 'Visits', 'Likes', 'Comments'])

# Control deslizante para limitar el número de filas a mostrar
st.write("Selecciona el número de filas a mostrar:")
num_rows = st.slider('Número de filas', min_value=1, max_value=len(data), value=len(data))

# Aplicar el limitador de filas a los datos
filtered_data = data.head(num_rows)

# Mostrar los datos filtrados
st.write("Datos filtrados:")
st.write(filtered_data)

# Función para guardar gráficos como archivos en un buffer
def save_plot_to_buffer(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return buf

# Gráfico de torta de la distribución de YouTubers por país
if 'Country' in filtered_data.columns:
    st.write("Distribución de YouTubers por país")
    country_counts = filtered_data['Country'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.pie(country_counts, labels=country_counts.index, autopct='%1.1f%%', startangle=140)
    ax.set_title('Distribución de YouTubers por País')
    ax.axis('equal')
    buf = save_plot_to_buffer(fig)
    st.pyplot(fig)
    st.download_button(label="Descargar gráfico de distribución por país", data=buf, file_name="distribucion_pais.png", mime="image/png")

# Gráfico de torta de la distribución de categorías
if 'Categories' in filtered_data.columns:
    st.write("Distribución de YouTubers por categoría")
    category_counts = filtered_data['Categories'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140)
    ax.set_title('Distribución de YouTubers por Categoría')
    ax.axis('equal')
    buf = save_plot_to_buffer(fig)
    st.pyplot(fig)
    st.download_button(label="Descargar gráfico de distribución por categoría", data=buf, file_name="distribucion_categoria.png", mime="image/png")

# Selección de variables para regresión lineal
st.write("Seleccione las variables para la regresión lineal simple:")
x_var = st.selectbox('Variable Independiente (X)', ['Suscribers', 'Likes', 'Comments'])
y_var = st.selectbox('Variable Dependiente (Y)', ['Visits'])

# Función para crear gráficos de regresión
def plot_regression(x, y, x_label, y_label, title):
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)
    r2 = r2_score(y, y_pred)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=x.flatten(), y=y, data=filtered_data, label='Datos reales', ax=ax)
    ax.plot(x, y_pred, color='red', label='Regresión lineal')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f'{title} (R² = {r2:.2f})')

    # Ajustar formato de los ejes
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:,.0f}'))
    ax.legend()
    buf = save_plot_to_buffer(fig)
    st.pyplot(fig)
    st.download_button(label="Descargar gráfico de regresión", data=buf, file_name="regresion_lineal.png", mime="image/png")

    # Mostrar valor de R²
    st.write(f'**Valor de R²**: {r2:.2f}')

# Ejecutar regresión lineal basada en selección del usuario
X = filtered_data[x_var].values.reshape(-1, 1)
y = filtered_data[y_var].values
plot_regression(X, y, x_var, y_var, f'Regresión Lineal entre {x_var} y {y_var}')

# Distribución de Datos
st.write("Distribución de Datos")
for column in ['Suscribers', 'Visits', 'Likes', 'Comments']:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(filtered_data[column], kde=True, bins=30, ax=ax)
    ax.set_xlabel(column)
    ax.set_ylabel('Frecuencia')
    ax.set_title(f'Distribución de {column}')

    # Ajustar formato de los ejes
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:,.0f}'))
    buf = save_plot_to_buffer(fig)
    st.pyplot(fig)
    st.download_button(label=f"Descargar gráfico de distribución de {column}", data=buf, file_name=f"distribucion_{column}.png", mime="image/png")

# Tendencias Temporales (si hay una columna de fecha)
if 'Date' in filtered_data.columns:
    filtered_data['Date'] = pd.to_datetime(filtered_data['Date'])
    st.write("Tendencias Temporales")
    fig, ax = plt.subplots(figsize=(14, 7))
    for column in ['Suscribers', 'Visits', 'Likes', 'Comments']:
        ax.plot(filtered_data['Date'], filtered_data[column], label=column)
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Valor')
    ax.set_title('Tendencias Temporales')
    ax.legend()

    # Ajustar formato de los ejes
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:,.0f}'))
    buf = save_plot_to_buffer(fig)
    st.pyplot(fig)
    st.download_button(label="Descargar gráfico de tendencias temporales", data=buf, file_name="tendencias_temporales.png", mime="image/png")

# Gráfico de Pareto de Categorías
if 'Categories' in filtered_data.columns:
    st.write("Gráfico de Pareto de Categorías")
    category_counts = filtered_data['Categories'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=category_counts.index, y=category_counts.values, ax=ax)
    ax.set_xlabel('Categorías')
    ax.set_ylabel('Número de YouTubers')
    ax.set_title('Distribución de YouTubers por Categoría')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    # Ajustar formato de los ejes
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:,.0f}'))
    buf = save_plot_to_buffer(fig)
    st.pyplot(fig)
    st.download_button(label="Descargar gráfico de Pareto de Categorías", data=buf, file_name="pareto_categorias.png", mime="image/png")

# Descargar archivo Excel con los datos
st.write("Descargar datos como archivo Excel")
excel_buffer = BytesIO()
with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
    filtered_data.to_excel(writer, sheet_name='Datos', index=False)
excel_buffer.seek(0)
st.download_button(label="Descargar datos como Excel", data=excel_buffer, file_name="datos_youtubers.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
