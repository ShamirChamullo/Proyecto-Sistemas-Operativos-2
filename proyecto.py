import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import io
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Configurar estilo de los gráficos
sns.set(style="whitegrid")

# Función para generar y guardar gráficos
def generate_and_save_plot(plot_func, file_name):
    plt.figure(figsize=(12, 8))
    plot_func()
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    st.image(buf, caption=file_name)
    st.download_button(label=f"Descargar {file_name}", data=buf, file_name=file_name, mime='image/png')
    plt.clf()

# Función para generar y guardar el archivo Excel
def save_to_excel(df, file_name):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    buf.seek(0)
    st.download_button(label=f"Descargar {file_name}", data=buf, file_name=file_name, mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

# Función para calcular y mostrar regresión lineal
def plot_regression_with_r2(x, y, df, x_label, y_label, title):
    X = df[[x]].values.reshape(-1, 1)
    Y = df[y].values.reshape(-1, 1)
    reg = LinearRegression().fit(X, Y)
    Y_pred = reg.predict(X)
    r2 = r2_score(Y, Y_pred)
    sns.lmplot(x=x, y=y, data=df, aspect=1.5, scatter_kws={'s':100}, line_kws={'color':'red'})
    plt.title(f'{title}\nR² = {r2:.2f}')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    return r2

# Función para generar gráficos
def generate_plots(df):
    columns = df.columns.tolist()
    
    if 'age' in columns:
        # Histograma de la edad con línea de ajuste y probabilidad
        st.write("### Distribución de Edad con Línea de Ajuste y Probabilidad")
        def plot_age_histogram():
            sns.histplot(df['age'], bins=20, kde=True, color='skyblue', stat='density', linewidth=0)
            plt.title('Distribución de Edad con Línea de Ajuste')
            plt.xlabel('Edad')
            plt.ylabel('Densidad')
            # Añadir texto con probabilidad (ejemplo con valor fijo)
            plt.text(x=df['age'].mean(), y=0.1, s='Probabilidad: 0.51', horizontalalignment='center', fontsize=12, color='red')
        generate_and_save_plot(plot_age_histogram, 'histograma_edad.png')
    
    if 'target' in columns:
        # Gráfico de torta para la variable 'target'
        st.write("### Distribución de Objetivo (Target)")
        target_counts = df['target'].value_counts()
        def plot_target_pie():
            plt.pie(target_counts, labels=target_counts.index, autopct='%1.1f%%', colors=['lightcoral', 'lightskyblue'], startangle=140)
            plt.title('Distribución de Objetivo')
        generate_and_save_plot(plot_target_pie, 'grafico_torta_objetivo.png')

        # Gráfico de barras para el recuento de la variable 'target'
        st.write("### Conteo de Objetivo")
        def plot_target_count():
            sns.countplot(x='target', data=df, palette='viridis')
            plt.title('Conteo de Objetivo')
            plt.xlabel('Objetivo (0 = No, 1 = Sí)')
            plt.ylabel('Número de Pacientes')
        generate_and_save_plot(plot_target_count, 'conteo_objetivo.png')

    if 'restingBP' in columns and 'serumcholestrol' in columns:
        # Gráfico de dispersión con regresión lineal simple
        st.write("### Regresión Lineal: Presión Arterial vs Colesterol")
        def plot_regression():
            r2 = plot_regression_with_r2('restingBP', 'serumcholestrol', df, 'Presión Arterial en Reposo', 'Colesterol en Suero', 'Regresión Lineal de Presión Arterial vs Colesterol')
            st.write(f"R²: {r2:.2f}")
        generate_and_save_plot(plot_regression, 'regresion_presion_colesterol.png')

    if 'oldpeak' in columns and 'target' in columns:
        # Boxplot de 'oldpeak' por 'target'
        st.write("### Oldpeak por Objetivo")
        def plot_oldpeak_boxplot():
            sns.boxplot(x='target', y='oldpeak', data=df, palette='Set2')
            plt.title('Oldpeak por Objetivo')
            plt.xlabel('Objetivo (0 = No, 1 = Sí)')
            plt.ylabel('Oldpeak')
        generate_and_save_plot(plot_oldpeak_boxplot, 'boxplot_oldpeak.png')

    if 'age' in columns and 'restingBP' in columns:
        # Regresión lineal de edad vs presión arterial en reposo
        st.write("### Regresión Lineal: Edad vs Presión Arterial en Reposo")
        def plot_regression_age_bp():
            r2 = plot_regression_with_r2('age', 'restingBP', df, 'Edad', 'Presión Arterial en Reposo', 'Regresión Lineal de Edad vs Presión Arterial en Reposo')
            st.write(f"R²: {r2:.2f}")
        generate_and_save_plot(plot_regression_age_bp, 'regresion_edad_presion.png')

    if 'age' in columns and 'serumcholestrol' in columns:
        # Regresión lineal de edad vs colesterol en suero
        st.write("### Regresión Lineal: Edad vs Colesterol en Suero")
        def plot_regression_age_chol():
            r2 = plot_regression_with_r2('age', 'serumcholestrol', df, 'Edad', 'Colesterol en Suero', 'Regresión Lineal de Edad vs Colesterol en Suero')
            st.write(f"R²: {r2:.2f}")
        generate_and_save_plot(plot_regression_age_chol, 'regresion_edad_colesterol.png')

    if 'restingBP' in columns and 'oldpeak' in columns:
        # Regresión lineal de presión arterial en reposo vs oldpeak
        st.write("### Regresión Lineal: Presión Arterial en Reposo vs Oldpeak")
        def plot_regression_bp_oldpeak():
            r2 = plot_regression_with_r2('restingBP', 'oldpeak', df, 'Presión Arterial en Reposo', 'Oldpeak', 'Regresión Lineal de Presión Arterial en Reposo vs Oldpeak')
            st.write(f"R²: {r2:.2f}")
        generate_and_save_plot(plot_regression_bp_oldpeak, 'regresion_presion_oldpeak.png')

# Interfaz de usuario de Streamlit
st.title('Análisis de Datos de Pacientes')

uploaded_file = st.file_uploader("Selecciona un archivo CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(f"Datos cargados: {uploaded_file.name}")
    
    if not df.empty:
        st.write("### Vista previa de los datos")
        st.write(df.head())

        generate_plots(df)

        # Guardar y permitir la descarga del archivo Excel
        save_to_excel(df, 'datos_pacientes.xlsx')
    else:
        st.error("El archivo CSV está vacío.")
else:
    st.info("Por favor, sube un archivo CSV para analizar.")
