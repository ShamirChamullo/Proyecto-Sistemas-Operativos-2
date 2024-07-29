import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import io

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
    df.to_excel(buf, index=False)
    buf.seek(0)
    st.download_button(label=f"Descargar {file_name}", data=buf, file_name=file_name, mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

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
            sns.lmplot(x='restingBP', y='serumcholestrol', data=df, aspect=1.5, scatter_kws={'s':100}, line_kws={'color':'red'})
            plt.title('Regresión Lineal de Presión Arterial vs Colesterol')
            plt.xlabel('Presión Arterial en Reposo')
            plt.ylabel('Colesterol en Suero')
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

    if 'gender' in columns and 'age' in columns and 'serumcholestrol' in columns:
        # Regresión Lineal Simple por Género
        genders = df['gender'].unique()
        for gender in genders:
            st.write(f"### Regresión Lineal por Género: {gender}")
            gender_df = df[df['gender'] == gender]
            if not gender_df.empty:
                def plot_gender_regression():
                    sns.lmplot(x='age', y='serumcholestrol', data=gender_df, aspect=1.5, scatter_kws={'s':100}, line_kws={'color':'red'})
                    plt.title(f'Regresión Lineal de Edad vs Colesterol para {gender}')
                    plt.xlabel('Edad')
                    plt.ylabel('Colesterol en Suero')
                file_name = f'regresion_edad_colesterol_{gender}.png'
                generate_and_save_plot(plot_gender_regression, file_name)
            else:
                st.write(f"No hay datos disponibles para el género {gender}.")

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
