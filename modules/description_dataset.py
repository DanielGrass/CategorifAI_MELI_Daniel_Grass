import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from fitter import Fitter, get_common_distributions
import plotly.express as px
import matplotlib.pyplot as plt

def description_dataset():
    metadata={
                        "Variable": [
                            "account_id", "date", "transaction_details", "chq_no", "value_date",
                            "withdrawal_amt", "deposit_amt", "balance_amt", "category", "city", "device"
                        ],
                        "Descripción": [
                            "Número de cuenta involucrado en la transacción.",
                            "Fecha de la transacción.",
                            "Narración o descripción de la transacción en los estados de cuenta bancarios.",
                            "Número de cheque asociado con la transacción, si corresponde.",
                            "Fecha de finalización de la transacción.",
                            "Monto retirado en la transacción.",
                            "Monto depositado en la transacción.",
                            "Saldo actual de la cuenta después de la transacción.",
                            "Categoría asignada basada en los detalles de la transacción.",
                            "Ciudad donde se asume que ocurrió la transacción.",
                            "Tipo de dispositivo utilizado para la transacción (e.g., Móvil, Escritorio, Tablet)."
                        ]
                    }
    # Crear un DataFrame con la descripción de las columnas
    description_df = pd.DataFrame(metadata)
    st.write(description_df)

def null_analysis(df):
    # Crear un DataFrame con la información de calidad de datos para cada columna
    data_quality = {
        "Columna": [],
        "Cantidad Nulls": [],
        "Valores únicos": [],
        "Tipo de datos": [],
        "Valor mínimo": [],
        "Valor máximo": [],
    }

    # Iterar por cada columna del DataFrame original
    for col in df.columns:
        data_quality["Columna"].append(col)
        data_quality["Cantidad Nulls"].append(df[col].isnull().sum())
        data_quality["Valores únicos"].append(df[col].nunique())
        data_quality["Tipo de datos"].append(df[col].dtype)
        if df[col].dtype == 'float64':
            data_quality["Valor mínimo"].append(df[col].min())
            data_quality["Valor máximo"].append(df[col].max())
        else:
            data_quality["Valor mínimo"].append(0)
            data_quality["Valor máximo"].append(0)


    # Convertir el diccionario en un DataFrame
    quality_df = pd.DataFrame(data_quality)
    st.write(quality_df)

    summary = """
                **`account_id`**: La columna contiene 10 identificadores únicos en 116,201 registros, sin valores nulos. Se recomendaria convertirla a entero en caso de que el identificador sea puramente numérico y no tiene relevancia mantener un formato específico, como los ceros a la izquierda..

                **`date`**: Tiene el tipo de dato correcto, pero todas las horas están registradas como `00:00:00`, limitando el análisis temporal preciso. Considerar almacenar las fechas como Unix Timestamp o agregar la hora real para un análisis detallado.

                **`transaction_details`**: Con 2,499 nulos (2.15%) y alta cardinalidad con 44,806 valores únicos.

                **`chq_no`**: 99.21% de los registros son nulos, lo que sugiere baja relevancia en las transacciones con cheque. 

                **`value_date`**: Similar a `date`, tiene todas las horas en `00:00:00`. Se recomienda ajustar el formato o capturar la hora exacta para mejorar la precisión temporal.

                **`withdrawal_amt` y `deposit_amt`**: Ambas son excluyentes con nulos que coinciden con el total de registros, sin transacciones negativas.

                **`balance_amt`**: La columna presenta balances negativos, lo que podría ser permitido o un error. Se sugiere validar si estos saldos son aceptables y analizar cuántos usuarios los presentan.

                **`category`**: Sin valores nulos y con 18 categorías diferentes. 

                **`city`**: Contiene 10 ciudades y no presenta nulos, lo que indica buena cobertura geográfica. 

                **`device`**: Con 3 tipos de dispositivos y sin valores nulos.
                """
    st.write(summary)

# Función para graficar barras de una columna categórica con línea de Pareto usando Plotly
def plot_categorical_distribution_with_pareto(column_name, data, color):
            # Contar las ocurrencias de cada categoría y calcular el porcentaje acumulado
            category_counts = data[column_name].value_counts()
            category_percent = 100 * category_counts.cumsum() / category_counts.sum()
            
            # Crear el gráfico de barras
            fig = go.Figure()

            # Agregar barras
            fig.add_trace(go.Bar(
                x=category_counts.index,
                y=category_counts.values,
                marker_color=color,
                name='Frecuencia',
                yaxis='y1'
            ))

            # Agregar la línea de Pareto en un eje secundario
            fig.add_trace(go.Scatter(
                x=category_counts.index,
                y=category_percent,
                mode='lines+markers',
                name='% Acumulado',
                line=dict(color='red', dash='dash'),
                marker=dict(color='red'),
                yaxis='y2'
            ))

            # Configuración de los ejes y título
            fig.update_layout(
                title=f"Distribución de {column_name.capitalize()} y Línea de Pareto",
                xaxis_title=column_name.capitalize(),
                yaxis=dict(title='Frecuencia', side='left', showgrid=False),
                yaxis2=dict(title='% Acumulado', side='right', overlaying='y', range=[0, 100], showgrid=False),
                template="plotly_white"
            )
            
            # Mostrar el gráfico en Streamlit
            st.plotly_chart(fig)


# Función para visualizar las distribuciones antes y después de eliminar outliers
def plot_distribution_with_outlier_removal(column, df):
    st.subheader(f"Análisis de `{column}`")

    # Visualización inicial de la distribución
    st.write("### Distribución Original")
    fig_hist = px.histogram(df, x=column, nbins=50, title=f"Histograma de {column} (Original)", marginal="box")
    st.plotly_chart(fig_hist)


    # Cálculo del IQR para detectar y remover outliers
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_no_outliers = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    # Visualización de la distribución sin outliers
    st.write("### Distribución sin Outliers")
    fig_hist_no_outliers = px.histogram(df_no_outliers, x=column, nbins=50, title=f"Histograma de {column} (Sin Outliers)", marginal="box")
    st.plotly_chart(fig_hist_no_outliers)

    # Análisis de ajuste de distribuciones con Fitter en datos sin outliers
    st.write("### Ajuste de Distribuciones con Fitter")
   
    # Crear el objeto Fitter con las distribuciones seleccionadas
    f = Fitter(df_no_outliers[column].dropna(), distributions=[
        'expon', 'norm', 'johnsonsb', 'lognorm', 'uniform'
    ])
    
    # Ajustar las distribuciones
    f.fit()
    best_fit = f.get_best(method='sumsquare_error')

    # Mostrar el mejor ajuste encontrado
    st.write(f"Mejor ajuste para `{column}`: {list(best_fit.keys())[0]}")

    # Mostrar el resumen de los ajustes sin argumento 'ax'
    fig = plt.figure(figsize=(10, 5))
    f.summary()  # Generar y mostrar el gráfico del ajuste
    st.pyplot(fig)  # Mostrar el gráfico en Streamlit

    
