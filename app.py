import streamlit as st
from modules.theme_toggle import initialize_theme, toggle_theme, apply_styles
from modules.description_dataset import description_dataset, null_analysis, plot_categorical_distribution_with_pareto
from data.data_loader import load_local_parquet
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from fitter import Fitter, get_common_distributions

# Inicializar el estado del tema
initialize_theme()

# Configuración de la página
st.set_page_config(page_title="CategorifAI", page_icon=":bar_chart:", layout="wide")

# Aplicar los estilos según el modo seleccionado
apply_styles()

# URL del logo
logo_url = "https://http2.mlstatic.com/frontend-assets/ml-web-navigation/ui-navigation/6.6.73/mercadolibre/logo_large_25years@2x.png?width=300"

# Variables para la selección de secciones
if 'selected_main' not in st.session_state:
    st.session_state.selected_main = None
    
# Barra lateral con el logo y menú
with st.sidebar:
    st.image(logo_url, use_column_width=True)  # Muestra el logo desde la URL
    st.title("Menú Principal")
      
    # Botón para cambiar entre modo claro y oscuro con íconos de sol y luna
    theme_button = "🌙" if not st.session_state.dark_mode else "☀️"
    st.button(theme_button, on_click=toggle_theme)

    # Botones principales para secciones
    if st.button("Requerido"):
        st.session_state.selected_main = "Requerido"

    if st.button("Deseable"):
        st.session_state.selected_main = "Deseable"

    if st.button("Bonus"):
        st.session_state.selected_main = "Bonus"

# Menú horizontal a la derecha basado en la selección
st.title("Data Science Technical Challenge - CategorifAI")
st.subheader("Presentado por: Daniel Grass")

# Cargael archivo .parquet usando la función cacheada
df = load_local_parquet()


if st.session_state.selected_main:
    # Mostrar un menú horizontal según la selección del botón principal
    if st.session_state.selected_main == "Requerido":
        st.header("Tareas requeridas")
        st.write("**`Data QA`**: Se debe chequear la calidad del dataset para hacer una evaluación de qué tan apropiados son los datos para tareas de Data Science. Proponga un conjunto de correcciones en los datos de ser necesario.")
        st.write("**`Reporting`**: Documente los resultados e insights obtenidos durante la exploración y describa conclusiones desde una perspectiva de negocio, soportado por gráficos / tablas / métricas.")
        st.write("**`Machine Learning`**: Describa las posibles tareas de Machine Learning que podrían realizarse desde el dataset dado, que podrían ser valiosas en el dominio dado (sólo explicar, **no entrenar un modelo**)")
        
       

        # Mostrar la Descripción de las Columnas del Dataset
        st.header("Descripción de las Columnas del Dataset")
        description_dataset()

        st.header("1. Imprimir los datos originales del archivo Parquet proporcionado.")
        if df is not None:
            st.write(df)  # Muestra el DataFrame en Streamlit

        
        # Análisis de tipos de datos y valores nulos
        st.header("2. Análisis de tipos de datos y valores nulos")        
        null_analysis(df)

        # Análisis Categórico:
        st.header("3. Análisis Categórico:")
        
        # Análisis de la columna `category`
        st.subheader("Distribución de `category`")
        plot_categorical_distribution_with_pareto('category', df, color='skyblue')

        # Análisis de la columna `city`
        st.subheader("Distribución de `city`")
        plot_categorical_distribution_with_pareto('city', df, color='orange')

        # Análisis de la columna `device`
        st.subheader("Distribución de `device`")
        plot_categorical_distribution_with_pareto('device', df, color='purple')

        # Análisis de la columna `transaction_details`
        st.subheader("Distribución de `account_id`")
        plot_categorical_distribution_with_pareto('account_id', df, color='pink')
       
        st.subheader("Conclusiones del Análisis Categórico:")
        conclusiones_categorico = """       

        1. **Distribución de `Category`**:
        - El análisis de la variable `Category` revela que una gran mayoría de las transacciones (96.53%) se concentran en solo cuatro categorías: **Miscellaneous**, **Transfer**, **Investment**, y **Subscriptions**. Esto sugiere que la actividad financiera de los usuarios está altamente centralizada en estas áreas, lo que podría indicar patrones específicos de uso o falta de diversificación en las clasificaciones de las transacciones. Es crucial explorar estas categorías más a fondo para identificar si esta centralización responde a necesidades específicas de los usuarios o si podría beneficiarse de una reclasificación más detallada.

        2. **Distribución de `City` y `Device`**:
        - Las variables `City` y `Device` presentan una distribución más uniforme, donde cada categoría muestra una participación similar. Esto indica que no hay una fuerte dependencia de la transacción en función de la ubicación geográfica o el tipo de dispositivo utilizado. Esta distribución homogénea sugiere que las transacciones se distribuyen equitativamente entre las ciudades y los dispositivos, reflejando un comportamiento de usuario relativamente constante sin sesgos significativos hacia un tipo de dispositivo o una ciudad específica. Este patrón uniforme podría implicar que no hay una necesidad urgente de segmentar estrategias por ubicación o tipo de dispositivo en este momento.
        
        3. **Distribución de `account_id`**:
        -  Un pequeño grupo de cuentas domina la mayoría de las transacciones. Las primeras dos cuentas concentran una gran proporción de la actividad total (67.65%), sugiriendo usuarios muy activos. Esto ofrece una oportunidad para analizar el comportamiento de los usuarios, identificando patrones que podrían guiar estrategias de segmentación y retención como modelos RFM.
        
        4. **Distribución de `transaction_details`**:
        -  La variable transaction_details presenta una alta cardinalidad, lo que significa que contiene un gran número de valores únicos que dificulta su análisis directo y la identificación de patrones claros. se propone realizar una reclasificación de transaction_details agrupando los registros por palabras clave o frases similares. Esto se puede lograr mediante técnicas de procesamiento de lenguaje natural (NLP), como la identificación de palabras clave, la eliminación de stopwords, y el uso de algoritmos de similaridad textual como fuzzy matching o TF-IDF. Estas técnicas permitirán agrupar transacciones en categorías más manejables y coherentes, como "pagos", "transferencias", "compras", entre otras, facilitando su análisis.
        """

        # Mostrar el texto en Streamlit
        st.markdown(conclusiones_categorico)



        # Asegúrate de que las columnas son datetime (aunque ya lo son, es una verificación adicional)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['value_date'] = pd.to_datetime(df['value_date'], errors='coerce')

        # Calcular la diferencia en segundos entre 'value_date' y 'date'
        df['time_difference_seconds'] = (df['value_date'] - df['date']).dt.total_seconds()

        # Mostrar el resultado
        st.write("Diferencia en segundos entre 'value_date' y 'date':")
        st.write(df[['date', 'value_date', 'time_difference_seconds']])
                     
       

    elif st.session_state.selected_main == "Deseable":
        menu_options = st.radio(
            "Opciones de Deseable",
            options=["Versionado de Código", "Feature Engineering", "Modelo Predictivo", "Mostrar Skills en Python", "Casos de Uso", "Métricas"],
            horizontal=True
        )

        # Mostrar el contenido correspondiente
        if menu_options == "Versionado de Código":
            st.header("Versionado de Código")
            st.write("Versionado de código con Git (incluso puede publicarse en tu cuenta personal de GitHub).")

        elif menu_options == "Feature Engineering":
            st.header("Feature Engineering")
            st.write("Indicar y calcular posibles candidatos de features que podrían utilizarse tanto columnas originales y transformaciones.")

        elif menu_options == "Modelo Predictivo":
            st.header("Modelo Predictivo")
            st.write("Realice un modelo predictivo.")

        elif menu_options == "Mostrar Skills en Python":
            st.header("Mostrar Skills en Python")
            st.write("Teniendo buenas prácticas en la estructura del código y la documentación.")

        elif menu_options == "Casos de Uso":
            st.header("Casos de Uso")
            st.write("Describir posibles casos de uso a tratar con este dataset que podrían agregar valor al negocio dado, indicando métodos, técnicas, y algoritmos por cada uno de ellos, así como justificando las decisiones tomadas.")

        elif menu_options == "Métricas":
            st.header("Métricas")
            st.write("Definir y calcular las métricas que considere más relevantes para la problemática propuesta.")

    elif st.session_state.selected_main == "Bonus":
        menu_options = st.radio(
            "Opciones de Bonus",
            options=["Manejo de Environment de Desarrollo", "Identificar Nuevos Atributos"],
            horizontal=True
        )

        # Mostrar el contenido correspondiente
        if menu_options == "Manejo de Environment de Desarrollo":
            st.header("Manejo de Environment de Desarrollo")
            st.write("Manejo de environment de desarrollo mediante alguna tecnología (e.g. Docker, virtualenv, conda).")

        elif menu_options == "Identificar Nuevos Atributos":
            st.header("Identificar Nuevos Atributos")
            st.write("Identificar nuevos atributos / tablas que podrían ser relevantes o necesarias para un mejor análisis.")