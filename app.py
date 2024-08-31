import streamlit as st
from modules.theme_toggle import initialize_theme, toggle_theme, apply_styles
from modules.description_dataset import description_dataset, null_analysis, plot_categorical_distribution_with_pareto, plot_distribution_with_outlier_removal, analysis_datetime_variables, analysis_withdrawal_deposit
from data.data_loader import load_local_parquet
import pandas as pd
import numpy as np
import plotly.express as px

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
        
        ##################################################################################
        ###############1. Imprimir los datos originales del archivo Parquet proporcionado.
        ##################################################################################
        st.header("1. Imprimir los datos originales del archivo Parquet proporcionado.")
        if df is not None:
            st.write(df)  # Muestra el DataFrame en Streamlit

        ##################################################################################
        ###############2. Análisis de tipos de datos y valores nulos
        ##################################################################################
        st.header("2. Análisis de tipos de datos y valores nulos")        
        null_analysis(df)

        ##################################################################################
        ###############3. Análisis Categórico:
        ##################################################################################
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

        ############################################################################################
        ###############4. Análisis variables continuas (Distribuciones y tratamiento de anomalos):
        ############################################################################################
        st.header("4. Análisis variables continuas (Distribuciones y tratamiento de anomalos:")

        numerical_columns = ['withdrawal_amt', 'deposit_amt', 'balance_amt']
        # Visualización de cada variable continua antes y después de remover outliers
        for col in numerical_columns:
            plot_distribution_with_outlier_removal(col, df)
        
    

        conclusiones_variables_continuas = """
                        ### Conclusiones de Análisis variables continuas (Distribuciones y tratamiento de anomalos

                        **Análisis de `withdrawal_amt`**:

                        - El histograma de withdrawal_amt muestra que aproximadamente el 80% de los registros se encuentran en el rango de [0 - 10M]. Esto sugiere una alta concentración de transacciones de retiro en valores bajos a moderados, lo cual es típico en escenarios donde la mayoría de los usuarios realizan transacciones pequeñas o medianas.
                        - Se detecta un valor máximo significativo de 459M, indicando la presencia de outliers que pueden distorsionar la interpretación de los datos y afectar los análisis estadísticos.
                        - Después de eliminar los outliers utilizando el rango intercuartílico (IQR), los datos se ajustan al rango [0 - 12M], con una concentración predominante en el intervalo [0 - 250k]. En este rango se encuentran 30,958 de los 53,549 registros totales, lo que refuerza que la mayoría de las transacciones están en valores relativamente bajos.
                        - Se utilizó Fitter para identificar la mejor distribución teórica que se ajusta a los datos sin outliers, resultando en la distribución Johnson SB como la más adecuada. Esta distribución es conocida por su flexibilidad en modelar datos asimétricos y con colas largas.
                        - Es crucial realizar pruebas de hipótesis, como la de Kolmogorov-Smirnov, para validar si la distribución ajustada se adapta adecuadamente a los datos. Este paso permitirá confirmar si el modelo teórico es estadísticamente significativo o si se requiere un ajuste adicional.
                        
                        **Análisis de `deposit_amt`**:

                        - El histograma de deposit_amt muestra que aproximadamente el 81% de los registros se concentran en el rango [0 - 10M]. Al igual que en withdrawal_amt, esto sugiere que la mayoría de las transacciones de depósito son pequeñas, con una alta frecuencia en los valores bajos.
                        - Al eliminar los outliers mediante el IQR, los datos se ajustan al rango [0 - 11.7M], con una concentración notable en el rango [0 - 250k]. Este grupo comprende 26,146 de los 62,652 registros, evidenciando que las transacciones más frecuentes son de montos menores.
                        - El análisis con Fitter identificó que la Lognormal es la distribución que mejor se ajusta a los datos sin outliers. Esta distribución es común en datos financieros donde los valores son positivos y presentan una asimetría hacia la derecha, como es el caso de las transacciones monetarias.
                        - Al igual que con withdrawal_amt, es fundamental validar el ajuste con pruebas de hipótesis como Kolmogorov-Smirnov para asegurar que la Lognormal representa adecuadamente los datos y no se deben considerar otras distribuciones.

                       **Análisis de `balance_amt`**:

                        - El análisis del histograma de balance_amt revela que aproximadamente el 97.5% de los registros tienen un balance negativo, mientras que solo el 2.5% de los registros muestran un balance positivo. Esta distribución indica que la mayoría de las cuentas están en déficit, lo cual podría reflejar un comportamiento común de los usuarios o posibles inconsistencias en los datos de las transacciones.
                        - Adicionalmente, se identificó que únicamente 2 usuarios presentan balances positivos en sus cuentas, lo que sugiere que la situación financiera general de los usuarios es predominantemente negativa.
                        - Es recomendable ajustar el formato de las fechas en value_date y date para incluir horas y minutos. Esto permitirá un análisis más detallado de la cronología de las transacciones y una posible reconstrucción precisa de los balances.

                        **Segregación por Segmentos**:

                        - Se recomienda evaluar la posibilidad de segmentar los datos por categorías adicionales, como tipo de cliente, periodo o monto. Esto permitiría identificar patrones más específicos que podrían mejorar la toma de decisiones basada en los datos.
                        """
        ## Validar comportamiento balance_amt
        # Mensaje para la cantidad de registros con balance menor a 0
        total_negative_balances = df["balance_amt"][df["balance_amt"] < 0].count()
        percentage_negative_balances = np.round((total_negative_balances / df["balance_amt"].count()) * 100, 2)
        text_warning_balance_amt = (
            f"⚠️ **Cantidad de registros con balance negativo:** {total_negative_balances} registros, "
            f"representando el {percentage_negative_balances}% del total."
        )
        st.error(text_warning_balance_amt)

        # Mensaje para la cantidad de usuarios con balance menor a 0
        total_negative_accounts = df["account_id"][df["balance_amt"] < 0].nunique()
        percentage_negative_accounts = np.round((total_negative_accounts / df["account_id"].nunique()) * 100, 2)
        text_warning_account_id = (
            f"⚠️ **Cantidad de usuarios con balance negativo:** {total_negative_accounts} usuarios, "
            f"representando el {percentage_negative_accounts}% de todos los usuarios."
        )
        st.error(text_warning_account_id)

        # Mostrar las conclusiones en Streamlit
        st.markdown(conclusiones_variables_continuas)

                     
        ############################################################################################
        ###############5. Análisis de Fechas de Transacción y Finalización:
        ############################################################################################
        st.header("5. Análisis de Fechas de Transacción y Finalización")
    
        analysis_datetime_variables(df)
        analysis_withdrawal_deposit(df)
        conclusiones_fechas_transaccion = """
                        ### Conclusiones de Análisis de Fechas de Transacción y Finalización

                        **Análisis de Conteo Total de Transacciones Diarias**:
                        - La gráfica muestra una tendencia general de aumento en la cantidad de transacciones diarias a lo largo del tiempo, especialmente a partir de mediados de 2016. Este crecimiento podría indicar un incremento en la actividad de los usuarios o en la adopción de los servicios a lo largo del periodo analizado.                        
                        - Se observan varios picos pronunciados en la cantidad de transacciones, especialmente entre 2017 y 2018. Estos picos podrían estar asociados a eventos específicos, promociones, campañas comerciales, o comportamientos anómalos que podrían requerir un análisis adicional para identificar sus causas.                        
                        - Desde 2015 hasta mediados de 2016, la actividad de transacciones parece más estable y con un volumen relativamente bajo, lo que podría reflejar una etapa temprana o más controlada del sistema.
                        - Se recomienda realizar un análisis de causalidad para identificar si ciertos eventos, políticas o promociones están directamente relacionados con los aumentos o disminuciones en la actividad de las transacciones.    
                        - Los account_ids '409000438620' y '1196428' son los que parecen tener más picos anomalos.           

                        **Análisis de Tendencia Diaria de Depósitos y Retiros con Indicadores de Promedio**:
                        - La gráfica muestra que el promedio diario de retiros (withdrawal_amt) es de 185.7B, mientras que el promedio diario de depósitos (deposit_amt) es de 184.3B. Esta diferencia indica que, en general, los retiros superan a los depósitos, lo cual es un factor clave que podría estar contribuyendo a los balances negativos observados en las cuentas. Este comportamiento sugiere que la salida de fondos es más alta que la entrada, lo cual puede tener implicaciones importantes sobre la salud financiera general de los usuarios o la operación del sistema.
         
                        """
        # Mostrar las conclusiones en Streamlit
        st.markdown(conclusiones_fechas_transaccion)

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