import pandas as pd
import streamlit as st

@st.cache_data
def load_local_parquet(file_name='bank_transactions.parquet'):
    """
    Carga un archivo .parquet desde la carpeta data.

    Parameters:
    - file_name: Nombre del archivo .parquet a cargar.

    Returns:
    - df: DataFrame con los datos cargados.
    """
    file_path = f'data/{file_name}'  # Ruta al archivo dentro de la carpeta data
    try:
        # Cargar el archivo .parquet en un DataFrame
        df = pd.read_parquet(file_path)
        return df
    except Exception as e:
        st.error(f"Error al cargar el archivo {file_name}: {e}")
        return None
