import re
import streamlit as st


def transactions_details_cleaning(df):   
    # Crear la columna 'withdrawal_ind' que toma 1 si fue retiro, 0 indica deposito
    df['withdrawal_ind'] = df['withdrawal_amt'].notnull().astype(int)

    # Quitar None de transaction_details
    df_only_withdrawal = df[df['withdrawal_ind']==1]
        
    df_only_withdrawal['transaction_details_non_nan'] = df['transaction_details'].fillna(value="nan")

    st.write("Número de transaction_details únicas para retiros (sin preprocesar) :", df_only_withdrawal['transaction_details_non_nan'].nunique())
    
   
    # Función de limpieza básica de texto
    def preprocess(texto):
        texto = texto.lower()  # Convertir a minúsculas
        texto = re.sub(r'\d+', '', texto)  # Eliminar números
        texto = re.sub(r"[^\w\s]", '', texto)  # Eliminar caracteres especiales
        return texto

    # Aplicar la limpieza de texto
    df_only_withdrawal['transaction_details_clean'] = df_only_withdrawal['transaction_details_non_nan'].apply(preprocess)   
    
    st.write("Número de transaction_details únicas para retiros (eliminando caracteres especiales y números):", df_only_withdrawal['transaction_details_clean'].nunique())

    return df_only_withdrawal
    
