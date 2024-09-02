from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import streamlit as st
import re


def feature_engineering(df):    
    st.write(
        "En los datos de entrenamiento, se excluyó la categoría 'Miscellaneous', dado que estos son lo que queremos reclasificar. "
        "Se aplicó One-Hot Encoding a las columnas `device` y `city`, ya que son variables nominales sin un orden predefinido "
        "(si hubieran tenido ordinalidad, habría sido más apropiado usar Label Encoding). "
        "Para la columna `transaction_details_clean`, se extrajeron características textuales utilizando TF-IDF para capturar "
        "la relevancia de las palabras en el contexto de las descripciones de las transacciones."
        "De variables númericas voy a usar `withdrawal_amt`, dado que `deposit_amt` será nula en todos los casos y `balance_amt`"
        "tiene un alto porcentaje de negativos."
    )
    # Filtrar datos no Miscellaneous para entrenamiento
    train_data = df[df['category'] != 'Miscellaneous'].reset_index(drop=True)  
    st.write(
             """Al contar la `cantidad de transacciones de retiro por categoria` se observa que algunas categorías tienen un número 
             significativamente menor de registros en comparación con otras, lo que presenta un desafío debido al desbalance de 
             clases en los datos."""
             ,train_data['category'].value_counts().reset_index()
            )    
    st.write(
             """
                `Propuesta de Reagrupación de Categorías`:
                Para abordar el desbalance de clases y mejorar la clasificación de las transacciones, se propone agrupar las categorías originales en grupos más amplios.
                Estos `Group Categories` serán utilizados como nuevo objetivo para la predicción, permitiendo una mejor distribución de los registros y mejora la capacidad 
                del modelo para aprender de los datos y generalizar correctamente en nuevas predicciones.
                | **Group Category**                              | **Category**                  |
                |-------------------------------------------------|-------------------------------|
                | **Financial and Savings**                       | Transfer                      |
                |                                                 | Investment                    |
                |                                                 | Loan Payment                  |
                |                                                 | Insurance                     |
                | **Payments for Services and Basic Needs**       | Utility Bill                  |
                |                                                 | Subscriptions                 |
                |                                                 | Transportation                |
                | **Shopping and Consumption**                    | Shopping                      |
                |                                                 | Electronics & Gadgets         |
                |                                                 | Food & Dining                 |
                |                                                 | Pets & Pet Care               |
                | **Health and Wellness**                         | Health & Wellness             |
                |                                                 | Charity & Donations           |
                | **Personal Care, Entertainment and Education**  | Travel                        |
                |                                                 | Entertainment                 |
                |                                                 | Education                     |
                |                                                 | Childcare & Parenting         |

             """
            )
    # Definir el diccionario para agrupar categorías
    group_mapping = {
        'Financial and Savings': ['Transfer', 'Investment', 'Loan Payment', 'Insurance'],
        'Payments for Services and Basic Needs': ['Utility Bill', 'Subscriptions', 'Transportation'],
        'Shopping and Consumption': ['Shopping', 'Electronics & Gadgets', 'Food & Dining', 'Pets & Pet Care'],
        'Health and Wellness': ['Health & Wellness', 'Charity & Donations'],
        'Personal Care, Entertainment and Education': ['Travel', 'Entertainment', 'Education', 'Childcare & Parenting']
    }
    # Función para asignar grupo basado en la categoría
    def assign_group(category):
        for group, categories in group_mapping.items():
            if category in categories:
                return group
        return 'Other'  # Default group if not matched

    # Aplicar la función al DataFrame
    train_data['group_category'] = train_data['category'].apply(assign_group)

    st.write(
             """`cantidad de transacciones de retiro por group_category`"""
             ,train_data['group_category'].value_counts().reset_index()
            ) 

    # One-Hot Encoding para variables categóricas, excepto `category`
    train_data_encoded = pd.get_dummies(train_data, columns=['device', 'city'])

    # Extraer características de texto con TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
    X_text = vectorizer.fit_transform(train_data['transaction_details_clean'])

    # Unir las características relevantes (TF-IDF, montos y categóricas codificadas)
    X = pd.concat(
        [
            pd.DataFrame(X_text.toarray()),
            train_data_encoded[
                ['withdrawal_amt']
                + list(train_data_encoded.columns[train_data_encoded.columns.str.startswith(('device_', 'city_'))])
            ]
        ],
        axis=1
    )
    st.write("`Conjunto X de entrenamiento`",X)
    # Usar `category` directamente como la variable objetivo sin codificación
    y = train_data['group_category']

    st.write("`Conjunto y de entrenamiento`",y)
    return X, y


def feature_engineering_predict(df):    
    # Función de limpieza básica de texto
    def preprocess(texto):
        texto = texto.lower()  # Convertir a minúsculas
        texto = re.sub(r'\d+', '', texto)  # Eliminar números
        texto = re.sub(r"[^\w\s]", '', texto)  # Eliminar caracteres especiales
        return texto

    df['transaction_details_non_nan'] = df['transaction_details'].fillna(value="nan")
    # Aplicar la limpieza de texto
    df['transaction_details_clean'] = df['transaction_details_non_nan'].apply(preprocess)   

    # One-Hot Encoding para variables categóricas, excepto `category`
    train_data_encoded = pd.get_dummies(df, columns=['device', 'city'])

    # Extraer características de texto con TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
    X_text = vectorizer.fit_transform(df['transaction_details_clean'])

    # Unir las características relevantes (TF-IDF, montos y categóricas codificadas)
    X_predict = pd.concat(
        [
            pd.DataFrame(X_text.toarray()),
            train_data_encoded[
                ['withdrawal_amt']
                + list(train_data_encoded.columns[train_data_encoded.columns.str.startswith(('device_', 'city_'))])
            ]
        ],
        axis=1
    )
    st.write("`Conjunto X de predicción`", X_predict)

    return X_predict