from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import streamlit as st

def feature_engineering(df):    
    st.write(
        "En los datos de entrenamiento, se excluyó la categoría 'Miscellaneous', dado que estos son lo que queremos reclasificar. "
        "Se aplicó One-Hot Encoding a las columnas `device` y `city`, ya que son variables nominales sin un orden predefinido "
        "(si hubieran tenido un orden lógico, habría sido más apropiado usar Label Encoding). "
        "Para la columna `transaction_details_clean`, se extrajeron características textuales utilizando TF-IDF para capturar "
        "la relevancia de las palabras en el contexto de las descripciones de las transacciones."
        "De variables númericas voy a usar `withdrawal_amt`, dado que `deposit_amt` será nula en todos los casos y `balance_amt`"
        "tiene un alto porcentaje de negativos."
    )
    # Filtrar datos no Miscellaneous para entrenamiento
    train_data = df[df['category'] != 'Miscellaneous'].reset_index(drop=True)    
    
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
    st.write("Conjunto X de entrenamiento ",X)
    # Usar `category` directamente como la variable objetivo sin codificación
    y = train_data['category']
    st.write("Conjunto y de entrenamiento ",y)


