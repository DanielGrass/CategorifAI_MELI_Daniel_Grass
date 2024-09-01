import pandas as pd
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from fuzzywuzzy import process
from concurrent.futures import ThreadPoolExecutor
from stqdm import stqdm
import streamlit as st


def transactions_details_preprocesing(df):
    # Load the English spaCy model
    nlp = spacy.load("en_core_web_sm")

    # Load stopwords in English
    stop_words = set(stopwords.words('english'))
    
    # Text preprocessing function
    def preprocess(texto, columna):        
        caracteres_especiales = r"[^\w\s]"  # Cualquier caracter que no sea alfanumérico o espacio
        def clean_string(texto):
            texto = texto.lower()  # Convert to lowercase
            # Eliminar números y caracteres especiales
            texto = re.sub(r'\d+', '', texto)  # Eliminar números
            texto = re.sub(caracteres_especiales, '', texto)  # Eliminar caracteres especiales
            return texto
        
        return df[columna].apply(clean_string)
    
    st.write("Número de transaction_details únicas (sin preprocesar):", df['transaction_details'].nunique())
    # Aplicar la función a la columna deseada y crear una nueva columna
    df['transaction_details_non_nan'] = df['transaction_details'].fillna(value="nan")
    df['transaction_details_clean'] = preprocess(df, 'transaction_details_non_nan')
    
    #df['transaction_details_clean'] = df['transaction_details_non_nan'].apply(preprocess)
    st.write("Número de transaction_details únicas (eliminando caracteres especiales y números):", df['transaction_details_clean'].nunique())

    # Extract common words
    all_words = ' '.join(df['transaction_details_clean']).split()
    word_freq = Counter(all_words)
    common_words = word_freq.most_common(10)

    # Extract keywords using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['transaction_details_clean'])
    keywords = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.sum(axis=0).A1
    keywords_tfidf = sorted(list(zip(keywords, tfidf_scores)), key=lambda x: x[1], reverse=True)
    
    # Group similar words using fuzzy matching
    def group_similar_words(word_list):
        grouped = {}
        for word in word_list:
            match = process.extractOne(word, grouped.keys())
            if match and match[1] > 80:  # Set a similarity threshold
                grouped[match[0]].append(word)
            else:
                grouped[word] = [word]
        return grouped

    # grouped_keywords = stqdm(group_similar_words([kw[0] for kw in keywords_tfidf]))

    # Display results in Streamlit
    # st.write("### Identified Keywords")
    # st.write("**Most Common Words**:", common_words)
    # st.write("**Keywords by TF-IDF**:", keywords_tfidf[:10])
    # st.write("**Grouped Keywords**:", grouped_keywords)

    # # Display identified keywords in DataFrame
    # st.dataframe(df[['transaction_details', 'cleaned_details']])
