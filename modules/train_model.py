from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, f1_score
import streamlit as st

def train_random_forest_with_grid_search(X, y):
    # Codificar las categorías en números
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)  # Convierte las categorías en números

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    # Definir el modelo base
    model = RandomForestClassifier(random_state=42)

    # Definir los hiperparámetros a ajustar
    param_grid = {
        'n_estimators': [50, 100, 150],       # Número de árboles en el bosque
        'max_depth': [10, 20, 30],            # Profundidad máxima de cada árbol
        'min_samples_split': [2, 5, 10],      # Número mínimo de muestras requeridas para dividir un nodo
        'min_samples_leaf': [1, 2, 4]         # Número mínimo de muestras necesarias en cada hoja
    }

    # Definir el scoring como F1 Score con promedio ponderado
    scoring = make_scorer(f1_score, average='weighted')

    # Configurar GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring=scoring, n_jobs=-1)

    # Realizar la búsqueda de hiperparámetros
    grid_search.fit(X_train, y_train)

    # Mostrar los mejores parámetros y el mejor score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    st.write(f"Mejores Hiperparámetros de `random forest`: {best_params}")
    st.write(f"Mejor `F1 Score` obtenido: {best_score:.4f}")

    # Extraer el mejor modelo después de la búsqueda
    best_model = grid_search.best_estimator_

    # Realizar predicciones con el modelo entrenado
    y_pred = best_model.predict(X_test)

    # Retornar lo necesario para calcular métricas
    return y_test, y_pred, best_model, label_encoder
