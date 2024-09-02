from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def display_classification_metrics(y_test, y_pred, label_encoder):
    # Matriz de Confusión
    st.write("### Matriz de Confusión:")
    # Calcular la matriz de confusión
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Graficar la matriz de confusión
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    st.pyplot(plt)
    # Generar el reporte de clasificación
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

    # Mostrar el reporte en Streamlit
    st.subheader("Reporte de Clasificación:")
    st.text(report)

    st.write("### Análisis de Métricas por Clase:")
    st.write("**`1. Financial and Savings`**")
    st.write("- **Precision:** 1.00")
    st.write("- **Recall:** 1.00")
    st.write("- **F1-score:** 1.00")
    st.write("- **Conclusión:** El modelo predice esta categoría con una `precisión perfecta`. No hay falsos positivos ni falsos negativos, lo que indica un `excelente rendimiento` para esta clase.")

    st.write("**`2. Health and Wellness`**")
    st.write("- **Precision:** 1.00")
    st.write("- **Recall:** 0.99")
    st.write("- **F1-score:** 1.00")
    st.write("- **Conclusión:** La `precisión es perfecta`, pero el `recall es ligeramente menor`, lo que sugiere que casi todas las transacciones reales se capturan correctamente. Solo hay un pequeño número de falsos negativos.")

    st.write("**`3. Payments for Services and Basic Needs`**")
    st.write("- **Precision:** 0.99")
    st.write("- **Recall:** 0.99")
    st.write("- **F1-score:** 0.99")
    st.write("- **Conclusión:** El modelo maneja bien esta clase con un muy buen equilibrio entre precisión y recall. Hay una `pequeña cantidad de errores`, pero en general, el modelo clasifica esta categoría de manera efectiva.")

    st.write("**`4. Personal Care, Entertainment, and Education`**")
    st.write("- **Precision:** 1.00")
    st.write("- **Recall:** 0.80")
    st.write("- **F1-score:** 0.89")
    st.write("- **Conclusión:** Aunque la `precisión es alta`, el `recall es significativamente más bajo`, lo que indica que el modelo no captura una proporción considerable de transacciones reales de esta clase. Podría beneficiarse de ajustar los datos de entrenamiento o las características para mejorar la detección.")

    st.write("**`5. Shopping and Consumption`**")
    st.write("- **Precision:** 1.00")
    st.write("- **Recall:** 0.92")
    st.write("- **F1-score:** 0.96")
    st.write("- **Conclusión:** El modelo tiene una `alta precisión` pero `pierde algunos casos` reales. Esto sugiere que puede haber ligeras confusiones con otras categorías, pero en general, el rendimiento sigue siendo sólido.")

    st.write("###`Conclusiones Generales:`")
    st.write("- El modelo demuestra un `rendimiento muy fuerte` en general, con altas métricas de precisión, recall y F1-score en casi todas las categorías.")
    st.write("- La clase `Personal Care, Entertainment, and Education` muestra un recall `notablemente más bajo`, lo que indica que podría beneficiarse de un enfoque de ajuste adicional, como aumentar los datos de entrenamiento de esa clase o ajustar los hiperparámetros para capturar mejor las instancias.")
    st.write("- La `precisión` en todas las clases es `excelente`, lo que significa que el modelo tiene muy pocos falsos positivos, lo cual es crucial para evitar clasificaciones incorrectas en un contexto de análisis financiero.")
