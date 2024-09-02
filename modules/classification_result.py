import streamlit as st
from modules.description_dataset import plot_categorical_distribution_with_pareto

def pre_classification_result(df, tittle):
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
        return 'Miscellaneous'  # Default group if not matched

    # Aplicar la función al DataFrame
    df['group_category'] = df['category'].apply(assign_group)

    # Gráfico para Depósitos
    st.subheader(tittle)
    plot_categorical_distribution_with_pareto(
        column_name='group_category',
        data= df,
        color='green'
    )

    