# theme_toggle.py
import streamlit as st

# Inicializar el estado del tema
def initialize_theme():
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = True  # Por defecto, inicia en modo claro

# Función para alternar el tema
def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

# Función para aplicar los estilos CSS según el modo seleccionado
def apply_styles():
    if st.session_state.dark_mode:
        # Modo oscuro
        st.markdown(
            """
            <style>
            .main {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            [data-testid="stSidebar"] {
                background-color: #333333;
                color: #ffffff;
            }
            h1, h2, h3, .st-expander, .st-expander p, .st-expander div {
                color: #ffffff;
            }
            .stButton>button {
                color: white;
                background-color: #003087;
                border-radius: 8px;
                border: 2px solid #ffd700;
            }
            .stButton>button:hover {
                background-color: #ffd700;
                color: #003087;
            }
            .stRadio div {
                color: #ffffff; /* Estilo del texto de los radio buttons en modo oscuro */
            }
            .stAlert {
                background-color: #2c2c2c;
                color: #ffffff;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        # Modo claro
        st.markdown(
            """
            <style>
            .main {
                background-color: #ffffff;
                color: #003087;
            }
            [data-testid="stSidebar"] {
                background-color: #f5f5f5;
                color: #003087;
            }
            h1, h2, h3, .st-expander, .st-expander p, .st-expander div {
                color: #003087;
            }
            .stButton>button {
                color: white;
                background-color: #003087;
                border-radius: 8px;
                border: 2px solid #ffd700;
            }
            .stButton>button:hover {
                background-color: #ffd700;
                color: #003087;
            }
            .stRadio div {
                color: #003087; /* Estilo del texto de los radio buttons en modo claro */
            }
            .stAlert {
                background-color: #e0e0e0;
                color: #003087;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
