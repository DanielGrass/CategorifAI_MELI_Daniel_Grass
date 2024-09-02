mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"dsgrassg@unal.edu.co\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
