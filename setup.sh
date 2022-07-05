mkdir -p ~/.streamlit/

echo "\
primaryColor = '#E694FF'
backgroundColor = '#00172B'
secondaryBackgroundColor = '#0083B8'
textColor = '#C6CDD4'
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
\n\
" > ~/.streamlit/config.toml*/