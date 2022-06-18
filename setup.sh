mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"bhanu0925@gmail.com\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[theme]\n\
 textColor='#f3eeee'\n\
 secondaryBackgroundColor='#161414'\n\
 font='sans serif'\n\
 primaryColor='#2961a2'\n\
 backgroundColor='#3b9cd0'\n\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml

