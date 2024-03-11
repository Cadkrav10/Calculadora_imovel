import streamlit as st
import pickle as pk
#import 

# Título do aplicativo
st.title('Quanto custa sua casa ?')

# Adicione um widget de botão
#button = st.button('Clique em mim!')

# Exiba uma mensagem quando o botão for clicado
#if button:
#    st.write('O botão foi clicado!')

filename = 'modelo_treinado.sav'
modelo_treinado = pk.load(open(filename, 'rb'))

scaler = pk.load(open('scaler.pkl', 'rb'))

col1, col2 = st.columns(2)
with col1:
    quartos = st.slider("Quartos",
                        value=1,
                        min_value=0,
                        max_value=5,
                        step=1)
    
    with col2:
        banheiros = st.slider("Banheiros",
                        value=1,
                        min_value=0,
                        max_value=5,
                        step=1)
        
#---------------

col3, col4 = st.columns(2)
with col3:
    vagas = st.slider("Vagas",
                        value=1,
                        min_value=0,
                        max_value=5,
                        step=1)
    
    with col4:
        area = st.number_input("Area (m2)")
        

X_novo = [[area, quartos, banheiros, vagas]]
X_novo_scaled = scaler.transform(X_novo)

y_predito = modelo_treinado.predict(X_novo_scaled)    

st.divider()

#st.write(y_predito)                    
st.write(f'<span style="font-family: Arial, sans-serif; font-size: 18px; color: black;">{y_predito}</span>', unsafe_allow_html=True)

