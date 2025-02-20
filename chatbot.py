import streamlit as st

import requests
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)

url = "https://neuralsearchx-dev.azure-api.net"
    
# Criando histórico de mensagens e buffer de memória para contexto no chat
history = StreamlitChatMessageHistory(key="messages") 
memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input", chat_memory=history)


#Iniciando o chat
if len(history.messages) == 0:
    history.add_ai_message("Como posso te ajudar?")

st.title("OBMEP 2025")

# Mostrando histórico de mensagem ao reiniciar o app
for message in history.messages:
    st.chat_message(message.type).write(message.content)

# Enviando mensagem e recebendo resposta
if prompt := st.chat_input("Envie sua dúvida"):
    st.chat_message("human").write(prompt)
    response = requests.get(
        url+'/search/search-engine',
        params={
            'collection_id': '1c518227-43c5-4b33-8e72-0a4757d2bbe8',
            'query': prompt,
            'enable_deduplication': 'true'
        }
    )
    response = response.json()['predicted_answer']
    st.chat_message("ai").write(response)