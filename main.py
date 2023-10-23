import streamlit as st
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.chains import LLMChain
import openai
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone

# Setando as chaves API através do Streamlit Secrets
pinecone.init(
    api_key=st.secrets["PINECONE_API_KEY"],
    environment=st.secrets["PINECONE_ENVIRONMENT"]
)

# Criando histórico de mensagens e buffer de memória para contexto no chat
history = StreamlitChatMessageHistory(key="messages") 
memory = ConversationBufferMemory(chat_memory=history)

# Acessando os embeddings para busca contextual pelas informações do vestibular
embeddings = OpenAIEmbeddings(openai_api_key="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")
docsearch = Pinecone.from_existing_index("unicampresolucao", embeddings)

# Set up the LLMChain, passing in memory
template = """Você é um chatbot tendo uma conversa com um humano.

{history}
Human: {human_input}
AI: """
prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)
llm_chain = LLMChain(llm=OpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"]), prompt=prompt, memory=memory)

if len(history.messages) == 0:
    history.add_ai_message("Como posso te ajudar?")

st.title("Vestibular Unicamp 2024")

# Criando o prompt com o argumento recuperado
def augment_prompt(query: str):
    results = docsearch.similarity_search(query, k=3)
    source_knowledge = "\n".join([x.page_content for x in results])
    augmented_prompt = f"""Usando o contexto abaixo, responda a query:

    Contexto:
    {source_knowledge}

    Query: {query}"""
    return augmented_prompt

# Mostrando histórico de mensagem ao reiniciar o app
for message in history.messages:
    st.chat_message(message.type).write(message.content)

if prompt := st.chat_input("Envie sua dúvida"):
    st.chat_message("human").write(prompt)
    response = llm_chain.run(augment_prompt(prompt))
    st.chat_message("ai").write(response)
