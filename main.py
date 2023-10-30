import streamlit as st
import pinecone

from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline, EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

# Setando as chaves API através do Streamlit Secrets
pinecone.init(
    api_key=st.secrets["PINECONE_API_KEY"],
    environment=st.secrets["PINECONE_ENVIRONMENT"]
)

# Criando histórico de mensagens e buffer de memória para contexto no chat
history = StreamlitChatMessageHistory(key="messages") 
memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input", chat_memory=history)

# Acessando os embeddings para busca contextual pelas informações do vestibular
embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
retriever = Pinecone.from_existing_index("unicampresolucao", embeddings).as_retriever()
llm = ChatOpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"], temperature=0)    

#Compressão dos documentos para filtrar partes relevantes
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=120, separator=". ")
redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[splitter, redundant_filter, relevant_filter]
)

compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever=retriever)

#Definindo o template do prompt
template = """Você é um chatbot tendo uma conversa com um humano.
Dado as seguintes partes extraídas da resolução do vestibular da Unicamp, crie uma resposta final concisa e direta com o que foi perguntado.
Se não for possível responder pelo contexto, diga que não possui a resposta.

{context}

{chat_history}
Human: {human_input}
Chatbot:"""
prompt = PromptTemplate(
    input_variables=["chat_history", "human_input", "context"], template=template
)
chain = load_qa_chain(
    OpenAI(temperature=0), chain_type="stuff", memory=memory, prompt=prompt, verbose=True
)


#Iniciando o chat
if len(history.messages) == 0:
    history.add_ai_message("Como posso te ajudar?")

st.title("Vestibular Unicamp 2024")

# Mostrando histórico de mensagem ao reiniciar o app
for message in history.messages:
    st.chat_message(message.type).write(message.content)

# Enviando mensagem e recebendo resposta
if prompt := st.chat_input("Envie sua dúvida"):
    documents = compression_retriever.get_relevant_documents(prompt, top_k=10)
    st.chat_message("human").write(prompt)
    response = chain({"input_documents": documents, "human_input": prompt})#, return_only_outputs=True)
    st.chat_message("ai").write(response["output_text"])
