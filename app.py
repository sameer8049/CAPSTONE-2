import streamlit as st
from streamlit_chat import message   #This library provides a component for displaying chat-like conversations in a Streamlit app

from langchain.chains import ConversationalRetrievalChain #This chain likely combines elements of language modeling, vector retrieval, and conversation history management to provide an integrated question-answering solution.

from langchain.embeddings import HuggingFaceEmbeddings #to embed text data into a vector space for similarity calculations.

from langchain.llms import LlamaCpp #LlamaCpp is used as a component to generate responses to user queries.It likely employs a pre-trained model, "Mistral-7B-Instruct," to provide contextually relevant answers to questions.

from langchain.text_splitter import RecursiveCharacterTextSplitter #used to split text documents, such as PDFs, into smaller chunks for processing

from langchain.vectorstores import FAISS #this is langchain part #https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/

from langchain.memory import ConversationBufferMemory #which is used to manage the conversation history. It stores previous user queries and the chatbot's responses, ensuring that the chatbot has access to prior interactions. This helps maintain context and provides more relevant responses as the conversation progresses.

from langchain.document_loaders import PyPDFLoader #This component is used to load text from PDF documents
import os
import tempfile




def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your PDF", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat(user_input, chain, st.session_state['history'])

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="fun-emoji")
                message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")

def create_conversational_chain(vector_store):
    # Create llm
    llm = LlamaCpp(
    streaming = True,
    model_path="mistral-7b-instruct-v0.1.Q3_K_M.gguf",
    temperature=0.75,
    top_p=1, 
    verbose=True,
    n_ctx=4096
)
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                 retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                 memory=memory)
    return chain

def main():
    # Initialize session state
    initialize_session_state()
    st.title("Multi-PDF ChatBot using Mistral-7B-Instruct :books:")
    # Initialize Streamlit
    st.sidebar.title("Document Processing")
    uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)


    if uploaded_files:
        text = []
        for file in uploaded_files:
            file_extension = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            loader = None
            if file_extension == ".pdf":  #.pdf format only 
                loader = PyPDFLoader(temp_file_path)

            if loader:
                text.extend(loader.load())
                os.remove(temp_file_path)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=20)  #chunk_Overlap: The chunk_overlap parameter specifies how much overlap there is between consecutive chunks. Overlapping can help ensure that important information is not split across chunks
        text_chunks = text_splitter.split_documents(text)

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                           model_kwargs={'device': 'cpu'})

        # Create vector store using FAISS
        vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

        # Create the chain object
        chain = create_conversational_chain(vector_store)

        
        display_chat_history(chain)

if __name__ == "__main__":
    main()
