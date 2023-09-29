import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import Replicate
from langchain.embeddings import HuggingFaceEmbeddings
from htmlTemplates import css, bot_template, user_template
from dotenv import load_dotenv
import os

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    os.environ['REPLICATE_API_TOKEN'] = "r8_NRsPmNsSub7ZKmDDn4RYqnWg8PrpZEU2IwL1H"
    llm = Replicate(
        model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
        input={"temperature": 0.75, "max_length": 3000}
        )

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()

















# import streamlit as st
# from langchain.llms import OpenAI
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.chains import RetrievalQA

# from langchain.document_loaders import PyPDFLoader, OnlinePDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import Pinecone
# from langchain.llms import Replicate
# from sentence_transformers import SentenceTransformer
# from langchain.chains.question_answering import load_qa_chain
# from langchain.chains import ConversationalRetrievalChain

# import pinecone
# import os
# import sys
# os.environ['CURL_CA_BUNDLE'] = ''
# from langchain.llms import HuggingFaceHub
# import subprocess
# import io
# from tempfile import NamedTemporaryFile







# # Page title
# st.set_page_config(page_title='Query PDFs App')
# st.title('Query PDFs App')

# # File upload
# uploaded_file = st.file_uploader('Upload an article', type='pdf')
# # Query text
# query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not uploaded_file)

# # Replicate API token
# os.environ['REPLICATE_API_TOKEN'] = "r8_NRsPmNsSub7ZKmDDn4RYqnWg8PrpZEU2IwL1H"

# # Initialize Pinecone
# pinecone.init(api_key='d7cc45ff-30de-49e2-baa2-584056f46b1c', environment='gcp-starter')

# if uploaded_file is not None:
#     pdf_data = uploaded_file.read()
#     with NamedTemporaryFile(delete=False) as tmp:  # open a named temporary file
#         tmp.write(pdf_data)                      # write data from the uploaded file into it
#         data = PyPDFLoader(tmp.name).load()        # <---- now it works!
#     os.remove(tmp.name)

#     # Convert the result from bytes to a string and display it to the user
#     text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
#     docs=text_splitter.split_documents(data)
#     embeddings = HuggingFaceEmbeddings()


#     # Set up the Pinecone vector database
#     index_name = "langchaintest"
#     index = pinecone.Index(index_name)
#     vectordb = Pinecone.from_documents(docs, embeddings, index_name=index_name)

#     # Initialize Replicate Llama2 Model
#     llm = Replicate(
#         model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
#         input={"temperature": 0.75, "max_length": 3000}
#         )

#     # Set up the Conversational Retrieval Chain
#     qa_chain = ConversationalRetrievalChain.from_llm(
#         llm,
#         vectordb.as_retriever(search_kwargs={'k': 2}),
#         return_source_documents=True
#     )
#     # Start chatting with the chatbot
#     chat_history = []
#     if query_text.lower() in ["exit", "quit", "q"]:
#         print('Exiting')
#         sys.exit()
#     result = qa_chain({'question': query_text})
#     answer = result['answer']
#     st.write(answer)
