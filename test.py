import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from PyPDF2 import PdfReader
#Please install PdfReader

from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')


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
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

pdf_docs = [open("train.pdf","rb")]
raw_text = get_pdf_text(pdf_docs)
text_chunks = get_text_chunks(raw_text)
vectorstore = get_vectorstore(text_chunks)
conversation_chain = get_conversation_chain(vectorstore)
while True:
    user_question = input("You: ")
    response = conversation_chain.invoke({'question': user_question})
    print(f"GPT: {response.get('answer')}")