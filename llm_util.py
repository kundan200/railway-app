import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI
from langchain_community.callbacks import get_openai_callback

import langchain
langchain.verbose = False

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

class LLM:
  def __init__(self) -> None:
     self.knowledgeBase = ''
     self.read_pdf()
  
  def process_text(self, text):
    text_splitter = CharacterTextSplitter(
      separator="\n",
      chunk_size=1000,
      chunk_overlap=200,
      length_function=len
    )
    chunks = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    return knowledge_base
  
  def read_pdf(self):
    pdf_reader = PdfReader(open("train.pdf", "rb"))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    self.knowledgeBase = self.process_text(text)
    
  def answer_to_the_question(self, query):
    prompt = f"As the inquiry officer stationed at the railway station in India, when presented with the user's question, you must respond to the question in line with the provided context, Don't mention rather than that. If you're unable to provide an answer, kindly direct the user to seek assistance from nearby officers. The user query might have been misspelled since we used speech-to-text, So please understand the query with your intelligence. user's question as follows:  {query}"
    docs = self.knowledgeBase.similarity_search(query)
    llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    chain = load_qa_chain(llm, chain_type="stuff")
    with get_openai_callback() as cost:
        response = chain.invoke(input={"question": prompt, "input_documents": docs})
        # print(response["output_text"])
    return response["output_text"]