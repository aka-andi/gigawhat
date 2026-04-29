from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from utils import ingest_sources, format_docs
from configs import PROMPT_TEMPLATE, EMBEDDING_MODEL
import os

""" 1. INGEST DATA """

if not os.path.exists("data/chroma_db") or os.listdir("data/raw/new"):
    ingest_sources()

""" 2. BUILD COMPONENTS + CHAIN """

vector_store = Chroma(
    persist_directory="data/chroma_db",
    embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL)
)
retriever = vector_store.as_retriever(search_kwargs={"k": 5}) # k = tunable chunk count

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

llm = ChatOpenAI(model="gpt-4o-mini")

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

""" 3. RETRIEVAL AND GENERATION VIA USER INPUT LOOP  """

# example question: What are the projected US energy consumption trends?
user_input_question = "What would you like to ask related to energy? (hit 'Enter' if finished)\n"
user_input = input(user_input_question)

while user_input != "":
    response = chain.invoke(user_input)
    print("\n========================================\n")
    print(response)
    data_sources = retriever.invoke(user_input)
    print("\nSources:", [data.metadata.get("source", "unknown") for data in data_sources])
    user_input = input(user_input_question)
