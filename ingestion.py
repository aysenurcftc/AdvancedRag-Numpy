from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

urls = [
    "https://numpy.org/doc/stable/user/whatisnumpy.html",
    "https://numpy.org/install/",
    "https://numpy.org/doc/stable/user/absolute_beginners.html",
    "https://numpy.org/doc/stable/user/basics.creation.html",
    "https://numpy.org/doc/stable/user/basics.indexing.html",
    "https://numpy.org/doc/stable/user/basics.types.html",
    "https://numpy.org/doc/stable/user/basics.broadcasting.html",
    "https://numpy.org/doc/stable/user/basics.copies.html",
    "https://numpy.org/doc/stable/user/basics.strings.html",
    "https://numpy.org/doc/stable/user/basics.rec.html",
    "https://numpy.org/doc/stable/user/quickstart.html",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

vectorstore = Chroma.from_documents(
     documents=doc_splits,
     collection_name="rag-chroma",
     embedding=OpenAIEmbeddings(),
     persist_directory="./.chroma",
 )

retriever = Chroma(
    collection_name="rag-chroma",
    persist_directory="./.chroma",
    embedding_function=OpenAIEmbeddings(),
).as_retriever()