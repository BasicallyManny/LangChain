#This script is to help me get the gist of embeddings,vectir, Databases and, VectorDBQA chain and RetrievalIQA

import os

from dotenv import find_dotenv, load_dotenv #imports to allow us to find and load enviroment variables

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from pinecone import Pinecone
from langchain.vectorstores import Pinecone as PineconeStore

load_dotenv()
load_dotenv(find_dotenv())

#initialize vector database
Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))

if __name__ == '__main__':

    #Load in the text document
    print("Vector Store")
    loader=TextLoader("/Users/Manny/OneDrive/Desktop/LangChain/Assets/mediumblog1.txt",encoding="utf8")
    document=loader.load()

    #split the chunks into its specific chunks
    text_splitter=CharacterTextSplitter(chunk_size=50,chunk_overlap=0)
    text = text_splitter.split_documents(document)
    print(len(text))
    #print(document)
    #Initiate the Embeddings
    embeddings = OpenAIEmbeddings(open_api_key=os.environ.get('OPENAI_API_KEY'))
    #PineCode: Vector Databse
    docsearch = PineconeStore.from_documents(text,embeddings,index_name="medium-blogs-embedding-index") #convert text into vectors