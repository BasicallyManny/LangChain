from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter #Breaks huge transcripts into smaller chunks for vector storage
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

#vector Database
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA

#load enviroments
from dotenv import load_dotenv
load_dotenv()
#assign embedding variable
embeddings =OpenAIEmbeddings()

#get yt video and convert it to vectors and store it in PineCone
def createVectorDBfromURL(video_url:str)->PineconeVectorStore:
    #load youtube video from URL
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcripts = loader.load()#get youtube video transcript

    #Split the transcript into chunks to be embedded into vector data base
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)#Split into chunks and idenintify chunk overlap
    docs=text_splitter.split_documents(transcripts)
    #assign database
    db = PineconeVectorStore.from_documents(docs,embeddings,index_name="medium-blogs-embedding-index")#convert chunks into vectors
    #dont forget to specify index name when using Pinecone database
    return db

#get the result from query (The Question that the user asks)
# @param k: The number of documents thats sent to the LLM
def get_response_from_query(db,query,k=4): #if k=4 we will only be sending 4 document chunks to the LLm
    docs=db.similarity_search(query,k=k)#run similarity search based on query
    docs_page_content=" ".join([d.page_content for d in docs])#join those 4 docs to create 1 doc to keep within the token limit

    llm=OpenAI()#initiate llm wwith text model; choose on openAI website

    #create prompt template
    prompt = PromptTemplate(
        input_variables=["question","docs"],
        #@Param: Docs is the similarity search we did above
        template="""                    
                     You are a helpful Youtube Assistant that can answer questions about videos
                     based on the video's transcript.set

                     Anser the following question: {question}
                     By searching the following video transcript: {docs}

                     Only use the factual information from the transcript to answer the questions.

                     If you feel like you don't have enough information to answer the question,
                     say "I need more information"

                     Your answers should be detailed
                 """,
    )

    #create llm chain
    #Chains refer to sequences of calls - whether to an LLM, a tool, or a data preprocessing step. 
    chain=LLMChain(llm=llm, prompt=prompt)
    #run the chain to get a response from the LLM
    response=chain.run(question=query, docs=docs_page_content)
    response=response.replace("\n"," ")#format the response. Replace new lines with a space
    return response,docs
