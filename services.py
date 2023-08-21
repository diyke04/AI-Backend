import os
import sys
import tiktoken
import pinecone
from dotenv import load_dotenv
from typing import List
from langchain.document_loaders import YoutubeLoader,PyPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter,TokenTextSplitter





def get_youtube_transcript(url:str)->str:
    loader =YoutubeLoader.from_youtube_url(youtube_url=url)
    docs=loader.load()
    texts=docs[0].page_content
    summary=generate_summary(text=texts)

    return summary
   

def generate_summary(text:str)->str:
    token_to_use=calculate_text_token(text=text,model='gpt-3.5-turbo-16k')

    if token_to_use >15000:
         print('high token to be used')
         text_chunks=get_text_chunks_1(text=text)

         all_summary=''

         for text in text_chunks:
            llm =ChatOpenAI(temperature=0,model='gpt-3.5-turbo-16k')
            template = """summerize this piece of text in detail and concise format "{text}" . """

            prompt = PromptTemplate(
                    input_variables=["text"], template=template
                )

            chain = LLMChain(llm=llm, prompt=prompt,verbose=False)
            summary=chain.run(text=text)
            all_summary+=summary
        
         #with open(f'{all_summary[:20]}.txt','w') as file:
                #file.write(all_summary)
            
                
    else:
           print('low token to be used')

           all_summary=''
           
           llm =ChatOpenAI(temperature=0,model='gpt-3.5-turbo-16k')
           template = """summerize this piece of text in detail and concise format "{text}" . """

           prompt = PromptTemplate(
                    input_variables=["text"], template=template
                )

           chain = LLMChain(llm=llm, prompt=prompt,verbose=True)
           summary=chain.run(text=text)
           all_summary+=summary
        
           #with open(f'{all_summary[:20]}.txt','w') as file:
                #file.write(all_summary)

    return all_summary

def get_text_chunks_1(text:str)->List[str]:
    text_splitter = TokenTextSplitter(chunk_size=10000, chunk_overlap=0)
    chunks = text_splitter.split_text(text=text)
    

    return chunks

def get_text_chunks_2(text:str)->List[str]:
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_text(text=text)

    return chunks


def calculate_text_token(text:str,model:str)->int:
     encoding = tiktoken.get_encoding("cl100k_base")
     encoding = tiktoken.encoding_for_model(model)
     tokens_integer=len(encoding.encode(text))
     #print(f"{len(tokens_integer)} is the number of tokens in my text")
     return tokens_integer



def convert_pdf_vector(pdf):
     
     load_dotenv()
     
     pdf=os.path.abspath(pdf)
     loader=PyPDFLoader(pdf)
     docs=loader.load_and_split()
     print(sys.getrecursionlimit())

     sys.setrecursionlimit(1500)

     print(sys.getrecursionlimit())



     embeddings=OpenAIEmbeddings()

     pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))

     vector=Pinecone.from_documents(documents=docs,embedding=embeddings,index_name=os.getenv("PINECONE_INDEX_NAME"))

    

     return vector
