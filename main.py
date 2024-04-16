from fastapi import FastAPI
from pydantic import BaseModel

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from fastapi.middleware.cors import CORSMiddleware

from pinecone import Pinecone

import os
from os.path import join, dirname
from dotenv import load_dotenv

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# set up OPEN AI API base chat model
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=OPENAI_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)

index_pc_name = 'sample-movies'

index_pc = pc.Index(index_pc_name)

vector_store = PineconeVectorStore(index_pc, embeddings.embed_query, 'summary')

retriever = vector_store.as_retriever()

template = """I want you to use the data below, to answer the question. You are the person that holds movie library,
    and only can only answer based on the movies you have in your library.

    Data: {context}

    Question: {question}

    Helpful Answer:"""

custom_rag_prompt = PromptTemplate.from_template(template)

rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
)


class Body(BaseModel):
    query: str


@app.get("/api/generate")
async def root(query: Body):

    chat_response = rag_chain.invoke(query.query)

    return {
        'status': 'success',
        "message": "Query generated successfully!",
        "response": chat_response
    }
