import os

from typing import Annotated

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from dotenv import load_dotenv
from database import engine, SessionLocal
from sqlalchemy.orm import Session

import pinecone
from langchain.vectorstores.pinecone import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.output_parsers import CommaSeparatedListOutputParser


from services import get_youtube_transcript, convert_pdf_vector
from auth import (
    authenticate_user,
    get_current_user,
    create_access_token,
    get_password_hash,
)
import models
from settings import origins


app = FastAPI()

models.Base.metadata.create_all(bind=engine)


def get_db():
    try:
        db = SessionLocal()

        yield db
    finally:
        db.close()


db_dependency = Annotated[Session, Depends(get_db)]
user_dependency = Annotated[dict, Depends(get_current_user)]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up OpenAI API credentials


@app.post("/api/create_user")
async def create_user(user: models.NewUser, db: db_dependency):
    try:
        newuser = models.User()
        exist = db.query(models.User).filter_by(email=user.email).first()
        if exist is not None:
            return {
                "response": HTTPException(
                    status_code=400, detail="you already have an account login"
                )
            }

        newuser.email = user.email
        newuser.password = get_password_hash(password=user.password)

        db.add(newuser)
        db.commit()

        return {
            "response": HTTPException(
                status_code=201, detail=f"user {user.email} created successfully"
            )
        }

    except:
        return {"response": "Error occured please try again."}


@app.post("/token", response_model=models.Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
):
    print("userform", form_data.username)
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/api/login", response_model=models.Token)
async def login_for_access_token(
    form_data: models.NewUser, db: Session = Depends(get_db)
):
    user = authenticate_user(db, form_data.email, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/")
def read_root(user: user_dependency):
    if user:
        return {"answer": "welcome to my AI build"}

    return {"unauthorised"}


@app.post("/api/chat_pdf")
def chat_pdf(user: user_dependency, text: models.PDFCHAT):
    if user:
        load_dotenv()
        llm = ChatOpenAI()
        embeddings = OpenAIEmbeddings()

        pinecone.init(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENVIRONMENT"),
        )

        vectorstore = Pinecone.from_existing_index(
            index_name=os.getenv("PINECONE_INDEX_NAME"), embedding=embeddings
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
        )

        response = qa.run(text.message)

        return response

    else:
        return HTTPException(status_code=401, detail="not authorised")


@app.post("/api/chat")
def chat(user: user_dependency, message: models.Chat):
    if user:
        load_dotenv()

        llm = ChatOpenAI(temperature=0.6)
        print("message:", message.message)
        response = llm.predict_messages(messages=message.message)
        return response
    else:
        return HTTPException(status_code=401, detail="not authorised")


@app.post("/api/translate")
def translate(user: user_dependency, detail: models.Translate):
    if user:
        load_dotenv()

        template = """Translate the following text : {message} from {from_language} to {to_language}:"""

        translation_prompt = PromptTemplate(
            input_variables=["message", "from_language", "to_language"],
            template=template,
        )

        llm = ChatOpenAI(temperature=0, max_tokens=500, model="gpt-3.5-turbo")
        chain = LLMChain(llm=llm, prompt=translation_prompt, verbose=True)

        response = chain.run(
            message=detail.message,
            from_language=detail.from_language,
            to_language=detail.to_language,
        )

        return response
    else:
        return HTTPException(status_code=401, detail="not authorised")


@app.post("/api/ytsummerize")
async def youtube_video_summerizer(user: user_dependency, detail: models.Summerize):
    if user:
        load_dotenv()
        docs = get_youtube_transcript(url=detail.youtube_url)
        return docs
    else:
        return HTTPException(status_code=401, detail="not authorised")


@app.post("/api/business_name_generator")
def AI_business_name_generator(user: user_dependency, detail: models.BusinessName):
    if user:
        output_paser = CommaSeparatedListOutputParser()
        format_instructions = output_paser.get_format_instructions()
        print(detail)

        load_dotenv()

        template = """List 20 uqnuie business name in the "{industry}" industy, with this list of Keywords:"{keywords}" .return only the names and nothing else. \n\n {format_instructions}"""

        prompt = ChatPromptTemplate(
            messages=[HumanMessagePromptTemplate.from_template(template=template)],
            input_variables=["keywords", "industry"],
            partial_variables={"format_instructions": format_instructions},
        )

        _input = prompt.format_messages(
            keywords=detail.keyword, industry=detail.industry
        )

        model = ChatOpenAI(temperature=0, max_tokens=200, model="gpt-3.5-turbo")
        output = model(_input)
        response = output_paser.parse(output.content)

        return response
    else:
        return HTTPException(status_code=401, detail="not authorised")


@app.post("/api/sql_generator")
def generate_sql_code(user: user_dependency, text: models.SqlGenerator):
    if user:
        load_dotenv()

        template = (
            """You are an SQL query assistant.create an SQL query from this "{text}" """
        )

        prompt = PromptTemplate(input_variables=["text"], template=template)

        llm = ChatOpenAI(temperature=0, max_tokens=200, model="gpt-3.5-turbo")
        chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

        response = chain.run(text=text.detail)

        return response

    else:
        return HTTPException(status_code=401, detail="not authorised")


@app.post("/api/prompt_generator")
def generate_prompt(user: user_dependency, detail: models.PromptGenerator):
    if user:
        load_dotenv()

        template = """your a prompt design specialist, create prompt for a task with this detail: "{detail}".return only the generated prompt and nothing else. """

        prompt = PromptTemplate(input_variables=["detail"], template=template)

        llm = ChatOpenAI(temperature=0.7, max_tokens=200, model="gpt-3.5-turbo")
        chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

        response = chain.run(detail=detail.detail)

        return response

    else:
        return HTTPException(status_code=401, detail="not authorised")


@app.post("/api/convert_pdf")
def pdf_to_vector(pdf: models.PDFFILE):
    response = convert_pdf_vector(pdf=pdf.file)

    return response
