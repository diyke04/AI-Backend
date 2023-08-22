from pydantic import BaseModel, FilePath
from typing import List

from sqlalchemy import Column, Integer, String
from database import Base


class Translate(BaseModel):
    message: str
    from_language: str
    to_language: str


class Summerize(BaseModel):
    youtube_url: str


class BusinessName(BaseModel):
    keyword: List[str]
    industry: str


class PromptGenerator(BaseModel):
    detail: str


class SqlGenerator(BaseModel):
    detail: str


class Chat(BaseModel):
    message: List[str]


class PDFCHAT(BaseModel):
    message: str


class PDFFILE(BaseModel):
    file: FilePath


class NewUser(BaseModel):
    email: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    email: str


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, index=True, primary_key=True)
    email = Column(String, unique=True, index=True)
    username = Column(String)
    password = Column(String)


primary_key = (True,)
