from fastapi import FastAPI

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import CSVLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

import os

app = FastAPI()

with open('hidden.txt') as file:
    apiKey = file.readline()
    os.environ["OPENAI_API_KEY"] = apiKey
cv_file_path = './document/cvs.csv'
employee_info_file_path = './document/employeeinfo.csv'
text_loader = DirectoryLoader('./document/', glob='**/*.txt')
cv_csv_loader = CSVLoader(file_path=cv_file_path, source_column="employee_name", encoding='utf-8')
employee_info_loader = CSVLoader(file_path=employee_info_file_path, source_column="employee_name", encoding='utf-8')

# loaders = [text_loader, cv_csv_loader, employee_info_loader]
# data = []
#
# for loader in loaders:
#     data.extend(loader.load())

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

employeeInfoDocument = text_splitter.split_documents(employee_info_loader.load())
employeeCvDocument = text_splitter.split_documents(cv_csv_loader.load())
medicalBenefit = text_splitter.split_documents(text_loader.load())

embedding = OpenAIEmbeddings()
employeeInfoVectorstore = Chroma.from_documents(employeeInfoDocument, embedding)
employeeCvVectorstore = Chroma.from_documents(employeeCvDocument, embedding)
medicalBenefitVectorstore = Chroma.from_documents(medicalBenefit, embedding)

employeeInfoQa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0),
                                                       employeeInfoVectorstore.as_retriever())
employeeCvQa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0), employeeCvVectorstore.as_retriever())
medicalBenefitQa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0),
                                                         medicalBenefitVectorstore.as_retriever())

chatHistory = []


# def ge_api_response(prompt: str) -> str | None:
#     text: str | None = None
#     try:
#         response: dict = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",
#             prompt=prompt,
#             temperature=0.8,
#             max_tokens=60,
#             frequency_penalty=0,
#             presence_penalty=0,
#             suffix="Hy There!",
#         )
#         choises: dict = response.get('choices')[0]
#         text = choises.get('text')
#     except Exception as e:
#         print('Error',e)
#
#     return text


@app.get("/")
async def root():
    return {"message": "Welcome to WizzBuddy"}


@app.get("/{qa}/{question}")
async def say_hello(qa: str, question: str):
    if qa == 'einfo':
        result = employeeInfoQa({"question": question, "chat_history": chatHistory}).get('answer')
    elif qa == 'ecv':
        result = employeeCvQa({"question": question, "chat_history": chatHistory}).get('answer')
    else:
        result = medicalBenefitQa({"question": question, "chat_history": chatHistory}).get('answer')
    return {"message": result}
