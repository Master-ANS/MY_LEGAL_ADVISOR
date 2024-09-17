import os
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_conversational_chain():
    prompt_template = """
    If someone gives you the offence or felony that they have commited so please provide them with the information on - what are the sections under the indian penal code and other provided documents 
    would they be charged aand what are the section that will or can be imposed on them according to the offence/felony 
    and also tell the maximum imprisonment or fine that can be imposed on them , maximum imprisonment is calculated by taking maximum of the  imprisonment 
    duration under all the section , the maximum imprisonment time is not the addition of the imprisonment time uundeer all the section but the maximum of the imprisonment time under those section,
    , 
    Always give information about all the sections possible
    ,

    also always , tell the user whether the sections that will/can be imposed on them will be bailable or non-bilable -- Explicitly mentiaon and highlight the bailable and non bailable part of the case
    , 
    always end your answer with - "Consult to a legal advisor for more information";
    ,

    Explain everything in breif in breif ,

    ,
    on the prompt above explicity mention under a heading the maximum imprisonment or  maximum fine or both

    ,

    Dont write "Important Note: The actual charges and punishment will depend on the specific circumstances of the case,
      including the severity of the injury, the intent behind the act, and the evidence presented."
      these kind of things or similar things

    \n\n
    Context:\n{context}?\n
    Question:\n{question}\n

    Answer: 
    """

    model =  ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=1)
    prompt = PromptTemplate(template = prompt_template , input_variables=["context" , "question"])
    chain = load_qa_chain(model , chain_type = "stuff" , prompt = prompt)
    return chain
# Aysuh navneet singh

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    newDb = FAISS.load_local("vectorstore2", embeddings, allow_dangerous_deserialization=True)
    docs = newDb.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents":docs, "question":user_question},
        return_only_outputs=True)
    print(response)
    st.write("Reply: " , response["output_text"])


def main():
    st.set_page_config("Chat withh lawyer 👩🏻‍⚖️")
    st.header("Chat with Lawyer")

    user_question = st.text_input("Ask about your legal matters with out AI Lawyer")

    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()















