import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

import google.generativeai as genai
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser



from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# def get_pdf_text(pdf_docs):
#     text=""
#     for pdf in pdf_docs:
#         pdf_reader= PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text+= page.extract_text() or ""  # Ensure None is handled
#     return  text

# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         try:
#             pdf_reader = PdfReader(pdf)
#             for page in pdf_reader.pages:
#                 page_text = page.extract_text() or ""
#                 text += page_text
#         except Exception as e:
#             st.error(f"Error processing PDF {pdf.name}: {e}")
#             return "" # Return empty string to prevent further errors
#     print("Extracted Text:", text)
#     return text

def get_pdf_text(pdf_docs):
    text = ""
    if not pdf_docs:
        st.error("No PDFs uploaded.")
        return ""

    for pdf in pdf_docs:
        try:
            print(f"Processing PDF: {pdf.name}")  # Debugging
            pdf_reader = PdfReader(pdf)
            page_count = len(pdf_reader.pages)
            print(f"  Page Count: {page_count}")  # Debugging

            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text() or ""
                print(f"  Page {page_num + 1}: Text Length - {len(page_text)}")  # Debugging
                text += page_text

        except Exception as e:
            st.error(f"Error processing PDF {pdf.name}: {e}")
            return ""  # Stop on error, prevent further processing
    print("Final Extracted Text Length:", len(text))  # Debugging
    return text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


# def get_vector_store(text_chunks):
#     # ✅ Validate that text_chunks is a list of strings
#     if not all(isinstance(chunk, str) for chunk in text_chunks):
#         raise TypeError("text_chunks must be a list of strings")

#     embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")

def get_vector_store(text_chunks):
    if not text_chunks:
        st.error("No valid text chunks were generated from the PDFs.")
        return

    if not all(isinstance(chunk, str) for chunk in text_chunks):
        raise TypeError("text_chunks must be a list of strings")

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        st.success("Vector store created successfully.")
    except Exception as e:
        st.error(f"Error creating embeddings: {e}")
        return


# def get_conversational_chain():

#     prompt_template = """
    
#     Context:\n {context}?\n
#     Question: \n{question}\n

#     Answer:
#     """

#     model = ChatGoogleGenerativeAI(model="gemini-pro",
#                              temperature=0.3)

#     prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

#     return chain 

def get_conversational_chain():
    prompt_template = """
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3)  # Correct model name
    chain = (
        {"context": lambda x: "\n".join([doc.page_content for doc in x["input_documents"]]), "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return chain



def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        if not os.path.exists("faiss_index"):
            st.error("Vector store not found. Please upload and process PDFs first.")
            return

        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        if not docs:
            st.warning("No relevant information found in the documents.")
            return

        chain = get_conversational_chain()

        st.write("Generating Response...")

        # response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        response = chain.invoke({"input_documents": docs, "question": user_question})
        
        # st.write("Reply: ", response["output_text"])
        print(f"Response Type: {type(response)}") #add this line
        print(f"Response: {response}") #add this line
    
        st.write("Reply: ", response)

    except Exception as e:
        st.error(f"An error occurred: {e}")





def main():
    # st.set_page_config("Chat PDF")
    st.set_page_config(page_title="Chat PDF")

    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()