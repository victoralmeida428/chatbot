import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.error(f'{pdf.name} - {e}')
    
    return text

def get_chunk_text(text):
    text_spliter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_spliter.split_text(text)
    return chunks

def get_embedding(chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl',
                                               model_kwargs={"device": "cpu"})
    vectore_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectore_store

def main():
    st.set_page_config('ChatBot ', page_icon=':robot_face:')

    st.title('ChatBox')
    with st.sidebar:
        files = st.file_uploader('Insert Document', accept_multiple_files=True)
        button = st.button('Process', 'btn_process')
        if button:
            with st.spinner('processing...'):
                
                if files[0].type == 'application/pdf':
                    raw_text = get_pdf_text(files)
                    chunk_text = get_chunk_text(raw_text)
                    embeddings = get_embedding(chunk_text)
                    
                else:
                    st.error('type file is not supported')

if __name__=='__main__':
    main()                
