import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
import base64
import textwrap
from langchain_community.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from constants import CHROMA_SETTINGS

checkpoint = "model/LaMini-T5-738M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    device_map="auto",
    torch_dtype = torch.float32,
    offload_folder="offload"
)

@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model = base_model,
        tokenizer=tokenizer,
        max_length=500,
        do_sample=True,
        temperature=0.3,
        top_p=0.95
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

def qa_llm():
    llm = llm_pipeline()
    local_embedding_model = SentenceTransformer("/model/all-MiniLM-L6-v2")
    embeddings =  SentenceTransformerEmbeddings(model=local_embedding_model)
    db = Chroma(persist_directory="db",embedding_function=embeddings,client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = retriever,
        return_source_documents = True
    )
    return qa

def process_answer(instruction):
    response = ''
    instruction=instruction
    qa=qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer,generated_text

def main():
    st.title("Search in your PDF")
    with st.expander("About the App"):
        st.markdown(
            """
This is Generative AI powered question & answering app that responds to questions about the PDF files.
"""
        )
        question = st.text_area("Enter your question")
        if st.button("Search"):
            st.info("Your Question: "+question)
            st.info("Your Answer: ")
            answer, metadata = process_answer(question)
            st.write(answer)
            st.write(metadata)


if __name__ == "__main__":
    main()
