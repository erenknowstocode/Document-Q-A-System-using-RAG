import boto3
import streamlit as st

from langchain_community.chat_models import BedrockChat
from langchain_community.embeddings import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA


# --------------------------------------------------
# AWS Bedrock Client
# --------------------------------------------------
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

# --------------------------------------------------
# Embeddings (Amazon Titan)
# --------------------------------------------------
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    client=bedrock
)

# --------------------------------------------------
# Data Ingestion
# --------------------------------------------------
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )

    return splitter.split_documents(documents)

# --------------------------------------------------
# Create FAISS Vector Store
# --------------------------------------------------
def create_vector_store(docs):
    vectorstore = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore.save_local("faiss_index")

# --------------------------------------------------
# Claude 3 Haiku (CORRECT Chat Model)
# --------------------------------------------------
def get_claude_chat_model():
    return BedrockChat(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        client=bedrock,
        model_kwargs={
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 512,
            "temperature": 0.2
        }
    )

# --------------------------------------------------
# Prompt Template
# --------------------------------------------------
PROMPT_TEMPLATE = """
Use the following context to answer the question.
Provide a detailed explanation (around 250 words).
If you do not know the answer, say you do not know.

<context>
{context}
</context>

Question: {question}
"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

# --------------------------------------------------
# RAG QA Chain
# --------------------------------------------------
def get_response(llm, vectorstore, query):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        ),
        chain_type_kwargs={"prompt": PROMPT}
    )

    result = qa_chain.invoke({"query": query})
    return result["result"]

# --------------------------------------------------
# Streamlit App
# --------------------------------------------------
def main():
    st.set_page_config(
        page_title="Chat with PDF using AWS Bedrock",
        layout="wide"
    )

    st.header("Chat with PDF using AWS Bedrock (Claude 3 + RAG)")

    user_question = st.text_input("Ask a question based on the PDF documents")

    with st.sidebar:
        st.title("Vector Store")

        if st.button("Create / Update Vectors"):
            with st.spinner("Processing PDFs..."):
                docs = data_ingestion()
                create_vector_store(docs)
                st.success("Vector store created successfully")

    if user_question:
        if st.button("Generate Answer (Claude 3 Haiku)"):
            with st.spinner("Generating answer..."):
                vectorstore = FAISS.load_local(
                    "faiss_index",
                    bedrock_embeddings,
                    allow_dangerous_deserialization=True
                )

                llm = get_claude_chat_model()
                answer = get_response(llm, vectorstore, user_question)
                st.write(answer)


if __name__ == "__main__":
    main()
