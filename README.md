# Chat with PDF using AWS Bedrock (RAG)

This project is a **Retrieval-Augmented Generation (RAG)** application that allows users to ask questions from a collection of PDF documents.  
It uses **AWS Bedrock** for embeddings and text generation, **FAISS** for vector similarity search, and **Streamlit** for a simple interactive UI.

The project is built as a hands-on PoC to understand how **LLMs can be combined with document retrieval** to build practical question-answering systems.

## Overview

The application workflow is as follows:

- PDF documents are loaded from a local directory
- Documents are split into chunks for efficient retrieval
- Text embeddings are generated using **Amazon Titan Embeddings**
- Embeddings are stored locally using **FAISS**
- User questions are matched against relevant document chunks
- A Bedrock-hosted LLM generates answers using retrieved context
- Results are displayed through a **Streamlit web interface**

## Architecture

- **AWS Bedrock** – Foundation models for embeddings and text generation  
- **Amazon Titan Embeddings** – Converts text chunks into vector embeddings  
- **FAISS** – Vector store for similarity search  
- **LangChain** – Manages retrieval and LLM orchestration  
- **Streamlit** – Web UI for user interaction  
- **boto3** – AWS SDK for Python  

## Features

- Query multiple PDF documents using natural language  
- Local vector index creation with FAISS  
- Retrieval-Augmented Generation (RAG) pipeline  
- Ability to switch between different Bedrock LLMs  
- Simple UI to update vectors and ask questions  

## Project Structure
```
.
├── requirements.txt
├── app.py          
└── README.md
```

## Models Used (AWS Bedrock)

### Embeddings Model
- `amazon.titan-embed-text-v1`

### Text Generation Models
- `meta.llama2-70b-chat-v1`
- `ai21.j2-mid-v1`

## Prerequisites

- AWS account with access to **AWS Bedrock**
- IAM role or user with the following permission:
  - `bedrock:InvokeModel`
- Python 3.9 or higher
- PDF files placed inside the `data/` directory

## Installation

Install the required dependencies:

```bash
pip install boto3 streamlit langchain faiss-cpu numpy
```

## Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

## Usage Steps

- Click "Vectors Update" to create or update the FAISS vector store
- Enter a question related to the uploaded PDF documents
- Choose the LLM to generate the response
- View the generated answer in the UI

## Notes

- FAISS index is stored locally for simplicity
- Chunk size and overlap can be tuned for better retrieval performance
- This project focuses on RAG concepts and AWS Bedrock integration, not production scaling
- Error handling and monitoring are intentionally minimal

## Future Improvements

- Move vector storage to a managed vector database
- Add metadata-based filtering
- Improve prompt templates
- Add authentication and access control
- Support large-scale document collections
