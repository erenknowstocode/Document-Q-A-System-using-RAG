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
- **Claude 3 Haiku** – Generates responses using retrieved context  
- **FAISS** – Vector store for similarity search  
- **LangChain** – Manages retrieval and LLM orchestration  
- **Streamlit** – Web UI for user interaction  
- **boto3** – AWS SDK for Python  

---

## Features

- Query multiple PDF documents using natural language  
- Local vector index creation with FAISS  
- Retrieval-Augmented Generation (RAG) pipeline  
- Uses chat-based LLMs via AWS Bedrock  
- Simple UI to update vectors and ask questions  

## Project Structure

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
├── data/ 
├── faiss_index/ 
├── app.py 
├── requirements.txt
└── README.md
```

## Models Used (AWS Bedrock)

### Embeddings Model
- `amazon.titan-embed-text-v1`

### Text Generation Model
- `anthropic.claude-3-haiku-20240307-v1:0`

> These models are **ACTIVE and ON_DEMAND** in the `us-east-1` region.

## Prerequisites

- AWS account with access to **AWS Bedrock**
- Bedrock model access enabled for:
  - Amazon Titan Embeddings
  - Claude 3 Haiku
- IAM role or user with permission:
  - `bedrock:InvokeModel`
- Python 3.9 or higher
- PDF files placed inside the `data/` directory

Configure AWS credentials:

```bash
aws configure
```
## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

## Usage Steps

- Place PDF files inside the data/ directory
- Click Create / Update Vectors to build the FAISS index
- Enter a question related to the uploaded documents
- Click Generate Answer to get a response from the LLM

## Notes

- FAISS index is stored locally for simplicity
- Pickle-based FAISS deserialization is explicitly enabled for locally generated indexes
- Chunk size and overlap can be tuned for better retrieval performance
- This project focuses on RAG concepts and AWS Bedrock integration, not production scaling
- Error handling and monitoring are intentionally minimal

## Future Improvements

- Move vector storage to a managed vector database
- Add metadata-based filtering
- Improve prompt templates
- Add authentication and access control
- Support large-scale document collections
- Add model fallback logic for multiple Bedrock LLMs
