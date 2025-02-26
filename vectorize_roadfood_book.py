# LIBRARIES 
from langchain_community.document_loaders import DataFrameLoader
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma

from langchain_openai import OpenAIEmbeddings

import pandas as pd
import yaml
import os

from pprint import pprint

# OPENAI API SETUP
os.environ['OPENAI_API_KEY'] = yaml.safe_load(open('credentials.yml'))['openai']
MODEL_EMBEDDING = 'text-embedding-ada-002'

# 1.0 DATA PREPARATION ----

df = pd.read_csv('data/Roadfood_10th_edition_processed.csv')
df.head()

# * Document Loaders
#   https://python.langchain.com/docs/integrations/document_loaders/pandas_dataframe 

loader = DataFrameLoader(df, page_content_column='content')

documents = loader.load()
# len(documents)

# documents[0]

# documents[0].metadata
# documents[0].page_content

# pprint(documents[0].page_content)

# for doc in documents:
#     print(len(doc.page_content))

# * Post Processing Text

# IMPORTANT: Prepend the title and author to the page content
# - Helps with adding sources and searching titles
for doc in documents:
    # Retrieve the title and author from the document's metadata
    restaurant_name = doc.metadata.get('title', 'Unknown Restaurant')
    address = doc.metadata.get('address', 'Unknown Address')
    
    # Prepend the title and author to the page content
    updated_content = f"Restaurant: {restaurant_name}\nAddress: {address}\n\n{doc.page_content}"
    
    # Update the document's page content
    doc.page_content = updated_content

# pprint(documents[0].page_content)

# * Text Embeddings

# OpenAI Embeddings
# - See Account Limits for models: https://platform.openai.com/account/limits
# - See billing to add to your credit balance: https://platform.openai.com/account/billing/overview

embedding_function = OpenAIEmbeddings(
    model=MODEL_EMBEDDING,
)

# Open Source Alternative:
# Requires Torch and SentenceTransformer packages:

# from sentence_transformers import SentenceTransformerEmbeddings
# embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


# * Langchain Vector Store: Chroma DB
# https://python.langchain.com/docs/integrations/vectorstores/chroma

# Check if the vector database already exists
persist_directory = "./data/chroma_rf10th"  # Create a subdirectory for this specific database

# Check if the directory exists and contains Chroma files
if os.path.exists(persist_directory) and os.path.exists(os.path.join(persist_directory, "chroma.sqlite3")):
    print("Loading existing Chroma database...")
    # Load the existing database
    vectorstore = Chroma(
        embedding_function=embedding_function,
        persist_directory=persist_directory
    )
else:
    print("Creating new Chroma database...")
    # Create a new database
    vectorstore = Chroma.from_documents(
        documents,
        embedding=embedding_function,
        persist_directory=persist_directory
    )

# vectorstore

# * Similarity Search: The whole reason we did this

result = vectorstore.similarity_search(
    query="where is the strangest hotdog?", 
    k = 4
)

# result

# pprint(result[0].page_content)

# Print the first line of each result
print("\nFirst line of each result:")
for i, doc in enumerate(result):
    first_line = doc.page_content.split('\n')[0]
    print(f"Result {i+1}: {first_line}")


