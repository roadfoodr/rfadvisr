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

# Read only specific columns from the CSV
columns_to_read = ['ID', 'Restaurant', 'URL', 'Address', 'City', 'State', 'Region', 
                   'phone', 'hours', 'Crossout', 'Honor Roll', 'Recommend', 
                   'long', 'lat', 'geohash', 'content']

df = pd.read_csv('data/Roadfood_10th_reprocessed_final.csv', usecols=columns_to_read)

# Filter out records where Crossout is 'y'
print(f"Total records before filtering: {len(df)}")
df = df[df['Crossout'] != 'y']
print(f"Records after removing crossed out entries: {len(df)}")

# Remove the Crossout column as it's no longer needed
df = df.drop(columns=['Crossout'])

df.head()

# * Document Loaders
#   https://python.langchain.com/docs/integrations/document_loaders/pandas_dataframe 

loader = DataFrameLoader(df, page_content_column='content')

documents = loader.load()
# len(documents)

# documents[0]

# * Examining Metadata
# Print all metadata fields from the first document to see what's available
print("\nDefault metadata fields in documents:")
if documents:
    print(documents[0].metadata.keys())

# * Controlling Metadata Fields
# If you want to control which metadata fields are stored in ChromaDB,
# you can modify the documents before creating the vector store:

# Option 1: Keep only specific metadata fields
# metadata_fields_to_keep = ['Restaurant', 'Address', 'State', 'City']
# for doc in documents:
#     doc.metadata = {k: doc.metadata[k] for k in metadata_fields_to_keep if k in doc.metadata}


# documents[0].metadata
# documents[0].page_content

# pprint(documents[0].page_content)

# for doc in documents:
#     print(len(doc.page_content))

# * Post Processing Text

# IMPORTANT: Prepend the restaurant name and location to the page content
# - Helps with adding sources and searching titles
for doc in documents:
    # Retrieve the restaurant name, city and state from the document's metadata
    restaurant_name = doc.metadata.get('Restaurant', 'Unknown Restaurant')
    city = doc.metadata.get('City', 'Unknown City')
    state = doc.metadata.get('State', 'Unknown State')
    
    # Prepend the restaurant name, city and state to the page content
    updated_content = f"Restaurant: {restaurant_name}\nLocation: {city}, {state}\n\n{doc.page_content}"
    
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

# * Examining Stored Metadata in ChromaDB
# Retrieve a document to see what metadata is stored
print("\nMetadata stored in ChromaDB:")
retrieved_doc = vectorstore.similarity_search("", k=1)[0]
print(retrieved_doc.metadata)

# * Filtering by Metadata in Searches
# You can filter searches by metadata
print("\nFiltered search by metadata:")
# Example: Search for restaurants in a specific state (if 'State' is in your metadata)
if 'State' in retrieved_doc.metadata:
    state_to_search = retrieved_doc.metadata['State']
    filtered_results = vectorstore.similarity_search(
        "good food",
        k=2,
        filter={"State": state_to_search}
    )
    for i, doc in enumerate(filtered_results):
        print(f"Result {i+1}: {doc.page_content.split('\\n')[0]}")

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


