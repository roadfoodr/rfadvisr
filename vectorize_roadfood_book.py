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

# Create a mapping of state abbreviations to full state names
state_mapping = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
    'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
    'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
    'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
    'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri',
    'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
    'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio',
    'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont',
    'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming',
    'DC': 'District of Columbia', 'PR': 'Puerto Rico', 'VI': 'Virgin Islands'
}

# Read only specific columns from the CSV
columns_to_read = ['ID', 'Restaurant', 'URL', 'Address', 'City', 'State', 'Region', 
                   'phone', 'hours', 'Crossout', 'Honor Roll', 'Recommend', 
                   'long', 'lat', 'geohash', 'sig_item', 'content']

df = pd.read_csv('data/Roadfood_10th_supplemented.csv', usecols=columns_to_read)

# Filter out records where Crossout is 'y'
print(f"Total records before filtering: {len(df)}")
df = df[df['Crossout'] != 'y']
print(f"Records after removing crossed out entries: {len(df)}")

# Remove the Crossout column as it's no longer needed
df = df.drop(columns=['Crossout'])

# Add the full state name based on the abbreviation
df['State Name'] = df['State'].map(state_mapping)
print(f"Added 'State Name' field to the dataframe")

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
    print(f"Example State: {documents[0].metadata.get('State')} -> State Name: {documents[0].metadata.get('State Name')}")

# * Controlling Metadata Fields
# If you want to control which metadata fields are stored in ChromaDB,
# you can modify the documents before creating the vector store:

# Option 1: Keep only specific metadata fields
# metadata_fields_to_keep = ['Restaurant', 'Address', 'State', 'State Name', 'City']
# for doc in documents:
#     doc.metadata = {k: doc.metadata[k] for k in metadata_fields_to_keep if k in doc.metadata}

# * Post Processing Text

# IMPORTANT: Prepend the restaurant name and location to the page content
# - Helps with adding sources and searching titles
for doc in documents:
    # Retrieve the restaurant name, city and state from the document's metadata
    restaurant_name = doc.metadata.get('Restaurant', 'Unknown Restaurant')
    city = doc.metadata.get('City', 'Unknown City')
    state = doc.metadata.get('State', 'Unknown State')
    state_name = doc.metadata.get('State Name', 'Unknown State')
    sig_item = doc.metadata.get('sig_item', 'Unknown Item')
    
    # Prepend the restaurant name, city and state to the page content
    updated_content = f"Restaurant: {restaurant_name}\nLocation: {city}, {state_name}\nSignature Item: {sig_item}\n\n{doc.page_content}"
    
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
    state_name_to_search = retrieved_doc.metadata['State Name']
    
    print(f"Filtering by state abbreviation '{state_to_search}':")
    filtered_results = vectorstore.similarity_search(
        "good food",
        k=2,
        filter={"State": state_to_search}
    )
    for i, doc in enumerate(filtered_results):
        print(f"Result {i+1}: {doc.page_content.split('\\n')[0]}")
    
    print(f"\nFiltering by full state name '{state_name_to_search}':")
    filtered_results_by_name = vectorstore.similarity_search(
        "good food",
        k=2,
        filter={"State Name": state_name_to_search}
    )
    for i, doc in enumerate(filtered_results_by_name):
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

# * Example: Search for restaurants in a specific state by name
print("\nSearching for restaurants in California:")
california_results = vectorstore.similarity_search(
    query="best seafood", 
    k=3,
    filter={"State Name": "California"}
)

for i, doc in enumerate(california_results):
    restaurant = doc.metadata.get('Restaurant', 'Unknown')
    city = doc.metadata.get('City', 'Unknown')
    print(f"Result {i+1}: {restaurant} in {city}, California")


