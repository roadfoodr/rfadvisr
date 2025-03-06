import streamlit as st
import os
import yaml
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import datetime

# Constants
EDITION = '10th'

# OPENAI API SETUP
os.environ['OPENAI_API_KEY'] = yaml.safe_load(open('credentials.yml'))['openai']
MODEL_EMBEDDING = 'text-embedding-ada-002'

# Initialize embedding function
embedding_function = OpenAIEmbeddings(
    model=MODEL_EMBEDDING,
)

# Load the existing Chroma database
persist_directory = f"./data/chroma_rf{EDITION}"
vectorstore = Chroma(
    embedding_function=embedding_function,
    persist_directory=persist_directory
)

def search_restaurants(query, num_results=4, save_results=False):
    """Search for restaurants based on the query"""
    if not query.strip():
        return "Please enter a search query."
    
    try:
        results = vectorstore.similarity_search(
            query=query,
            k=num_results
        )
        
        output = []
        for i, doc in enumerate(results):
            output.append(f"## Result {i+1}:\n\n{doc.page_content}\n\n---")
        
        # Save results to file if requested
        if save_results:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/search_results_{timestamp}.txt"
            os.makedirs(os.path.dirname(filename), exist_ok=True)  # Ensure directory exists
            with open(filename, "w") as f:
                f.write(f"Search query: {query}\n\n")
                for i, doc in enumerate(results):
                    f.write(f"Result {i+1}:\n")
                    f.write(f"{doc.page_content}\n")
                    f.write("-" * 50 + "\n\n")
            output.append(f"\n\n*Results saved to {filename}*")
        
        return "\n".join(output)
    except Exception as e:
        return f"Error during search: {str(e)}"

# Set up Streamlit page configuration
st.set_page_config(
    page_title=f"Roadfood {EDITION} Edition Search",
    page_icon="🍽️",
    layout="wide"
)

# Create Streamlit interface
st.title(f"Roadfood {EDITION} Edition Restaurant Search")
st.markdown("Search for restaurants based on your preferences, cuisine, location, etc.")

# Sidebar for inputs
with st.sidebar:
    st.header("Search Options")
    
    # Example queries
    st.subheader("Example Searches")
    example_queries = [
        "Where is the best BBQ?",
        "Unique seafood restaurants on the East Coast",
        "Famous diners in New Jersey",
        "Where can I find good pie?",
        "Historic restaurants with great burgers"
    ]
    
    selected_example = st.selectbox(
        "Try an example search:",
        [""] + example_queries,
        index=0
    )
    
    # Search parameters
    query_input = st.text_area(
        "What are you looking for?",
        value=selected_example,
        placeholder="e.g., 'best BBQ in Texas' or 'unique seafood restaurants'"
    )
    
    num_results = st.slider(
        "Number of results",
        min_value=1,
        max_value=10,
        value=4,
        step=1
    )
    
    save_checkbox = st.checkbox("Save results to file", value=False)
    
    search_button = st.button("Search")

# Main content area
if search_button and query_input:
    with st.spinner("Searching for restaurants..."):
        results = search_restaurants(query_input, num_results, save_checkbox)
        st.markdown(results)
elif search_button:
    st.warning("Please enter a search query.")

# Display some information about the app
with st.expander("About this app"):
    st.markdown(f"""
    This app searches through the Roadfood database to find restaurants matching your criteria.
    It uses vector embeddings to find the most relevant matches to your query.
    
    The database contains restaurants from the Roadfood guide {EDITION} edition.
    """)

# Run the app
# Note: No need for if __name__ == "__main__" in Streamlit
# Streamlit apps are run with the command: streamlit run roadfood_search_app.py