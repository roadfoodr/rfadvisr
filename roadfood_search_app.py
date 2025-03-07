import streamlit as st

import os
import yaml
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import datetime
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document

# Constants
EDITION = '10th'

# Set up Streamlit page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title=f"Roadfood {EDITION} Edition Search",
    page_icon="🍽️",
    layout="wide"
)

# OPENAI API SETUP
os.environ['OPENAI_API_KEY'] = yaml.safe_load(open('credentials.yml'))['openai']
MODEL_EMBEDDING = 'text-embedding-ada-002'
LLM_MODEL = 'gpt-3.5-turbo'

# Initialize embedding function
@st.cache_resource
def get_embedding_function():
    return OpenAIEmbeddings(
        model=MODEL_EMBEDDING,
    )

# Initialize LLM
@st.cache_resource
def get_llm():
    return ChatOpenAI(model=LLM_MODEL, temperature=0.7)

# Load the existing Chroma database
@st.cache_resource
def get_vectorstore():
    embedding_function = get_embedding_function()
    persist_directory = f"./data/chroma_rf{EDITION}"
    return Chroma(
        embedding_function=embedding_function,
        persist_directory=persist_directory
    )

# Get cached resources
embedding_function = get_embedding_function()
llm = get_llm()
vectorstore = get_vectorstore()

@st.cache_data
def generate_summary(query, full_content):
    """Generate a summary article from the search results using an LLM"""
    # Create the prompt for the LLM
    prompt_template = ChatPromptTemplate.from_template("""
    You are a food writer creating a summary article based on search results for "{query}".
    
    Here are the details of restaurants found in the search:
    {full_content}
    
    Create a summary article with:
    1. A catchy title related to the search query
    2. A bullet point list of each restaurant name and location
    3. A consolidated article (no more than 3 paragraphs) that summarizes the key points about these restaurants,
       highlighting what makes them special, their signature dishes, and any other interesting information.
    
    Format your response with markdown.
    """)
    
    # Generate the summary
    chain = prompt_template | llm
    response = chain.invoke({"query": query, "full_content": full_content})
    
    return response.content

@st.cache_data
def perform_search(query, num_results):
    """Perform the vector search and return results"""
    if not query.strip():
        return []
    
    try:
        results = vectorstore.similarity_search(
            query=query,
            k=num_results
        )
        return results
    except Exception as e:
        st.error(f"Error during search: {str(e)}")
        return []

def save_results_to_file(query, content):
    """Save results to a file and return the filename"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/search_results_{timestamp}.txt"
    os.makedirs(os.path.dirname(filename), exist_ok=True)  # Ensure directory exists
    
    with open(filename, "w") as f:
        f.write(f"Search query: {query}\n\n")
        f.write(content)
    
    return filename

# Create Streamlit interface
st.title(f"Roadfood {EDITION} Edition Restaurant Search")
st.markdown("Search for restaurants based on your preferences, cuisine, location, etc.")

# Example queries outside the form (these don't trigger re-renders)
example_queries = [
    "Where is the best BBQ?",
    "Unique seafood restaurants on the East Coast",
    "Famous diners in New Jersey",
    "Where can I find good pie?",
    "Historic restaurants with great burgers"
]

# Store the selected example in session state so we can use it in the form
if 'selected_example' not in st.session_state:
    st.session_state.selected_example = ""

def update_example():
    if st.session_state.example_selector:
        st.session_state.selected_example = st.session_state.example_selector

# Example selector outside the form
st.sidebar.header("Example Searches")
st.sidebar.selectbox(
    "Try an example search:",
    [""] + example_queries,
    key="example_selector",
    on_change=update_example
)

# Create a form for all search inputs
with st.sidebar:
    with st.form(key="search_form"):
        st.header("Search Options")
        
        # Search parameters
        query_input = st.text_area(
            "What are you looking for?",
            value=st.session_state.selected_example,
            placeholder="e.g., 'best BBQ in Texas' or 'unique seafood restaurants'"
        )
        
        num_results = st.slider(
            "Number of results",
            min_value=1,
            max_value=10,
            value=4,
            step=1
        )
        
        generate_article_checkbox = st.checkbox("Generate summary article", value=True)
        save_checkbox = st.checkbox("Save results to file", value=False)
        
        # Form submit button
        search_submitted = st.form_submit_button("Search")

# Main content area - only process when form is submitted
if search_submitted:
    if not query_input.strip():
        st.warning("Please enter a search query.")
    else:
        with st.spinner("Searching for restaurants..."):
            # Perform the search
            search_results = perform_search(query_input, num_results)
            
            if search_results:
                # Process and display results based on user preferences
                if generate_article_checkbox:
                    # Extract full content for summarization
                    full_content = "\n\n".join([doc.page_content for doc in search_results])
                    
                    # Generate summary
                    summary = generate_summary(query_input, full_content)
                    
                    # Display summary
                    display_content = summary
                    
                    # Save to file if requested
                    if save_checkbox:
                        filename = save_results_to_file(query_input, summary)
                        display_content += f"\n\n*Results saved to {filename}*"
                    
                    st.markdown(display_content)
                else:
                    # Display detailed results
                    output = []
                    for i, doc in enumerate(search_results):
                        output.append(f"## Result {i+1}:\n\n{doc.page_content}\n\n---")
                    
                    display_content = "\n".join(output)
                    
                    # Save to file if requested
                    if save_checkbox:
                        detailed_content = f"Search query: {query_input}\n\n"
                        for i, doc in enumerate(search_results):
                            detailed_content += f"Result {i+1}:\n"
                            detailed_content += f"{doc.page_content}\n"
                            detailed_content += "-" * 50 + "\n\n"
                        
                        filename = save_results_to_file(query_input, detailed_content)
                        display_content += f"\n\n*Results saved to {filename}*"
                    
                    st.markdown(display_content)

# Display some information about the app
with st.expander("About this app"):
    st.markdown(f"""
    This app searches through the Roadfood database to find restaurants matching your criteria.
    It uses vector embeddings to find the most relevant matches to your query.
    
    The database contains restaurants from the Roadfood guide {EDITION} edition.
    
    When generating a summary article, the app uses OpenAI's language model to create a concise
    overview of the search results, including restaurant names, locations, and key highlights.
    """)

# Run the app
# Note: No need for if __name__ == "__main__" in Streamlit
# Streamlit apps are run with the command: streamlit run roadfood_search_app.py