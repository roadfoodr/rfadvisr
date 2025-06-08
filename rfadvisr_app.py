import streamlit as st

import os
import yaml
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import datetime
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
import re
from pathlib import Path
from typing import TypedDict, Dict, Optional, List
from langgraph.graph import StateGraph, END
import json
from app.supabase_client import SupabaseManager
from app.supabase_helpers import store_search_scores

# Import filter tools
from app.filter_tools import filter_tools

# ---> ADDED LANGSMITH IMPORTS <---
from langsmith import Client
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
# ---> END LANGSMITH IMPORTS <---

# --- LangGraph State Definition ---
class FilterGenerationState(TypedDict):
    """Represents the state of our filter generation graph."""
    query: str                   # Original user query
    # available_metadata: Dict[str, str] # Schema description (Removed - Handled by tool descriptions)
    extracted_filters: List[Dict] # List of ChromaDB filter conditions extracted by tools
    error_message: Optional[str] # To capture errors during generation

# Constants
EDITION = '10th'


# Set up Streamlit page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title=f"Roadfood Advisor -- {EDITION} Edition",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- API KEY and LANGSMITH SETUP ---
# Prioritize environment variables (common in deployment like Modal)
# then fall back to credentials.yml (for local development)

openai_api_key = os.environ.get("OPENAI_API_KEY")
langsmith_api_key = os.environ.get("LANGSMITH_API_KEY")
langsmith_endpoint = os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com") # Default endpoint
langsmith_project = os.environ.get("LANGSMITH_PROJECT", f"rf_search_app_{EDITION}") # Default project

# If keys are not found in environment, try loading from credentials.yml
if not openai_api_key or not langsmith_api_key:
    print("--- API keys not found in environment, trying credentials.yml ---")
    try:
        credentials_path = 'credentials.yml'
        if os.path.exists(credentials_path):
            credentials = yaml.safe_load(open(credentials_path))
            if not openai_api_key:
                openai_api_key = credentials.get('openai')
            if not langsmith_api_key:
                # Only set LangSmith details from file if the key was found there
                if 'langsmith' in credentials:
                    langsmith_api_key = credentials['langsmith']
                    # Allow overriding endpoint/project from file only if not set by env
                    if "LANGSMITH_ENDPOINT" not in os.environ:
                        langsmith_endpoint = credentials.get('langsmith_endpoint', langsmith_endpoint)
                    if "LANGSMITH_PROJECT" not in os.environ:
                        langsmith_project = credentials.get('langsmith_project', langsmith_project)
                else:
                     print("--- LangSmith API key not found in credentials.yml ---")
        else:
            print(f"--- {credentials_path} not found. ---")
    except Exception as e:
        st.warning(f"Error loading credentials.yml: {str(e)}") # Use warning, don't stop

# Set OpenAI key in environment if found
if openai_api_key:
    os.environ['OPENAI_API_KEY'] = openai_api_key
else:
    # Critical error if no OpenAI key is available from either source
    st.error("OpenAI API key not found in environment variables or credentials.yml. Application cannot proceed.")
    st.stop() # Stop execution if OpenAI key is missing

# Configure LangSmith tracing if the key is available
if langsmith_api_key:
    os.environ['LANGSMITH_TRACING'] = 'true'
    os.environ['LANGSMITH_API_KEY'] = langsmith_api_key
    os.environ['LANGSMITH_ENDPOINT'] = langsmith_endpoint
    os.environ['LANGSMITH_PROJECT'] = langsmith_project
    print(f"--- LangSmith tracing enabled (Project: {langsmith_project}) ---")
else:
    # Inform user if LangSmith is disabled
    print("--- LangSmith API key not found in environment or credentials.yml, tracing disabled ---")
    # Ensure tracing variable is not set if key is missing
    if 'LANGSMITH_TRACING' in os.environ:
        del os.environ['LANGSMITH_TRACING']

# ---> INITIALIZE LANGSMITH CLIENT <---
# Initialize ONLY if tracing is intended (key was found)
if os.environ.get('LANGSMITH_TRACING') == 'true':
    try:
        ls_client = Client() # Client uses env vars set above
        print("--- LangSmith client initialized ---")
    except Exception as e:
        ls_client = None
        st.warning(f"Could not initialize LangSmith client: {e}. Feedback logging disabled.")
        print(f"--- LangSmith client initialization failed: {e} ---")
else:
    ls_client = None # Explicitly set to None if tracing is disabled
    print("--- LangSmith client not initialized as tracing is disabled ---")
# ---> END LANGSMITH CLIENT INIT <---

# ---> DEFINE HELPER/PROCESSING FUNCTIONS (Moved Here - Top Level) <--- 

# ---> HELPER TO DISPLAY DETAILED RESULTS (Moved Here) <---   
def display_detailed_results(search_results, query_input, save_checkbox):
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
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"search_results_{timestamp}.txt"
        st.download_button(
            label="Download Results",
            data=detailed_content,
            file_name=filename,
            mime="text/plain"
        )
    
    st.markdown(display_content)
# ---> END HELPER FUNCTION <--- 

# ---> DECORATED FUNCTION FOR SEARCH PROCESSING (Moved Here) <---    
@traceable(name="RF_search_query", run_type="chain") # <-- Re-enable this
def handle_search_request(query_input, num_results, pre_filter_checkbox, generate_article_checkbox, save_checkbox):
    """Handles the main logic for search, filtering, summarization. Returns results and metadata.""" # <-- Updated docstring
    results_dict = {
        "run_id": None,
        "query": query_input,
        "summary_result": None,
        "search_results": None,
        "generated_filter": None,
        "generate_article_mode": generate_article_checkbox,
        "save_mode": save_checkbox,
        "error_message": None,
        "info_message": None
    }
    generated_filter = {}
    try:
        # Capture run_id inside the traceable function using get_current_run_tree
        run_tree = get_current_run_tree() 
        if run_tree: 
            current_run_id = run_tree.id 
            results_dict["run_id"] = str(current_run_id)
            print(f"--- Captured LangSmith run_id (inside traceable): {results_dict['run_id']} ---") 
        else: 
            print("--- Failed to get run_tree inside traceable function ---") 
            # Allow processing to continue, but feedback won't work

        # ---> GUARDRAIL CHECK <-----
        if is_query_in_scope(query_input):
            # Query is IN SCOPE, proceed with processing
            with st.spinner("Analyzing query and searching for restaurants..."): 
                if pre_filter_checkbox:
                    generated_filter = generate_search_filter(query_input)
                else:
                    print("--- Skipping pre-filtering step as requested ---")

                search_docs = perform_search(
                    query_input,
                    num_results,
                    filter_dict=generated_filter
                )
                results_dict["search_results"] = search_docs # Store raw results

                if search_docs:
                    # Store search results and scores in Supabase
                    search_id = store_search_scores(
                        search_docs=search_docs,
                        query=query_input,
                        langsmith_run_id=results_dict["run_id"],
                        filter_applied=generated_filter,
                        search_type='detailed' if not generate_article_checkbox else 'summary'
                    )
                    if search_id:
                        print(f"--- Successfully stored search results with ID: {search_id} ---")
                    else:
                        print("--- Failed to store search results in Supabase ---")

                    if generate_article_checkbox:
                        full_content = "\n\n".join([doc.page_content for doc in search_docs])
                        with st.spinner("Generating summary..."): 
                            summary_result = generate_summary(query_input, full_content, search_docs)
                        results_dict["summary_result"] = summary_result # Store summary
                else:
                    results_dict["info_message"] = "No results found matching your query and filters."

        else:
            # Query is OUT OF SCOPE
            results_dict["error_message"] = "Sorry, I can only answer questions about restaurants and food based on the Roadfood guide. Please try a different query."
            # --- Remove st.error call & sidebar display --- 

    except Exception as e:
        print(f"--- Error within handle_search_request: {e} ---")
        results_dict["error_message"] = f"An error occurred during processing: {e}"
        # Error is automatically logged by the @traceable decorator, but capture for display
    
    # Always add the generated_filter to the results before returning
    results_dict["generated_filter"] = generated_filter
    return results_dict # <-- Return the dictionary
# ---> END DECORATED FUNCTION <---        

# ---> END HELPER/PROCESSING FUNCTIONS <--- 

MODEL_EMBEDDING = 'text-embedding-ada-002'
LLM_MODEL = 'gpt-3.5-turbo'
# LLM_MODEL = 'gpt-4o-mini'

# Initialize embedding function
@st.cache_resource
def get_embedding_function():
    return OpenAIEmbeddings(
        model=MODEL_EMBEDDING,
    )

# Initialize LLM
@st.cache_resource
def get_llm():
    """Get the base LLM model"""
    # Always return the base model
    return ChatOpenAI(model=LLM_MODEL, temperature=0.7)

# Determine the base directory for data files
def get_base_dir():
    """Get the base directory for data files based on environment"""
    # Check if running in Modal
    if os.path.exists("/root/data"):
        return Path("/root")
    else:
        return Path(".")

# Load the existing Chroma database
@st.cache_resource
def get_vectorstore():
    embedding_function = get_embedding_function()
    base_dir = get_base_dir()
    persist_directory = base_dir / f"data/chroma_rf{EDITION}"
    return Chroma(
        embedding_function=embedding_function,
        persist_directory=str(persist_directory)
    )

# Get cached resources
embedding_function = get_embedding_function()
# llm = get_llm() # Remove this global instance, it's not used
vectorstore = get_vectorstore()

def post_process_summary(summary_text, search_results):
    """
    Post-process the summary to bold all restaurant names and add hyperlinks to their first occurrence.
    
    Args:
        summary_text: The text of the summary generated by the LLM
        search_results: The search results containing restaurant metadata
    
    Returns:
        Processed summary text with formatting applied
    """
    # Extract restaurant names and URLs from search results
    restaurants = []
    for doc in search_results:
        # Extract metadata if available
        metadata = getattr(doc, 'metadata', {})
        restaurant_name = metadata.get('Restaurant', '')
        url = metadata.get('URL', '')
        
        # Only process if we have a restaurant name
        if restaurant_name:
            restaurants.append({
                'name': restaurant_name,
                'url': url
            })
    
    # Track first occurrences
    first_occurrence = {r['name']: True for r in restaurants}
    
    # Sort restaurant names by length (descending) to avoid partial replacements
    # (e.g., replace "Joe's Diner" before "Joe")
    restaurants.sort(key=lambda x: len(x['name']), reverse=True)
    
    # Process each restaurant name
    for restaurant in restaurants:
        name = restaurant['name']
        url = restaurant['url']
        
        # Skip empty names
        if not name.strip():
            continue
        
        # Create regex pattern to match whole words/phrases only
        # This avoids replacing parts of other words
        # Use word boundaries \b for single-word restaurant names
        if len(name.split()) == 1:
            # For single words, use word boundaries
            pattern = r'(?<!\*\*)\b' + re.escape(name) + r'\b(?!\*\*)'
        else:
            # For phrases, use the existing pattern
            pattern = r'(?<!\*\*)' + re.escape(name) + r'(?!\*\*)'
        
        # Check if the name appears in the text
        if re.search(pattern, summary_text, re.IGNORECASE):
            # For the first occurrence, add hyperlink (if URL exists) and bold
            if first_occurrence[name]:
                if url:
                    # Add https:// prefix if missing
                    if not url.startswith('http'):
                        url = f"https://{url}"
                    # Replace with hyperlinked and bolded version
                    replacement = f"[**{name}**]({url})"
                else:
                    # Just bold if no URL
                    replacement = f"**{name}**"
                
                # Replace only the first occurrence (case insensitive)
                summary_text = re.sub(pattern, replacement, summary_text, count=1, flags=re.IGNORECASE)
                first_occurrence[name] = False
                
                # Bold all subsequent occurrences
                # Use the same pattern for consistency
                summary_text = re.sub(pattern, f"**{name}**", summary_text, flags=re.IGNORECASE)
            
    return summary_text

def standardize_summary_headline(summary_text):
    """
    Standardize headline formatting in summaries to ensure consistent heading levels.
    
    This function:
    1. Identifies the title (content before the bullet list)
    2. Converts the title to an h3 header, removing any formatting or prefixes
    3. Leaves the rest of the content unchanged
    
    Args:
        summary_text: The text of the summary to process
        
    Returns:
        Processed summary text with standardized headings
    """
    # Split the text into lines for processing
    lines = summary_text.split('\n')
    
    # Skip any empty lines at the beginning
    start_index = 0
    while start_index < len(lines) and not lines[start_index].strip():
        start_index += 1
    
    # If we've reached the end, return the original text
    if start_index >= len(lines):
        return summary_text
    
    # Find the first bullet point or empty line after non-empty content
    bullet_index = -1
    for i in range(start_index + 1, len(lines)):
        line = lines[i].strip()
        # If this is a bullet point, we've found our separator
        if line.startswith('*') or line.startswith('-'):
            bullet_index = i
            break
        # If this is an empty line after we've seen content, it might be our separator
        if not line and i > start_index:
            # Check if the next line exists and is not empty (to avoid treating paragraph breaks as separators)
            if i + 1 < len(lines) and lines[i + 1].strip():
                bullet_index = i
                break
    
    # If we didn't find a bullet point or suitable empty line, just use the first line as the title
    if bullet_index == -1:
        # Extract and clean the first non-empty line as the title
        title = lines[start_index].strip()
        # Remove any markdown headers
        title = re.sub(r'^#+\s+', '', title)
        # Remove bold markers
        title = re.sub(r'\*\*', '', title)
        # Remove common title prefixes
        title = re.sub(r'^(?:Title:|Catchy Title:|Headline:)\s*', '', title, flags=re.IGNORECASE)
        
        # Replace the first line with the cleaned h3 title
        if title:  # Only if we have actual content
            lines[start_index] = f"### {title}"
        
        # Convert any other h1 or h2 headers to h3
        for i in range(start_index + 1, len(lines)):
            if lines[i].strip().startswith('# ') or lines[i].strip().startswith('## '):
                lines[i] = re.sub(r'^#+\s+', '### ', lines[i])
        
        return '\n'.join(lines)
    
    # We found a separator, so extract everything before it as the title
    title_lines = [line for line in lines[start_index:bullet_index] if line.strip()]
    
    # If we have title lines, process them
    if title_lines:
        # Join all title lines into a single string
        title_text = ' '.join([line.strip() for line in title_lines])
        
        # Clean up the title
        # Remove any markdown headers
        title_text = re.sub(r'^#+\s+', '', title_text)
        # Remove bold markers
        title_text = re.sub(r'\*\*', '', title_text)
        # Remove common title prefixes
        title_text = re.sub(r'^(?:Title:|Catchy Title:|Headline:)\s*', '', title_text, flags=re.IGNORECASE)
        
        # Create the new title as an h3 header
        new_title = f"### {title_text.strip()}"
        
        # Build the new content with the standardized title
        result = [new_title, '']  # Title followed by blank line
        result.extend(lines[bullet_index:])  # Add the rest unchanged
        
        return '\n'.join(result)
    
    # If we didn't find any title content, return the original text
    return summary_text

def load_prompt_template(prompt_type="advanced"):
    """Load the prompt template from file
    
    Args:
        prompt_type: Type of prompt to load ("basic", "advanced", "tool_calling", or "guardrail")
    
    Returns:
        The prompt template text or None if there was an error
    """
    base_dir = get_base_dir()
    # Adjust filename based on type
    if prompt_type == "tool_calling":
        filename = base_dir / f"prompts/tool_calling_prompt.txt"
    elif prompt_type == "guardrail": # Added guardrail type
        filename = base_dir / f"prompts/guardrail_prompt.txt"
    else:
        # Default to summary prompts if not tool_calling or guardrail
        filename = base_dir / f"prompts/{prompt_type}_summary_prompt.txt"
    
    try:
        with open(filename, "r") as f:
            return f.read()
    except Exception as e:
        st.error(f"Error reading prompt template {filename}: {str(e)}")
        return None

def reload_prompt_template():
    """Clear the cache to reload the prompt template"""
    # Clear the cache to force reload of prompt
    st.cache_data.clear()
    st.success("Prompt templates reloaded!")

def generate_summary(query, full_content, search_results):
    """Generate a summary article from the search results using the advanced prompt and base LLM
    
    Args:
        query: The search query
        full_content: The full content of the search results
        search_results: The search results objects
        
    Returns:
        The generated summary
    """
    # Read the ADVANCED prompt template from file
    prompt_text = load_prompt_template("advanced")
    
    # Debug statement to show which prompt is being used
    if prompt_text:
        print("--- Using ADVANCED prompt template from file ---")
    else:
        print("--- Using FALLBACK prompt template ---")
    
    # Fallback to hardcoded prompt if file can't be read
    if not prompt_text:
        # Simplified fallback prompt if needed (though ideally the file load works)
        prompt_text = """
        You are a food writer creating a detailed summary article based on search results for "{query}".
        
        Here are the details of restaurants found in the search:
        {full_content}
                
        Format your response with markdown, including a title and bullet points.
        """
    
    # Create the prompt for the LLM
    prompt_template = ChatPromptTemplate.from_template(prompt_text)
    
    # Get the base model
    model = get_llm() 
    
    # Generate the summary
    # ---> ADDED CONFIG TO NAME THE CHAIN RUN <---
    chain = (prompt_template | model).with_config({"run_name": "Summary Generation"})
    # ---> END CONFIG <---
    response = chain.invoke({"query": query, "full_content": full_content})
    
    # Post-process the summary to format restaurant names
    processed_summary = post_process_summary(response.content, search_results)
    
    return processed_summary

# --- Filter Generation Graph Nodes (Stubs) ---

# analyze_query_node removed - replaced by tool calling approach

# --- Tool Calling Node ---
def tool_calling_node(state: FilterGenerationState) -> Dict:
    """Uses an LLM to decide which filter tools to call based on the query.

    Args:
        state: The current graph state.

    Returns:
        A dictionary potentially containing the updated 'extracted_filters' list.
    """
    query = state["query"]
    print(f"--- TOOL CALLING NODE for query: '{query}' ---")

    # Get LLM and bind tools
    llm = get_llm()
    llm_with_tools = llm.bind_tools(filter_tools)

    # Load the system prompt from file
    system_prompt_text = load_prompt_template("tool_calling")
    if not system_prompt_text:
        # Fallback or error handling if prompt file fails to load
        st.error("Failed to load tool calling prompt! Using default behavior.")
        # Decide how to handle this - maybe return an error state?
        # For now, let's just proceed without a proper prompt which might fail.
        system_prompt_text = "You are a helpful assistant." # Basic fallback

    # Create ChatPromptTemplate using the loaded system prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_text),
        ("human", "{query}")
    ])

    # Chain and invoke
    chain = prompt | llm_with_tools
    try:
        ai_msg = chain.invoke({"query": query})
    except Exception as e:
        print(f"  Error invoking LLM with tools: {e}")
        return {"error_message": f"LLM invocation failed: {e}"}

    extracted_conditions = []
    if not ai_msg.tool_calls:
        print("  > LLM decided no tools were needed.")
    else:
        print(f"  > LLM decided to call tools: {[tc['name'] for tc in ai_msg.tool_calls]}")
        # The `bind_tools` method prepares the LLM call, but we need to explicitly run the tools
        # based on the ai_msg response in this LangGraph setup.

        # Create a mapping of tool names to tool objects for easy lookup
        available_tools_map = {tool.name: tool for tool in filter_tools}

        for tool_call in ai_msg.tool_calls:
            tool_name = tool_call['name']
            print(f"    - Attempting to execute tool: {tool_name}")
            tool_to_run = available_tools_map.get(tool_name)

            if not tool_to_run:
                print(f"    - Error: Tool '{tool_name}' requested by LLM but not found in available tools.")
                continue # Skip to the next tool call

            tool_args = tool_call.get('args')
            print(f"    - Tool Call Args: {tool_args}")

            try:
                # Explicitly run the tool with the provided arguments
                print(f"    - Running {tool_name}.run({tool_args})...")
                result = tool_to_run.run(tool_args)
                print(f"    - Tool {tool_name} executed. Result: {result}")

                # Append the result if it's not None
                if result is not None:
                    extracted_conditions.append(result)
                else:
                    # Log if the tool function genuinely returned None after execution
                    print(f"    - Tool '{tool_name}' executed but returned None.")

            except Exception as e:
                print(f"    - Error executing tool '{tool_name}' with args {tool_args}: {e}")
                # Optionally, update the graph state with an error message here
                # return {"error_message": f"Failed to execute tool {tool_name}: {e}"}

    print(f"  > Raw extracted conditions: {extracted_conditions}") # Log raw conditions

    # Deduplicate the extracted conditions
    unique_conditions = []
    seen_conditions = set()
    for condition in extracted_conditions:
        # Convert dict to canonical JSON string for comparison
        condition_str = json.dumps(condition, sort_keys=True)
        if condition_str not in seen_conditions:
            unique_conditions.append(condition)
            seen_conditions.add(condition_str)

    print(f"  > Deduplicated conditions: {unique_conditions}") # Log deduplicated conditions

    return {"extracted_filters": unique_conditions} # Return unique conditions

def format_filter_node(state: FilterGenerationState) -> Dict:
    """
    Formats the extracted list of conditions into the final ChromaDB 'where' clause.
    Currently assumes OR logic if multiple conditions exist.
    (Later: Could handle more complex logic like $and/$or or validation based on analysis)
    """
    print(f"--- FORMATTING FILTERS ---")
    conditions = state.get('extracted_filters', []) # Expecting a List[Dict]
    final_filter = {}

    if not conditions:
        print("  > No conditions extracted by tools.")
        final_filter = {} # Return empty dict if no filters
    elif len(conditions) == 1:
        final_filter = conditions[0] # Use the single condition directly
        print(f"  > Single condition: {final_filter}")
    else:
        # Change: Combine multiple conditions using $or (stub assumption)
        # A real implementation would need logic to decide between $or and $and
        final_filter = {"$or": conditions}
        print(f"  > (Stub) Combined conditions with $or: {final_filter}")

    # Update the state with the final formatted filter dictionary
    # Note: We are overwriting extracted_filters from List[Dict] to Dict here.
    # This is expected as it's the final format for ChromaDB.
    return {"extracted_filters": final_filter}

# --- End Filter Generation Graph Nodes ---

@st.cache_resource
def build_filter_graph():
    """Builds the LangGraph for filter generation."""
    graph = StateGraph(FilterGenerationState)

    # Add nodes
    # graph.add_node("analyze_query", analyze_query_node) # Removed
    # graph.add_node("extract_filters", extract_filters_node) # Removed
    graph.add_node("tool_caller", tool_calling_node) # New node using tools
    graph.add_node("format_filter", format_filter_node)

    # Define edges
    graph.set_entry_point("tool_caller") # Start with the tool caller
    # graph.add_edge("analyze_query", "extract_filters") # Removed
    # graph.add_edge("extract_filters", "format_filter") # Removed
    graph.add_edge("tool_caller", "format_filter") # Connect tool caller to formatter
    graph.add_edge("format_filter", END) # End the graph flow

    # Compile the graph
    print("--- Compiling filter generation graph --- (This should only happen once)")
    compiled_graph = graph.compile()
    return compiled_graph

# Instantiate the graph when the script loads
# Consider caching if graph compilation becomes complex/slow
filter_generation_graph = build_filter_graph()
# ---> ADDED CONFIG TO NAME THE GRAPH RUN <---
filter_generation_graph = filter_generation_graph.with_config({"run_name": "Filter Generation"})
# ---> END CONFIG <---

def generate_search_filter(query: str) -> Dict:
    """
    Runs the LangGraph to generate a ChromaDB 'where' filter from the user query.
    Returns an empty dictionary if no filters are generated or an error occurs.
    """
    initial_state = FilterGenerationState(
        query=query,
        # available_metadata=AVAILABLE_METADATA_FIELDS, # Removed - Not needed for tool-based approach
        extracted_filters=[], # Initialize as empty list, matching the state type hint
        error_message=None
    )
    try:
        print(f"--- Invoking filter generation graph for query: '{query}' ---")
        # Invoke the graph
        # Add config if needed later, e.g., for recursion limits
        final_state = filter_generation_graph.invoke(initial_state)

        # Check for errors reported by graph nodes
        if final_state.get("error_message"):
            print(f"Error during filter generation graph execution: {final_state['error_message']}")
             # Use Streamlit's warning for non-critical issues during filter generation
            try:
                st.warning(f"Filter generation issue: {final_state['error_message']}")
            except Exception: # Handle cases where st isn't available
                 pass
            # Return empty filter on graph execution error
            return {}

        print(f"--- Filter generation graph completed. Final state filters: {final_state.get('extracted_filters')} ---")
        # The format_filter_node ensures extracted_filters is a Dict at the end
        return final_state.get("extracted_filters", {})
    except Exception as e:
        # Catch critical errors during the invoke call itself
        # Use Streamlit's error reporting for critical errors
        try:
            st.error(f"Critical error during filter generation invoke: {e}")
        except Exception: # Handle cases where st isn't available
            pass
        print(f"Critical error during filter generation invoke: {e}") # Log for debugging
        return {}

# @st.cache_data # Disabled: filter_dict (dict) is not hashable. Need to convert to hashable type (e.g., sorted tuple) if caching is re-enabled.
def perform_search(query, num_results, filter_dict=None):
    """Perform the vector search and return results with scores, optionally applying filters."""
    if not query.strip():
        return []

    # Determine the actual filter to pass to ChromaDB
    # Pass None if filter_dict is empty or None, otherwise pass the filter_dict
    chroma_filter = filter_dict if filter_dict else None

    print(f"--- Performing search with filter: {chroma_filter} ---")

    try:
        # Call similarity_search_with_score to get both documents and scores
        results_with_scores = vectorstore.similarity_search_with_score(
            query=query,
            k=num_results,
            filter=chroma_filter # Use the adjusted filter
        )
        
        # Extract documents and scores
        results = []
        for doc, score in results_with_scores:
            # Add the score to the document's metadata
            doc.metadata['similarity_score'] = score
            results.append(doc)
            
        return results
    except Exception as e:
        # More specific error handling might be needed depending on ChromaDB exceptions
        st.error(f"Error during search: {str(e)}")
        print(f"Search error: {e}") # Log for debugging
        return []

def prepare_download_content(query, content):
    """Prepare content for download without saving to disk"""
    # Format the content for download
    formatted_content = f"Search query: {query}\n\n{content}"
    return formatted_content

def prepare_download_content_for_summaries(query, summary_text):
    """Prepare summary content for download
    
    Args:
        query: The search query
        summary_text: The generated summary text
        
    Returns:
        Formatted content for download
    """
    content = f"Search query: {query}\n\n"
    content += "SUMMARY:\n\n" # Simplified header
    content += summary_text
    return content

def display_summary(summary_text):
    """Display the generated summary
    
    Args:
        summary_text: The generated summary text to display
    """
    # Display feedback message with clickable arrow
    st.markdown(
        "⭐ *Your feedback is important to help improve this beta version! "
        "Please take a moment to rate this response using the buttons at the end of the summary* "
        "[⬇️](#feedback-section)"
    )
    
    # Display the single summary directly
    st.markdown(standardize_summary_headline(summary_text))

@st.cache_data # Cache results for the same query to save LLM calls
def is_query_in_scope(query: str) -> bool:
    """
    Uses an LLM to determine if the user's query is related to finding
    restaurants or food within the scope of the Roadfood database.
    Stores the raw classification result in session_state.
    """
    print(f"--- Checking scope for query: '{query}' ---")
    # Clear previous result from session state
    if 'last_guardrail_result' in st.session_state:
        del st.session_state.last_guardrail_result

    try:
        llm = get_llm()

        # Load the prompt from file
        prompt_text = load_prompt_template("guardrail")
        if not prompt_text:
            st.error("Failed to load guardrail prompt! Assuming query is in scope.")
            print("  > Error: Failed to load guardrail prompt file.")
            return True # Fail safe

        prompt = ChatPromptTemplate.from_template(prompt_text)
        # ---> ADDED CONFIG TO NAME THE CHAIN RUN <---
        chain = (prompt | llm).with_config({"run_name": "Guardrail Scope Check"})
        # ---> END CONFIG <---
        # chain = prompt | llm # Original line

        response = chain.invoke({"query": query})
        raw_classification = response.content.strip()
        # Store the raw result in session state for debugging
        st.session_state.last_guardrail_result = raw_classification

        classification = raw_classification.upper()
        print(f"  > Scope classification raw: {raw_classification}")
        print(f"  > Scope classification upper: {classification}")

        # More robust check: look for the keyword within the response
        is_in_scope = "IN_SCOPE" in classification
        print(f"  > Is in scope? {is_in_scope}")
        return is_in_scope

    except Exception as e:
        st.warning(f"Error during scope check: {e}. Proceeding assuming query is in scope.")
        print(f"  > Error during scope check: {e}")
        # Store error message in session state if needed
        st.session_state.last_guardrail_result = f"Error: {e}"
        # Fail-safe: If the check fails, assume it's in scope
        return True

# Create Streamlit interface
st.title(f"Roadfood Advisor -- {EDITION} Edition")
st.markdown("Search for restaurants based on your preferences, cuisine, location, etc.")

# Example queries outside the form (these don't trigger re-renders)
example_queries = [
    "Where is some great BBQ in Texas?",
    "Unique seafood restaurants on the East Coast",
    "Famous diners in New Jersey",
    "Where can I find good pie?",
    "Historic restaurants with great burgers",
    "Highlighted eateries on the West Coast",
    "Midwest delicacies"
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
        
        pre_filter_checkbox = st.checkbox("Pre-filter results", value=True)
        
        # Always generate summary - checkbox hidden but functionality preserved
        generate_article_checkbox = True  # Constant value instead of checkbox
        # generate_article_checkbox = st.checkbox("Generate summary article", value=True)  # Commented out
        
        save_checkbox = st.checkbox("Enable download option", value=False)

        # Form submit button
        search_submitted = st.form_submit_button("Search")

# Initialize session state for feedback if it doesn't exist
if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False
if 'current_run_id' not in st.session_state:
    st.session_state.current_run_id = None

# Session state to preserve results across feedback interactions
if 'last_query' not in st.session_state:
    st.session_state.last_query = None
if 'last_summary' not in st.session_state:
    st.session_state.last_summary = None
if 'last_detailed_results' not in st.session_state:
    st.session_state.last_detailed_results = []
if 'last_generate_article_mode' not in st.session_state:
    st.session_state.last_generate_article_mode = False
if 'last_save_mode' not in st.session_state:
    st.session_state.last_save_mode = False
if 'last_current_run_id' not in st.session_state:
    st.session_state.last_current_run_id = None
if 'last_generated_filter' not in st.session_state:
    st.session_state.last_generated_filter = None

# Main content area - only process when form is submitted
if search_submitted:
    # Reset feedback state and clear previous results for new search
    st.session_state.feedback_submitted = False
    st.session_state.feedback_score = None
    st.session_state.feedback_comment = ""
    st.session_state.last_query = None
    st.session_state.last_summary = None
    st.session_state.last_detailed_results = []
    st.session_state.last_generate_article_mode = False
    st.session_state.last_save_mode = False
    st.session_state.last_current_run_id = None
    st.session_state.last_generated_filter = None

    if not query_input.strip():
        st.warning("Please enter a search query.")
    else:
        # Execute the search logic
        search_outcome = handle_search_request(
            query_input=query_input,
            num_results=num_results,
            pre_filter_checkbox=pre_filter_checkbox,
            generate_article_checkbox=generate_article_checkbox,
            save_checkbox=save_checkbox
        )

        # Store results in session state if successful
        if search_outcome and not search_outcome.get("error_message"):
            st.session_state.last_query = search_outcome["query"]
            st.session_state.last_summary = search_outcome.get("summary_result")
            st.session_state.last_detailed_results = search_outcome.get("search_results", [])
            st.session_state.last_generate_article_mode = search_outcome["generate_article_mode"]
            st.session_state.last_save_mode = search_outcome["save_mode"]
            st.session_state.last_current_run_id = search_outcome.get("run_id")
            st.session_state.last_generated_filter = search_outcome.get("generated_filter", {})
            # Store info message if present
            st.session_state.last_info_message = search_outcome.get("info_message")
            st.session_state.last_error_message = None # Clear any previous error
        elif search_outcome:
            # Store error message if search failed
            st.session_state.last_error_message = search_outcome.get("error_message")
            st.session_state.last_query = query_input # Store query even on error for context
            st.session_state.last_generated_filter = search_outcome.get("generated_filter")
        else:
             st.session_state.last_error_message = "An unexpected error occurred in handle_search_request."
             st.session_state.last_query = query_input
        
        # Trigger immediate rerun to display results from session state
        st.rerun()

# --- Display results and feedback form (runs on initial load and after search/feedback actions) ---
if st.session_state.get('last_query'):
    # Display any stored error message
    if st.session_state.get('last_error_message'):
        st.error(st.session_state.last_error_message)
        # Display Guardrail info if available even on error
        if 'last_guardrail_result' in st.session_state:
            st.sidebar.subheader("Guardrail Result (Debug)")
            st.sidebar.text(st.session_state.last_guardrail_result)

    # Display results if no error
    elif st.session_state.get('last_generate_article_mode') and st.session_state.get('last_summary'):
        display_summary(st.session_state.last_summary)
        if st.session_state.get('last_save_mode'):
            download_content = prepare_download_content_for_summaries(st.session_state.last_query, st.session_state.last_summary)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"search_results_summary_{timestamp}.txt"
            st.download_button(
                label="Download Summary",
                data=download_content,
                file_name=filename,
                mime="text/plain"
            )
    elif st.session_state.get('last_detailed_results'):
        display_detailed_results(
            st.session_state.last_detailed_results, 
            st.session_state.last_query, 
            st.session_state.last_save_mode
            )
    elif st.session_state.get('last_info_message'): # Display info message (e.g., "No results")
         st.info(st.session_state.last_info_message)
    else:
        # Should not happen if last_query is set, but as a fallback
        st.info("No results to display.") 

    # Display debug info from sidebar (moved from handle_search_request)
    generated_filter = st.session_state.get('last_generated_filter') # GET FROM SESSION STATE
    st.sidebar.subheader("Generated Filter (Debug)") # Always show subheader
    if generated_filter is not None: # Check if filter exists (could be {} or None)
        if isinstance(generated_filter, dict) and generated_filter: # Check if it's a non-empty dictionary
            st.sidebar.json(generated_filter)
        elif isinstance(generated_filter, dict) and not generated_filter: # Check if it's an empty dictionary
             st.sidebar.json({"info": "No filter generated"})
        else: # Handle other cases like None or unexpected types
             st.sidebar.text(f"Filter data: {generated_filter}")
    else:
         st.sidebar.json({"info": "No filter data available"})

    if 'last_guardrail_result' in st.session_state and not st.session_state.get('last_error_message'):
        st.sidebar.subheader("Guardrail Result (Debug)")
        st.sidebar.text(st.session_state.last_guardrail_result)

    # Display feedback form if results were successfully generated and feedback not submitted
    run_id_for_feedback = st.session_state.get('last_current_run_id')
    if run_id_for_feedback and not st.session_state.get('feedback_submitted') and not st.session_state.get('last_error_message'):
        st.divider()
        st.subheader("Was this a successful response?", anchor="feedback-section")
        
        # Use st.columns(2) for Yes/No buttons
        feedback_button_cols = st.columns(2)
        with feedback_button_cols[0]:
            if st.button("👍 Yes", key="feedback_yes", use_container_width=True): 
                score = 1
                st.session_state.feedback_score = score
                # Rerun needed to update potential display changes based on score? Maybe not.
                # st.rerun()
                
        with feedback_button_cols[1]:
            if st.button("👎 No", key="feedback_no", use_container_width=True): 
                score = 0
                st.session_state.feedback_score = score
                # st.rerun()
        
        # Display current selection state (optional)
        current_score = st.session_state.get('feedback_score')
        if current_score == 1:
            st.write("✔️ You rated this successful.")
        elif current_score == 0:
            st.write("❌ You rated this not successful.")

        # Place text area and submit button below columns
        if 'feedback_comment' not in st.session_state:
            st.session_state.feedback_comment = ""
        
        st.session_state.feedback_comment = st.text_area(
            "Reason for rating (optional):", 
            key="feedback_comment_input",
            value=st.session_state.feedback_comment 
        )

        if st.button("Submit Feedback", key="feedback_submit"):
            score = st.session_state.get('feedback_score', None) 
            comment = st.session_state.feedback_comment
            # Use the run_id stored in session state
            run_id = run_id_for_feedback 
            
            if score is None:
                st.warning("Please select '👍 Yes' or '👎 No' before submitting.")
            elif ls_client and run_id:
                try:
                    ls_client.create_feedback(
                        run_id=run_id,
                        key="user_feedback", 
                        score=score,
                        comment=comment if comment else None, 
                        source_type="user" 
                    )
                    st.success("Feedback submitted successfully! Thank you.")
                    st.session_state.feedback_submitted = True
                    # Clear form state immediately
                    st.session_state.feedback_score = None
                    st.session_state.feedback_comment = ""
                    st.rerun() # Rerun to hide the form/show success message clearly
                except Exception as fb_error:
                    st.error(f"Failed to submit feedback: {fb_error}")
                    print(f"--- Error submitting feedback to LangSmith: {fb_error} ---")
            elif not ls_client:
                 st.warning("LangSmith client not available. Feedback not submitted.")
            else: # Should not happen if we got here
                 st.error("Could not find the Run ID. Feedback not submitted.")
    elif st.session_state.get('feedback_submitted'):
        # Optionally show persistent success message if feedback was submitted for the last query
        st.success("Feedback for this search has been submitted. Thank you!")

# Display some information about the app
with st.expander("About this app"):
    st.markdown(f"""
    This app helps you discover authentic restaurants from Jane and Michael Stern's *Roadfood* guide 
    ({EDITION} edition) using natural language queries.

    Simply describe what you're looking for — like "great BBQ in Texas" or "classic diners in New England" — 
    and the AI will parse your request to find matching restaurants using hybrid search that combines 
    location filters and restaurant types with semantic similarity.

    The app automatically generates a personalized travelogue highlighting what makes each recommended 
    restaurant special, perfect for planning your next road trip to discover those colorful, local gems 
    that make travel memorable.
    """)
# Add developer options in a collapsed expander at the bottom of the sidebar
# with st.sidebar:
#     with st.expander("Developer Options", expanded=False):
#         if st.button("Reload Prompt Template"):
#             reload_prompt_template()

# Run the app
# Note: No need for if __name__ == "__main__" in Streamlit
# Streamlit apps are run with the command:  python -m streamlit run roadfood_search_app.py