import gradio as gr
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

# Create Gradio interface
with gr.Blocks(title=f"Roadfood {EDITION} Edition Search") as demo:
    gr.Markdown(f"# Roadfood {EDITION} Edition Restaurant Search")
    gr.Markdown("Search for restaurants based on your preferences, cuisine, location, etc.")
    
    with gr.Row():
        with gr.Column():
            query_input = gr.Textbox(
                label="What are you looking for?",
                placeholder="e.g., 'best BBQ in Texas' or 'unique seafood restaurants'"
            )
            num_results = gr.Slider(
                minimum=1,
                maximum=10,
                value=4,
                step=1,
                label="Number of results"
            )
            save_checkbox = gr.Checkbox(label="Save results to file", value=False)
            search_button = gr.Button("Search")
        
    results_output = gr.Markdown(label="Search Results")
    
    search_button.click(
        fn=search_restaurants,
        inputs=[query_input, num_results, save_checkbox],
        outputs=results_output
    )
    
    gr.Examples(
        examples=[
            ["Where is the best BBQ?"],
            ["Unique seafood restaurants on the East Coast"],
            ["Famous diners in New Jersey"],
            ["Where can I find good pie?"],
            ["Historic restaurants with great burgers"]
        ],
        inputs=query_input
    )

# Launch the app
if __name__ == "__main__":
    demo.launch() 