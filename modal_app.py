import os
import shlex
import subprocess
from pathlib import Path

import modal

# Path to the Streamlit app script
streamlit_script_local_path = Path("roadfood_search_app.py")
streamlit_script_remote_path = "/root/roadfood_search_app.py"

# Create an image with the necessary dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "streamlit>=1.45.0",
        "langchain>=0.3.24",
        "langchain-openai>=0.3.14",
        "langchain-core>=0.3.56",
        "langchain-chroma>=0.2.2",
        "langgraph>=0.4.0",
        "openai>=1.3.0",
        "pyyaml>=6.0",
        "chromadb>=0.4.18",
        "modal>=0.53.3",
        "pydantic==2.9.2",
        "pydantic-core>=2.23.4",
    )
    .add_local_file(
        streamlit_script_local_path,
        streamlit_script_remote_path,
    )
)

# Define the Modal app
app = modal.App(name="roadfood-search-app", image=image)

# Check if the Streamlit script exists
if not streamlit_script_local_path.exists():
    raise RuntimeError(
        "roadfood_search_app.py not found! Make sure the Streamlit app script is in the same directory."
    )

# Mount the local directories that contain necessary files
LOCAL_PROMPTS_DIR = Path("prompts")
LOCAL_DATA_DIR = Path("data")

@app.function(
    mounts=[
        modal.Mount.from_local_dir(LOCAL_PROMPTS_DIR, remote_path="/root/prompts"),
        modal.Mount.from_local_dir(LOCAL_DATA_DIR, remote_path="/root/data"),
    ],
    secrets=[modal.Secret.from_name("openai-api-key")],
    timeout=600,
    allow_concurrent_inputs=100,
)
@modal.web_server(8000)
def run():
    # Set up environment variables
    os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]
    
    # Run Streamlit as a subprocess
    target = shlex.quote(streamlit_script_remote_path)
    cmd = f"streamlit run {target} --server.port 8000 --server.enableCORS=false --server.enableXsrfProtection=false"
    subprocess.Popen(cmd, shell=True)

# This is used when running the app locally with `modal serve modal_app.py`
if __name__ == "__main__":
    app.serve() 