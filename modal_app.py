import os
import shlex
import subprocess
from pathlib import Path

import modal

# Path to the Streamlit app script
streamlit_script_local_path = Path("rfadvisr_app.py")
streamlit_script_remote_path = "/root/rfadvisr_app.py"
requirements_local_path = Path("requirements.txt")
requirements_remote_path = "/root/requirements.txt"

# Read requirements from requirements.txt
with open(requirements_local_path) as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Create an image with the necessary dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(*requirements)
    .add_local_file(
        requirements_local_path,
        requirements_remote_path,
        copy=True
    )
    .add_local_file(
        streamlit_script_local_path,
        streamlit_script_remote_path,
        copy=True
    )
    .add_local_dir(
        Path("prompts"),
        remote_path="/root/prompts",
    )
    .add_local_dir(
        Path("data"),
        remote_path="/root/data",
    )
    .add_local_dir(
        Path("app"),
        remote_path="/root/app",
    )
)

# Define the Modal app
app = modal.App(name="rfadvisr-app", image=image)

# Check if the Streamlit script exists
if not streamlit_script_local_path.exists():
    raise RuntimeError(
        "rfadvisr_app.py not found! Make sure the Streamlit app script is in the same directory."
    )


@app.function(
    secrets=[
        modal.Secret.from_name("openai-api-key"),
        modal.Secret.from_name("langsmith-api-key"),
        modal.Secret.from_name("supabase-rfadvisr_result_scores-key")
    ],
    timeout=600,
    allow_concurrent_inputs=100,
)
@modal.web_server(8000)
def run():
    # Set up environment variables
    os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]
    
    if "LANGSMITH_API_KEY" in os.environ:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
        print("--- LangSmith environment variables set for Modal deployment ---")
    else:
        print("--- LangSmith API key not found in Modal environment, tracing disabled ---")
    
    # Set up Supabase environment variables
    if all(key in os.environ for key in ["SUPABASE_URL", "SUPABASE_ANON_KEY", "SUPABASE_SERVICE_KEY"]):
        print("--- Supabase environment variables set for Modal deployment ---")
    else:
        print("--- Supabase credentials not found in Modal environment ---")
    
    # Run Streamlit as a subprocess
    target = shlex.quote(streamlit_script_remote_path)
    cmd = f"streamlit run {target} --server.port 8000 --server.enableCORS=false --server.enableXsrfProtection=false"
    subprocess.Popen(cmd, shell=True)

# This is used when running the app locally with `modal serve modal_app.py`
if __name__ == "__main__":
    app.serve() 