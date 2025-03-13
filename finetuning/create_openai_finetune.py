import pandas as pd
import yaml
import os
import json
import time
import openai
from openai import OpenAI
import random
from sklearn.model_selection import train_test_split

# Configuration constants
INPUT_CSV = 'data/Roadfood_10th_supplemented.csv'
OUTPUT_TRAIN_JSONL = 'data/roadfood_finetune_train.jsonl'
OUTPUT_TEST_JSONL = 'data/roadfood_finetune_test.jsonl'
PROMPT_COL = 'summary_q'
COMPLETION_COL = 'content'
BASE_MODEL = 'gpt-3.5-turbo'
SAMPLE_SIZE = 250
TEST_SIZE = 0.2  # 20% of data for testing
CREATE_ONLY = True  # Just create the JSONL file, don't submit job

def load_credentials():
    """Load API credentials from credentials.yml file"""
    try:
        credentials = yaml.safe_load(open('credentials.yml'))
        os.environ['OPENAI_API_KEY'] = credentials['openai']
        return True
    except Exception as e:
        print(f"Error loading credentials: {str(e)}")
        return False

def create_finetune_jsonl(input_csv, output_train_jsonl, output_test_jsonl, prompt_col, completion_col, sample_size, test_size=0.2):
    """
    Create JSONL files for fine-tuning from CSV data, with train/test split.
    
    For single-turn completions, we use the format:
    {"prompt": "<prompt text>", "completion": "<ideal generated text>"}
    """
    print(f"Reading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Check if columns exist
    if prompt_col not in df.columns:
        raise ValueError(f"Column '{prompt_col}' not found in CSV")
    if completion_col not in df.columns:
        raise ValueError(f"Column '{completion_col}' not found in CSV")
    
    # Replace NaN values with empty strings
    df[prompt_col] = df[prompt_col].fillna('')
    df[completion_col] = df[completion_col].fillna('')
    
    # Filter out rows where Crossout == 'y'
    if 'Crossout' in df.columns:
        df = df[df['Crossout'] != 'y']
        print(f"Filtered out rows where Crossout == 'y', {len(df)} rows remaining")
    
    # Filter out rows where either column is empty
    df_filtered = df[(df[prompt_col] != '') & (df[completion_col] != '')]
    print(f"Filtered out rows with empty values, {len(df_filtered)} rows remaining")
    
    # Randomly select sample_size rows
    if len(df_filtered) > sample_size:
        df_sampled = df_filtered.sample(sample_size, random_state=42)
        print(f"Randomly selected {sample_size} rows for dataset")
    else:
        df_sampled = df_filtered
        print(f"Using all {len(df_filtered)} available rows (less than requested {sample_size})")
    
    # Split into training and test sets
    train_df, test_df = train_test_split(df_sampled, test_size=test_size, random_state=42)
    print(f"Split into {len(train_df)} training examples and {len(test_df)} test examples")
    
    # Create training JSONL file
    print(f"Creating training data with {len(train_df)} examples...")
    with open(output_train_jsonl, 'w', encoding='utf-8') as f:
        for _, row in train_df.iterrows():
            # For completion fine-tuning, we need to add a space and end token
            completion = " " + row[completion_col] + "\n"
            
            # Create the JSON object and write to file
            json_obj = {
                "prompt": row[prompt_col],
                "completion": completion
            }
            f.write(json.dumps(json_obj) + '\n')
    
    print(f"Created training JSONL file at {output_train_jsonl}")
    
    # Create test JSONL file
    print(f"Creating test data with {len(test_df)} examples...")
    with open(output_test_jsonl, 'w', encoding='utf-8') as f:
        for _, row in test_df.iterrows():
            # For completion fine-tuning, we need to add a space and end token
            completion = " " + row[completion_col] + "\n"
            
            # Create the JSON object and write to file
            json_obj = {
                "prompt": row[prompt_col],
                "completion": completion
            }
            f.write(json.dumps(json_obj) + '\n')
    
    print(f"Created test JSONL file at {output_test_jsonl}")
    
    return output_train_jsonl, output_test_jsonl

def upload_file(file_path):
    """Upload a file to OpenAI and return the file ID"""
    client = OpenAI()
    
    print(f"Uploading file {file_path} to OpenAI...")
    with open(file_path, "rb") as file:
        response = client.files.create(
            file=file,
            purpose="fine-tune"
        )
    
    file_id = response.id
    print(f"File uploaded with ID: {file_id}")
    return file_id

def create_finetune_job(training_file_id, validation_file_id=None, model="gpt-3.5-turbo"):
    """Create a fine-tuning job with the uploaded files"""
    client = OpenAI()
    
    print(f"Creating fine-tuning job with training file {training_file_id}...")
    
    # Create job parameters
    job_params = {
        "training_file": training_file_id,
        "model": model
    }
    
    # Add validation file if provided
    if validation_file_id:
        job_params["validation_file"] = validation_file_id
        print(f"Using validation file {validation_file_id}")
    
    # Create the job
    response = client.fine_tuning.jobs.create(**job_params)
    
    job_id = response.id
    print(f"Fine-tuning job created with ID: {job_id}")
    return job_id

def check_finetune_status(job_id):
    """Check the status of a fine-tuning job"""
    client = OpenAI()
    
    print(f"Checking status of fine-tuning job {job_id}...")
    response = client.fine_tuning.jobs.retrieve(job_id)
    
    status = response.status
    print(f"Job status: {status}")
    
    # Print additional details if available
    if hasattr(response, 'fine_tuned_model') and response.fine_tuned_model:
        print(f"Fine-tuned model ID: {response.fine_tuned_model}")
    
    if hasattr(response, 'finished_at') and response.finished_at:
        print(f"Finished at: {response.finished_at}")
    
    return status

def main():
    # Load API credentials
    if not load_credentials():
        print("Failed to load credentials. Exiting.")
        return
    
    # Create JSONL files with train/test split
    train_jsonl, test_jsonl = create_finetune_jsonl(
        INPUT_CSV, 
        OUTPUT_TRAIN_JSONL, 
        OUTPUT_TEST_JSONL,
        PROMPT_COL, 
        COMPLETION_COL,
        SAMPLE_SIZE,
        TEST_SIZE
    )
    
    if CREATE_ONLY:
        print("JSONL files created. Exiting without creating fine-tuning job.")
        return
    
    # Upload files to OpenAI
    train_file_id = upload_file(train_jsonl)
    test_file_id = upload_file(test_jsonl)
    
    # Create fine-tuning job with both training and validation files
    job_id = create_finetune_job(train_file_id, test_file_id, BASE_MODEL)
    
    # Check initial status
    status = check_finetune_status(job_id)
    
    print("\nFine-tuning job created successfully!")
    print(f"Job ID: {job_id}")
    print(f"Initial status: {status}")

if __name__ == "__main__":
    main() 