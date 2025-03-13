import yaml
import os
import json
import pandas as pd
import random
from openai import OpenAI

# Configuration constants
MODEL_ID = "ft:gpt-3.5-turbo-0125:personal:roadfood-10th:BAhLsi24"  # Replace with your actual model ID
INPUT_CSV = 'data/Roadfood_10th_supplemented.csv'
TRAIN_JSONL = 'data/roadfood_finetune_train.jsonl'  # Training set created by create_openai_finetune.py
TEST_JSONL = 'data/roadfood_finetune_test.jsonl'  # Test set created by create_openai_finetune.py
PROMPT_COL = 'summary_q'
COMPLETION_COL = 'content'
NUM_SAMPLES = 5
TEMPERATURE = 0.7
USE_TEST_SET = False  # Set to True to use the test set instead of random samples

def load_credentials():
    """Load API credentials from credentials.yml file"""
    try:
        credentials = yaml.safe_load(open('credentials.yml'))
        os.environ['OPENAI_API_KEY'] = credentials['openai']
        return True
    except Exception as e:
        print(f"Error loading credentials: {str(e)}")
        return False

def test_with_jsonl(model_id, test_jsonl, num_samples=5, temperature=0.7):
    """Test the fine-tuned model with examples from a JSONL test set"""
    client = OpenAI()
    
    # Load the test set
    with open(test_jsonl, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]
    
    if len(test_data) == 0:
        print(f"No test examples found in {test_jsonl}")
        return
    
    # Select random samples if needed
    if num_samples > len(test_data):
        num_samples = len(test_data)
        print(f"Only {num_samples} test examples available")
    
    if num_samples < len(test_data):
        test_samples = random.sample(test_data, num_samples)
    else:
        test_samples = test_data
    
    print(f"\nTesting fine-tuned model: {model_id}")
    print(f"Using {num_samples} examples from test set {test_jsonl}\n")
    
    for i, example in enumerate(test_samples, 1):
        prompt = example['prompt']
        expected = example['completion'].strip()
        
        print(f"\n--- Test Example {i} ---")
        print(f"Prompt (truncated): {prompt[:150]}..." if len(prompt) > 150 else f"Prompt: {prompt}")
        
        try:
            # Call the fine-tuned model using chat completions endpoint
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=temperature
            )
            
            completion = response.choices[0].message.content.strip()
            print(f"\nModel response: {completion}")
            print(f"Expected completion: {expected}")
            
        except Exception as e:
            print(f"Error calling model: {str(e)}")

def test_with_sample(model_id, input_csv, prompt_col, num_samples=5, temperature=0.7):
    """Test the fine-tuned model with random samples from the dataset"""
    client = OpenAI()
    
    # Load the dataset
    df = pd.read_csv(input_csv)
    
    # Filter for rows with non-empty prompt column
    df_filtered = df[df[prompt_col].notna() & (df[prompt_col] != '')]
    
    if len(df_filtered) == 0:
        print(f"No valid samples found in {input_csv} with column {prompt_col}")
        return
    
    # Select random samples
    if num_samples > len(df_filtered):
        num_samples = len(df_filtered)
        print(f"Only {num_samples} valid samples available")
    
    samples = df_filtered.sample(num_samples)
    
    print(f"\nTesting fine-tuned model: {model_id}")
    print(f"Using {num_samples} random samples from {input_csv}\n")
    
    for i, (_, row) in enumerate(samples.iterrows(), 1):
        prompt = row[prompt_col]
        restaurant_name = row['Restaurant'] if 'Restaurant' in row else f"Sample {i}"
        
        print(f"\n--- Sample {i}: {restaurant_name} ---")
        print(f"Prompt (truncated): {prompt[:150]}..." if len(prompt) > 150 else f"Prompt: {prompt}")
        
        try:
            # Call the fine-tuned model using chat completions endpoint
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=temperature
            )
            
            completion = response.choices[0].message.content.strip()
            print(f"\nModel response: {completion}")
            
            # If we have the actual completion in the dataset, show it for comparison
            if 'summary_q' in row and row['summary_q']:
                print(f"Actual summary question: {row['summary_q']}")
            
        except Exception as e:
            print(f"Error calling model: {str(e)}")

def test_with_custom_prompt(model_id, prompt, temperature=0.7):
    """Test the fine-tuned model with a custom prompt"""
    client = OpenAI()
    
    print(f"\nTesting fine-tuned model: {model_id}")
    print(f"Using custom prompt: {prompt[:150]}..." if len(prompt) > 150 else f"Using custom prompt: {prompt}")
    
    try:
        # Call the fine-tuned model using chat completions endpoint
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=temperature
        )
        
        completion = response.choices[0].message.content.strip()
        print(f"\nModel response: {completion}")
        
    except Exception as e:
        print(f"Error calling model: {str(e)}")

def main():
    # Load API credentials
    if not load_credentials():
        print("Failed to load credentials. Exiting.")
        return
    
    # Test with the test set if available and enabled
    if USE_TEST_SET and os.path.exists(TEST_JSONL):
        test_with_jsonl(MODEL_ID, TEST_JSONL, NUM_SAMPLES, TEMPERATURE)
    else:
        # Otherwise test with random samples from the dataset
        test_with_sample(MODEL_ID, INPUT_CSV, PROMPT_COL, NUM_SAMPLES, TEMPERATURE)
    
    # Example of testing with a custom prompt
    # custom_prompt = """
    # "In the rough" was never so agreeable. On an alfresco dining
    # area perfumed by sea breezes and protected from marauding
    # seagulls, one dines on lobster steamed to such perfect plumpness that meat erupts when the shell is broken.
    # """
    # test_with_custom_prompt(MODEL_ID, custom_prompt, TEMPERATURE)

if __name__ == "__main__":
    main() 