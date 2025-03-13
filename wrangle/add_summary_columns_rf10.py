import pandas as pd
import yaml
import os
import json
import time
import anthropic
import re


# OPENAI API SETUP
os.environ['ANTHROPIC_API_KEY'] = yaml.safe_load(open('credentials.yml'))['anthropic']
MODEL = 'claude-3-7-sonnet-20250219'

# Maximum number of rows to process
MAX_ROWS = 1040

# Initialize Anthropic client
client = anthropic.Anthropic()

# Print Anthropic client version for debugging
print(f"Anthropic library version: {anthropic.__version__}")

df = pd.read_csv('data/Roadfood_10th_supplemented.csv')

print("Columns:", ", ".join(df.columns))

# Check if 'sig_item' column exists, if not add it
if 'sig_item' not in df.columns:
    df['sig_item'] = ""
    print("Added 'sig_item' column")

# Check if 'summary_q' column exists, if not add it
if 'summary_q' not in df.columns:
    df['summary_q'] = ""
    print("Added 'summary_q' column")


# Replace NaN values with empty strings
df['sig_item'] = df['sig_item'].fillna('')
df['summary_q'] = df['summary_q'].fillna('')

print("Replaced any NaN values with empty strings")

# Process rows until we've sent MAX_ROWS new requests
requests_sent = 0
row_index = 0

while requests_sent < MAX_ROWS and row_index < len(df):
    # Only skip if BOTH sig_item AND summary_q have data (not empty)
    if df.loc[row_index, 'sig_item'] != "" and df.loc[row_index, 'summary_q'] != "":
        print(f"Skipping row {row_index} - already has complete data")
        row_index += 1
        continue
    
    # Otherwise, process this row
    restaurant_name = df.loc[row_index, 'Restaurant']
    print(f"sending {restaurant_name} (row {row_index}, request {requests_sent + 1}/{MAX_ROWS})")
    
    # Get the content for this restaurant
    content = df.loc[row_index, 'content']
    
    # Create the prompt for Claude
    prompt = f"""
    Here is a writeup about a restaurant called "{restaurant_name}":

    {content}

    Based on this writeup:

    1. Identify the single main food item this restaurant is known for. Include minimal style descriptors 
    only if they're essential to understanding what makes this food distinctive 
    (e.g., "thin-crust pizza" instead of "Colony-style thin-crust pizza", or "fried seafood" instead of 
    "New England-style fried seafood")  If there is a specific name for the signature item, use that.  
    If not, just use the stylistic category.

    2. Provide a question someone might ask that would make this restaurant writeup a perfect answer 
    (e.g., "Where can I find unique hot dogs in Chicago?" rather than questions about specific menu details)

    Return your answer as a JSON object with the following format:
    {{
        "signature_item": "Food category with regional style or distinctive approach",
        "summary_question": "A question that would make this restaurant writeup a perfect answer"
    }}

    Just return the JSON object, nothing else.
    """

    # Send to Claude and get response
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            temperature=0.4,
            system="You are a helpful assistant that extracts information from restaurant writeups and returns it in JSON format.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract the response content
        response_text = response.content[0].text
        
        # Print just the first 100 characters of the response for brevity
        print("\nResponse (truncated):", response_text[:100] + "..." if len(response_text) > 100 else response_text)
        
        # Clean the response - remove markdown code blocks if present
        # This handles responses like ```json { ... } ```
        cleaned_response = re.sub(r'^```json\s*|\s*```$', '', response_text.strip())
        
        # Parse the JSON response
        try:
            result = json.loads(cleaned_response)
            
            # Update the dataframe
            df.loc[row_index, 'sig_item'] = result.get('signature_item', '')
            df.loc[row_index, 'summary_q'] = result.get('summary_question', '')
            
            print(f"Updated row {row_index}:")
            print(f"  Signature item: {df.loc[row_index, 'sig_item']}")
            print(f"  Summary question: {df.loc[row_index, 'summary_q']}")
            
            # Save after each successful update
            df.to_csv('data/Roadfood_10th_supplemented.csv', index=False)
            
            # Increment counters
            requests_sent += 1
            row_index += 1
            
            # Add a small delay to avoid rate limiting
            time.sleep(1)
            
        except json.JSONDecodeError:
            print(f"Failed to parse JSON for {restaurant_name}. Response (truncated): {response_text[:100]}...")
            row_index += 1
            
    except Exception as e:
        print(f"Error processing {restaurant_name}: {str(e)}")
        row_index += 1

# Final save of the updated dataframe
df.to_csv('data/Roadfood_10th_supplemented.csv', index=False)
print(f"Dataframe updated and saved. Processed {requests_sent} requests.")
