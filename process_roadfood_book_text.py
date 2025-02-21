import csv
import pandas as pd
import os

# Constants
DATA_DIR = "data"
INPUT_FILE = os.path.join(DATA_DIR, "Roadfood_ 10th_edition_simplified.txt")
OUTPUT_FILE = os.path.join(DATA_DIR, "restaurant_titles.csv")

SKIP_TITLES = {'B.J.'}  # Strings that should never be considered titles
ALWAYS_TITLES = {'NICK\'S FAMOUS ROAST BEEF'}  # Strings that are always titles

US_STATES = {
    'ALABAMA', 'ALASKA', 'ARIZONA', 'ARKANSAS', 'CALIFORNIA', 'COLORADO', 'CONNECTICUT',
    'DELAWARE', 'FLORIDA', 'GEORGIA', 'HAWAII', 'IDAHO', 'ILLINOIS', 'INDIANA', 'IOWA',
    'KANSAS', 'KENTUCKY', 'LOUISIANA', 'MAINE', 'MARYLAND', 'MASSACHUSETTS', 'MICHIGAN',
    'MINNESOTA', 'MISSISSIPPI', 'MISSOURI', 'MONTANA', 'NEBRASKA', 'NEVADA', 'NEW HAMPSHIRE',
    'NEW JERSEY', 'NEW MEXICO', 'NEW YORK', 'NORTH CAROLINA', 'NORTH DAKOTA', 'OHIO',
    'OKLAHOMA', 'OREGON', 'PENNSYLVANIA', 'RHODE ISLAND', 'SOUTH CAROLINA', 'SOUTH DAKOTA',
    'TENNESSEE', 'TEXAS', 'UTAH', 'VERMONT', 'VIRGINIA', 'WASHINGTON', 'WEST VIRGINIA',
    'WISCONSIN', 'WYOMING'
}

def read_text_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def is_restaurant_title(line):
    # Skip empty lines
    if not line.strip():
        return False
    
    line = line.strip()
    
    # Check special cases first
    if line in SKIP_TITLES:
        return False
    if line in ALWAYS_TITLES:
        return True
        
    # Skip known non-title patterns
    skip_patterns = ["BLD ", "LD ", "BL ", "BD ", "B ", "L ", "D "]
    if any(line.startswith(pattern) for pattern in skip_patterns):
        return False
    
    # Skip if the line is just a state name
    if line in US_STATES:
        # print(f"Skipping state name: {line}")
        return False
    
    # Skip if first word contains numbers (likely an address)
    first_word = line.split()[0] if line.split() else ""
    if all(c.isdigit() for c in first_word):
        return False
    
    # Look for consecutive capital letters at start of line
    consecutive_caps = 0
    for char in first_word:
        if char.isupper():
            consecutive_caps += 1
            if consecutive_caps >= 2:  # Found two consecutive capitals
                return True
        elif not char.isalpha():  # Skip punctuation
            continue
        else:  # Lowercase letter resets the count
            consecutive_caps = 0
            
    return False

def is_similar_to_current(text, current_title):
    if not current_title:
        return False
        
    # Convert both to uppercase and remove punctuation for comparison
    text = ''.join(c for c in text.upper() if c.isalnum())
    current = ''.join(c for c in current_title.upper() if c.isalnum())
    
    # Check if one starts with the other
    return text.startswith(current) or current.startswith(text)

def normalize_quotes(text):
    """Replace curly quotes with straight quotes"""
    # Replace curly single quotes with straight single quote
    text = text.replace('\u2018', "'").replace('\u2019', "'")  # Left and right single quotes
    # Replace curly double quotes with straight double quote
    text = text.replace('\u201c', '"').replace('\u201d', '"')  # Left and right double quotes
    return text

def find_restaurant_entries(content):
    # Normalize quotes in the entire content first
    content = normalize_quotes(content)
    
    entries = []
    lines = content.split('\n')
    
    # First pass - remove state names and clean lines
    cleaned_lines = []
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("? "):
            line = line[2:]  # Remove the "? " prefix
            
        # Skip state names entirely
        if line in US_STATES:
            # print(f"Removing state name at line {i+1}: {line}")
            continue
            
        cleaned_lines.append((line, i+1))
    
    # Second pass - find restaurant titles
    current_title = None
    for line, line_num in cleaned_lines:
        if is_restaurant_title(line):
            # Always accept special cases regardless of similarity
            if line in ALWAYS_TITLES:
                current_title = line
                entries.append({
                    'title': line,
                    'line_number': line_num
                })
                continue
                
            # Skip if similar to current title
            if is_similar_to_current(line, current_title):
                print(f"Skipping similar to current title {current_title} at line {line_num}: {line}")
                continue
                
            # Found a new distinct title
            current_title = line
            entries.append({
                'title': line,
                'line_number': line_num
            })
    
    return entries

def save_titles_to_csv(entries, output_file):
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Convert entries to pandas DataFrame
    df = pd.DataFrame([entry['title'] for entry in entries], columns=['Title'])
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    return df

def main():
    content = read_text_file(INPUT_FILE)
    
    if content:
        print(f"Successfully read file. Total length: {len(content)} characters")
        
        entries = find_restaurant_entries(content)
        print(f"\nFound {len(entries)} potential restaurant entries.")
        
        # Print first 5 entries as sample
        print("\nFirst 5 entries found:")
        for entry in entries[:5]:
            print(f"Line {entry['line_number']}: {entry['title']}")

        # Save to CSV using pandas
        df = save_titles_to_csv(entries, OUTPUT_FILE)
        print(f"\nSaved {len(df)} titles to {OUTPUT_FILE}")
        print("\nDataFrame head:")
        print(df.head())
    else:
        print("Failed to read file")

if __name__ == "__main__":
    main() 