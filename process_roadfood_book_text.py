import csv
import pandas as pd
import os
import re

# Constants
DATA_DIR = "data"
INPUT_FILE = os.path.join(DATA_DIR, "Roadfood_ 10th_edition_simplified.txt")
OUTPUT_FILE = os.path.join(DATA_DIR, "restaurant_titles.csv")
OUTPUT_ADDRESSES_FILE = os.path.join(DATA_DIR, "addresses.csv")
PROCESSED_OUTPUT_FILE = os.path.join(DATA_DIR, "processed_roadfood.txt")

SKIP_TITLES = {'B.J.'}  # Strings that should never be considered titles
ALWAYS_TITLES = {'NICK\'S FAMOUS ROAST BEEF'}  # Strings that are always titles
HOURS_PATTERNS = ["BLD ", "LD ", "BL ", "BD ", "B ", "L ", "D "]  # Meal period indicators

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

# Add after other constants, before functions
STATE_ABBREVS = {
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
}

def mark_addresses(content):
    """
    Find addresses and normalize their formatting:
    1. Ensure one linebreak before address
    2. Remove internal linebreaks (replace with space)
    3. Ensure one linebreak after address
    4. Add start and end markers
    """
    states = '|'.join(STATE_ABBREVS)
    
    """
    Pattern matches:
    - Numbers followed by whitespace
    - Then alphanumeric text (including possible linebreak)
    - Ending with ", XX" where XX is state abbreviation
    Updated pattern to include parenthetical content
    Updated pattern to include Route/Rte. as optional prefix
    """
    address_pattern = r'((?:Route |Rte\. )?\d+\s[A-Za-z0-9\s.,\'"-()]+?,\s(?:' + states + r'))'
    
    def format_address(match):
        address = match.group(1)
        # Replace any internal linebreaks with space
        address = ' '.join(address.split())
        # Add start and end markers and ensure one linebreak before and after
        return f'\n|address start| {address} |address end|\n'
    
    # Replace addresses with formatted versions
    content = re.sub(address_pattern, format_address, content)
    
    # Clean up any duplicate markers and their associated linebreaks/spaces
    content = re.sub(r'\|address start\|\s+\|address start\|', '|address start|', content)
    content = re.sub(r'\|address end\|\s+\|address end\|', '|address end|', content)
    
    return content

def mark_urls(content):
    """
    Find URLs and add markers around them.
    URLs are defined as 2-40 characters with no spaces followed by .com/.biz/.org/.net
    Can have optional trailing slash (which will be removed)
    Must be on their own line
    """
    # Pattern matches:
    # ^ - start of line
    # \s* - optional whitespace
    # [^\s]{2,40} - 2-40 non-whitespace characters
    # \.(com|biz|org|net) - literal ".com" or ".biz" or ".org" or ".net"
    # /? - optional trailing slash
    # \s*$ - optional whitespace and end of line
    url_pattern = r'^(\s*[^\s]{2,40}\.(com|biz|org|net)/?)\s*$'
    
    def format_url(match):
        url = match.group(1).strip()
        # Remove trailing slash if present
        url = url.rstrip('/')
        return f'|URL start| {url} |URL end|\n'
    
    # Process line by line to ensure we only match full lines
    lines = content.split('\n')
    processed_lines = [re.sub(url_pattern, format_url, line) for line in lines]
    return '\n'.join(processed_lines)

def left_trim_lines(content):
    """Left trim all lines in the content while preserving empty lines"""
    return '\n'.join(line.lstrip() for line in content.split('\n'))

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
    if any(line.startswith(pattern) for pattern in HOURS_PATTERNS):
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

def normalize_quotes(text):
    """Replace curly quotes with straight quotes"""
    # Replace curly single quotes with straight single quote
    text = text.replace('\u2018', "'").replace('\u2019', "'")  # Left and right single quotes
    # Replace curly double quotes with straight double quote
    text = text.replace('\u201c', '"').replace('\u201d', '"')  # Left and right double quotes
    return text

def extract_address(lines, start_idx):
    """
    Extract address from lines starting at start_idx.
    Returns tuple of (address, lines_consumed) or (None, 0) if no address found.
    """
    # Check if address is on current line after double space
    current_line = lines[start_idx].strip()
    parts = current_line.split('  ', 1)
    if len(parts) > 1:
        addr = parts[1].strip()
        # Check if this part ends with a state abbreviation
        if any(addr.endswith(f", {state[:2]}") for state in US_STATES):
            return addr, 0

    # Look at next line(s)
    address_parts = []
    lines_consumed = 0
    
    for i in range(1, 3):  # Look up to 2 lines ahead
        if start_idx + i >= len(lines):
            break
            
        next_line = lines[start_idx + i].strip()
        # Skip empty lines
        if not next_line:
            continue
            
        # Check if line ends with state abbreviation
        if any(next_line.endswith(f", {state[:2]}") for state in US_STATES):
            address_parts.append(next_line)
            lines_consumed = i
            break
            
    if address_parts:
        return " ".join(address_parts), lines_consumed
    
    return None, 0

def find_restaurant_entries(content):
    # Normalize quotes in the entire content first
    content = normalize_quotes(content)
    entries = []
    lines = content.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("? "):
            line = line[2:]
        
        # Skip state names
        if line in US_STATES:
            i += 1
            continue
            
        if is_restaurant_title(line):
            # Get just the title part (in case address is on same line)
            title = line.split('  ', 1)[0].strip()
            
            # Look for address
            address, lines_consumed = extract_address(lines, i)
            
            entry = {
                'title': title,
                'address': address if address else '',
                'line_number': i + 1
            }
            entries.append(entry)
            
            # Skip lines consumed by address
            i += lines_consumed
        i += 1
    
    return entries

def save_titles_to_csv(entries, output_file):
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Convert entries to pandas DataFrame
    df = pd.DataFrame(entries)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    return df

def find_all_addresses(content):
    """Extract all addresses from content using the address pattern"""
    states = '|'.join(state[:2] for state in US_STATES)
    address_pattern = rf'(\d+\s[A-Za-z0-9\s.,\'"-]+?,\s(?:{states}))'
    
    # Find all matches
    addresses = re.findall(address_pattern, content)
    # Clean up addresses (remove internal whitespace)
    addresses = [' '.join(addr.split()) for addr in addresses]
    return addresses

def mark_phones(content):
    """
    Find phone numbers and add markers around them.
    Phone numbers are either:
    - ###-###-#### format
    - The text "No phone"
    Must be at start of line (but can have content after)
    """
    # Pattern matches:
    # ^ - start of line
    # \s* - optional whitespace
    # Either:
    #   - \d{3}-\d{3}-\d{4} (###-###-####)
    #   - No phone
    phone_pattern = r'^(\s*(?:\d{3}-\d{3}-\d{4}|No phone))'
    
    def format_phone(match):
        phone = match.group(1).strip()
        return f'|phone start| {phone} |phone end|\n'
    
    # Process line by line
    lines = content.split('\n')
    processed_lines = [re.sub(phone_pattern, format_phone, line) for line in lines]
    return '\n'.join(processed_lines)

def main():
    content = read_text_file(INPUT_FILE)
    
    if content:
        print(f"Successfully read file. Total length: {len(content)} characters")
        
        # Process the content and save to new file
        content = left_trim_lines(content)
        processed_content = mark_addresses(content)
        processed_content = left_trim_lines(processed_content)
        processed_content = mark_urls(processed_content)
        processed_content = left_trim_lines(processed_content)
        processed_content = mark_phones(processed_content)
        processed_content = left_trim_lines(processed_content)
        
        os.makedirs(os.path.dirname(PROCESSED_OUTPUT_FILE), exist_ok=True)
        with open(PROCESSED_OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write(processed_content)
        print(f"\nProcessed content saved to {PROCESSED_OUTPUT_FILE}")
        
        # Find and save addresses
        addresses = find_all_addresses(content)
        df_addresses = pd.DataFrame(addresses, columns=['address'])
        df_addresses.to_csv(OUTPUT_ADDRESSES_FILE, index=False)
        
        print(f"\nFound {len(addresses)} addresses.")
        print("\nFirst 5 addresses found:")
        for addr in addresses[:5]:
            print(f"    {addr}")
        
        # Continue with existing restaurant entries processing
        entries = find_restaurant_entries(content)
        print(f"\nFound {len(entries)} potential restaurant entries.")
        
        # Print first 5 entries as sample
        print("\nFirst 5 entries found:")
        for entry in entries[:5]:
            print(f"Line {entry['line_number']}: {entry['title']}")
            if 'address' in entry:
                print(f"    Address: {entry['address']}")

        # Save to CSV using pandas
        df = save_titles_to_csv(entries, OUTPUT_FILE)
        print(f"\nSaved {len(df)} entries to {OUTPUT_FILE}")
        print("\nDataFrame head:")
        print(df.head())
    else:
        print("Failed to read file")

if __name__ == "__main__":
    main() 