import csv
import pandas as pd
import os
import re

# Constants
DATA_DIR = "data"
INPUT_FILE = os.path.join(DATA_DIR, "Roadfood_ 10th_edition_simplified.txt")
OUTPUT_FILE = os.path.join(DATA_DIR, "restaurant_titles.csv")
OUTPUT_ADDRESSES_FILE = os.path.join(DATA_DIR, "addresses.csv")
OUTPUT_URLS_FILE = os.path.join(DATA_DIR, "urls.csv")
PROCESSED_OUTPUT_FILE = os.path.join(DATA_DIR, "processed_roadfood.txt")

SKIP_TITLES = {'B.J.'}  # Strings that should never be considered titles
ALWAYS_TITLES = {}  # Strings that are always titles
# ALWAYS_TITLES = {'NICK\'S FAMOUS ROAST BEEF'}  # Strings that are always titles
ALWAYS_URLS = {'cornelldairybar.cfm'}  # Strings that should always be considered part of URLs
ALWAYS_ADDRESSES = {
    ('605 8th Ave. S. Nashville, Tennessee', '605 8th Ave. S. Nashville, TN')  # (input pattern, replacement)
}
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

def left_trim_lines(content):
    """Left trim all lines in the content while preserving empty lines"""
    return '\n'.join(line.lstrip() for line in content.split('\n'))

def normalize_quotes(text):
    """Replace curly quotes with straight quotes"""
    # Replace curly single quotes with straight single quote
    text = text.replace('\u2018', "'").replace('\u2019', "'")  # Left and right single quotes
    # Replace curly double quotes with straight double quote
    text = text.replace('\u201c', '"').replace('\u201d', '"')  # Left and right double quotes
    return text

def remove_standalone_states(content):
    """Remove lines that consist only of a US state name"""
    lines = content.split('\n')
    filtered_lines = [line for line in lines if line.strip() not in US_STATES]
    return '\n'.join(filtered_lines)

def remove_question_mark_prefix(content):
    """
    Remove "? " prefix when followed by capital letters.
    Pattern:
    - Start of line
    - Literal "? "
    - One or more consecutive capital letters (ignoring punctuation)
    """
    # Pattern matches:
    # ^ - start of line
    # \? \s* - question mark followed by optional whitespace
    # (?=[A-Z]) - positive lookahead for capital letter
    pattern = r'^\?\s*(?=[A-Z])'
    
    # Process line by line to ensure we only match at start of lines
    lines = content.split('\n')
    processed_lines = [re.sub(pattern, '', line) for line in lines]
    return '\n'.join(processed_lines)

def mark_addresses(content):
    """
    Find addresses and normalize their formatting:
    1. Check for ALWAYS_ADDRESSES first
    2. Ensure one linebreak before address
    3. Remove internal linebreaks (replace with space)
    4. Ensure one linebreak after address
    5. Add start and end markers
    Only match addresses that are preceded by either:
    - Two or more consecutive capital letters (ignoring punctuation)
    - A |URL end| marker (which should be preserved)
    """
    # Handle ALWAYS_ADDRESSES first
    for pattern, replacement in ALWAYS_ADDRESSES:
        content = content.replace(pattern, f'\n|address start| {replacement} |address end|\n')
    
    states = '|'.join(STATE_ABBREVS)
    
    """
    Pattern matches either:
    - Two or more consecutive capital letters (allowing punctuation between) followed by whitespace, or
    - |URL end| marker
    Then captures everything from:
    - First number encountered (e.g., street number, route number), but not if preceded by "US "
    - Through all subsequent text (including additional numbers, street names, etc.)
    - Until reaching ", XX" where XX is state abbreviation
    - Including any optional parenthetical content after the state
    
    Examples:
    "MIKE'S KITCHEN 170 Randall St. (at Tabor-Franchi VFW Post 239) Cranston, RI"
    "SUGAR'S 1799 State Rd. 68 Embudo, NM"
    "CHOPE'S 16145 S. Hwy. 28 La Mesa, NM"
    Will not match: "along US 17, SC"
    """
    address_pattern = r'((?:\|URL end\||[A-Z][^a-z\s]*[A-Z][^a-z\s]*[A-Z\s]*?))\s*(?<!US\s)(\d+[^|]*?,\s(?:' + states + r')(?:\s*\([^)]+\))?)'

    def format_address(match):
        prefix, address = match.groups()
        # Replace any internal linebreaks with space
        address = ' '.join(address.split())
        # Add start and end markers and ensure one linebreak before and after
        # If prefix is a URL end tag, preserve it
        if '|URL end|' in prefix:
            return f'{prefix}\n|address start| {address} |address end|\n'
        # Otherwise, preserve the capital letters prefix
        return f'{prefix}\n|address start| {address} |address end|\n'
    
    # Replace addresses with formatted versions
    content = re.sub(address_pattern, format_address, content)
    
    # Clean up any duplicate markers and their associated linebreaks/spaces
    content = re.sub(r'\|address start\|\s+\|address start\|', '|address start|', content)
    content = re.sub(r'\|address end\|\s+\|address end\|', '|address end|', content)
    
    return content

def mark_urls(content):
    """
    Find URLs and add markers around them.
    URLs are defined as:
    - 2-40 characters with no spaces
    - Followed by .com/.biz/.org/.net/.edu
    - Optionally followed by:
        - A slash
        - Additional alphanumeric characters, dashes, dots, underscores, hash
        - Common file extensions (htm, html, aspx)
    Must be on their own line
    Special case: Any line containing a string from ALWAYS_URLS is treated as a URL
    """
    # Pattern matches:
    # ^ - start of line
    # \s* - optional whitespace
    # [^\s]{2,40} - 2-40 non-whitespace characters
    # \.(com|biz|org|net|edu) - literal ".com" or ".biz" or ".org" or ".net" or ".edu"
    # (?:/[\w\-._#]+(?:\.(?:htm|html|aspx))?)*/?  - optional path segments with optional trailing slash
    # \s*$ - optional whitespace and end of line
    url_pattern = r'^(\s*[^\s]{2,40}\.(com|biz|org|net|edu)(?:/[\w\-._#]+(?:\.(?:htm|html|aspx))?)*/?)\s*$'
    
    def format_url(match):
        url = match.group(1).strip()
        if url.endswith('/'):
            url = url.rstrip('/')
        return f'|URL start| {url} |URL end|\n'
    
    # Process line by line
    lines = content.split('\n')
    processed_lines = []
    for line in lines:
        # Special case for ALWAYS_URLS
        if any(always_url in line for always_url in ALWAYS_URLS):
            stripped_line = line.strip()
            processed_lines.append(f'|URL start| {stripped_line} |URL end|\n')
            continue
        # Normal URL processing
        processed_lines.append(re.sub(url_pattern, format_url, line))
    
    return '\n'.join(processed_lines)

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

def mark_hours(content):
    """
    Find hours and add markers around them.
    Hours are either:
    - Start with one of the HOURS_PATTERNS and end with 1-3 $ characters
    - Start with "(limited" or similar text and end with 1-3 $ characters
    - Just 1-3 $ characters on a line by themselves
    Must be at start of line
    """
    # Create pattern that matches any of the hours patterns
    hours_prefix = '|'.join(map(re.escape, HOURS_PATTERNS))
    
    # Pattern matches either:
    # 1. Hours pattern followed by text and $ signs
    # 2. Parenthetical text followed by $ signs
    # 3. Just $ signs
    hours_pattern = rf'^(\s*(?:(?:{hours_prefix})|(?:\([^)]+\))|(?:\s*)).*?[|]?\s*\${{1,3}})'
    
    def format_hours(match):
        hours = match.group(1).strip()
        return f'|hours start| {hours} |hours end|\n'
    
    # Process line by line
    lines = content.split('\n')
    processed_lines = [re.sub(hours_pattern, format_hours, line) for line in lines]
    return '\n'.join(processed_lines)

def mark_titles(content):
    """
    Find titles and add markers around them.
    Titles must:
    - Begin at start of line
    - Have 2+ consecutive capital letters or numbers
    - May include whitespace
    - End at end of line
    - Not be within other markers
    - Not contain full sentences (periods followed by spaces)
    """
    # First, split content into lines and process each line
    lines = content.split('\n')
    processed_lines = []
    
    for line in lines:
        # Skip if line is empty or contains any marker tags
        if not line.strip() or '|' in line:
            processed_lines.append(line)
            continue
            
        # Skip if line contains a period followed by space (likely a sentence)
        if re.search(r'\.\s', line):
            processed_lines.append(line)
            continue
            
        # Check for 2+ consecutive capitals/numbers at start
        consecutive_count = 0
        first_word = line.split()[0] if line.split() else ""
        for char in first_word:
            if char.isupper() or char.isdigit():
                consecutive_count += 1
                if consecutive_count >= 2:
                    # Found a title, add markers
                    processed_lines.append(f'|title start| {line.strip()} |title end|')
                    break
            elif char.isspace():
                continue
            else:
                consecutive_count = 0
        else:
            # No title found, keep original line
            processed_lines.append(line)
    
    return '\n'.join(processed_lines)

def find_all_titles(content):
    """Extract all titles from content using the title markers"""
    # Pattern matches anything between |title start| and |title end| markers
    title_pattern = r'\|title start\|\s*(.*?)\s*\|title end\|'
    
    # Find all matches
    titles = re.findall(title_pattern, content)
    return titles

def find_all_addresses(content):
    """Extract all addresses from content using the address markers"""
    # Pattern matches anything between |address start| and |address end| markers
    address_pattern = r'\|address start\|\s*(.*?)\s*\|address end\|'
    
    # Find all matches
    addresses = re.findall(address_pattern, content)
    return addresses

def find_all_urls(content):
    """Extract all URLs from content using the URL markers"""
    # Pattern matches anything between |URL start| and |URL end| markers
    url_pattern = r'\|URL start\|\s*(.*?)\s*\|URL end\|'
    
    # Find all matches
    urls = re.findall(url_pattern, content)
    return urls

def main():
    content = read_text_file(INPUT_FILE)
    
    if content:
        print(f"Successfully read file. Total length: {len(content)} characters")
                
        # Remove standalone state lines, normalize quotes, and process the content
        processed_content = normalize_quotes(content)
        processed_content = left_trim_lines(processed_content)
        processed_content = remove_standalone_states(processed_content)
        processed_content = left_trim_lines(processed_content)
        processed_content = mark_urls(processed_content)
        processed_content = left_trim_lines(processed_content)
        processed_content = mark_addresses(processed_content)
        processed_content = left_trim_lines(processed_content)
        processed_content = mark_phones(processed_content)
        processed_content = left_trim_lines(processed_content)
        processed_content = mark_hours(processed_content)
        processed_content = left_trim_lines(processed_content)
        processed_content = remove_question_mark_prefix(processed_content)
        processed_content = left_trim_lines(processed_content)
        processed_content = mark_titles(processed_content)
        processed_content = left_trim_lines(processed_content)
        
        os.makedirs(os.path.dirname(PROCESSED_OUTPUT_FILE), exist_ok=True)
        with open(PROCESSED_OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write(processed_content)
        print(f"\nProcessed content saved to {PROCESSED_OUTPUT_FILE}")
        
        # Find and save URLs
        urls = find_all_urls(processed_content)
        df_urls = pd.DataFrame(urls, columns=['url'])
        df_urls.to_csv(OUTPUT_URLS_FILE, index=False)
        print(f"\nFound {len(urls)} URLs.")

        # Find and save addresses
        addresses = find_all_addresses(processed_content)
        df_addresses = pd.DataFrame(addresses, columns=['address'])
        df_addresses.to_csv(OUTPUT_ADDRESSES_FILE, index=False)
        print(f"\nFound {len(addresses)} addresses.")

        # Find and save titles
        titles = find_all_titles(processed_content)
        df_titles = pd.DataFrame(titles, columns=['title'])
        df_titles.to_csv(OUTPUT_FILE, index=False)
        print(f"\nFound {len(titles)} titles.")
    else:
        print("Failed to read file")

if __name__ == "__main__":
    main() 