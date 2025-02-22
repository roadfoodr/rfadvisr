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
ALWAYS_TITLES = {'K. LAMAY\'S'}  # Strings that are always titles
NEVER_TITLES = {'B.J.'}  # Strings that should never be considered titles
ALWAYS_URLS = {'cornelldairybar.cfm'}  # Strings that should always be considered part of URLs
ALWAYS_ADDRESSES = {
    ('605 8th Ave. S. Nashville, Tennessee', '605 8th Ave. S. Nashville, TN'),  # (input pattern, replacement)
    ('Fulton and Dorrance St. at Kennedy Plz. Providence, RI', 'Fulton and Dorrance St. at Kennedy Plz. Providence, RI'),
    ('226 6th St. Augusta, GA', '226 6th St. Augusta, GA'),
    ('Windy Hill Rd. Smyrna, GA', 'Windy Hill Rd. Smyrna, GA'),
    ('Lane Packing Company 50 Lane Rd. Fort Valley, GA', 'Lane Packing Company 50 Lane Rd. Fort Valley, GA'),
    ('E2918 State Hwy. M-67 Trenary, MI', 'E2918 State Hwy. M-67 Trenary, MI'),
    ('3131 S. 27th St. Milwaukee, WI', '3131 S. 27th St. Milwaukee, WI'),
    ('N2030 Spring St. Stockholm, WI', 'N2030 Spring St. Stockholm, WI'),
    ('1516 W. 2nd Ave. Spokane, WA', '1516 W. 2nd Ave. Spokane, WA')
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
    'WISCONSIN', 'WYOMING', 'DISTRICT OF COLUMBIA'
}

# Add after other constants, before functions
STATE_ABBREVS = {
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC'
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
    - First number encountered (e.g., street number, route number) or the string "Route", but not if preceded by "US "
    - Through all subsequent text (including additional numbers, street names, etc.)
    - Until reaching ", XX" where XX is state abbreviation
    - Including any optional parenthetical content after the state
    
    Examples:
    "MIKE'S KITCHEN 170 Randall St. (at Tabor-Franchi VFW Post 239) Cranston, RI"
    "SUGAR'S 1799 State Rd. 68 Embudo, NM"
    "CHOPE'S 16145 S. Hwy. 28 La Mesa, NM"
    Will not match: "along US 17, SC"
    """
    # Updated pattern to allow directional prefixes (N., S., E., W.) before numbers
    # Updated pattern to include "Hwy. " as an additional prefix option
    address_pattern = r'((?:\|URL end\||[A-Z][^a-z\s]*[A-Z][^a-z\s]*[A-Z\s]*?))\s*(?<!US\s)((?:Route\s+|Exit\s+|Hwy\.\s+|(?:[NSEW]\.\s+)?(?:\d+))[^|]*?,\s(?:' + states + r')(?:\s*\([^)]+\))?)'


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
    # \.(com|biz|org|net|edu|coop) - literal ".com" or ".biz" or ".org" or ".net" or ".edu" or ".coop"
    # (?:/[\w\-._#]+(?:\.(?:htm|html|aspx))?)*/?  - optional path segments with optional trailing slash
    # \s*$ - optional whitespace and end of line
    url_pattern = r'^(\s*[^\s]{2,40}\.(com|biz|org|net|edu|coop)(?:/[\w\-._#]+(?:\.(?:htm|html|aspx))?)*/?)\s*$'
    
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
    Hours must:
    - Start with one of the HOURS_PATTERNS or "(limited" type text
    - End with 1-3 $ characters
    Must be at start of line
    """
    # Create pattern that matches any of the hours patterns
    hours_prefix = '|'.join(map(re.escape, HOURS_PATTERNS))
    
    # Pattern now requires either an hours prefix or parenthetical at start
    hours_pattern = rf'^(\s*(?:{hours_prefix}|(?:\([^)]+\))).*?\${{1,3}})'
    
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
    - Have 2+ consecutive capital letters or numbers (ignoring punctuation)
    - Must contain at least 1 capital letter
    - Must not contain any lowercase letters
    - May include whitespace
    - End at end of line
    - Not be within other markers
    - Not be in NEVER_TITLES set
    - Or be in ALWAYS_TITLES set
    """
    lines = content.split('\n')
    processed_lines = []
    
    for line in lines:
        # Check ALWAYS_TITLES first
        if line.strip() in ALWAYS_TITLES:
            processed_lines.append(f'|title start| {line.strip()} |title end|')
            continue
            
        # Skip if line is empty, contains markers, or is in NEVER_TITLES
        if not line.strip() or '|' in line or line.strip() in NEVER_TITLES:
            processed_lines.append(line)
            continue
            
        # Skip if line contains any lowercase letters or no capital letters
        if any(c.islower() for c in line) or not any(c.isupper() for c in line):
            processed_lines.append(line)
            continue
            
        # Check for 2+ consecutive capitals/numbers at start, ignoring punctuation
        consecutive_count = 0
        first_word = line.split()[0] if line.split() else ""
        for char in first_word:
            if char.isupper() or char.isdigit():
                consecutive_count += 1
                if consecutive_count >= 2:
                    # Found a title, add markers
                    processed_lines.append(f'|title start| {line.strip()} |title end|')
                    break
            elif char.isalpha():  # Only reset count on lowercase letters
                consecutive_count = 0
            # Ignore punctuation by not resetting the count
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

def mark_cost(content):
    """
    Find cost indicators and add markers around them.
    Costs must be either:
    1. Between "|" and "|hours end|" (1-3 $ characters)
    2. Standalone on a line (1-3 $ characters)
    """
    # First pattern (as before):
    # \| - literal pipe character
    # \s* - any amount of whitespace (including none)
    # (\${1,3}) - capture group for 1-3 $ characters
    # \s* - any amount of whitespace (including none)
    # \|hours end\| - literal "|hours end|" marker
    pattern1 = r'\|\s*(\${1,3})\s*\|hours end\|'
    replacement1 = r' |hours end|\n|cost start| \1 |cost end|\n'
    content = re.sub(pattern1, replacement1, content)
    
    # Second pattern:
    # ^ - start of line
    # \s* - any leading whitespace
    # (\${1,3}) - capture group for 1-3 $ characters
    # \s* - any trailing whitespace
    # $ - end of line
    pattern2 = r'^\s*(\${1,3})\s*$'
    replacement2 = r'|cost start| \1 |cost end|'
    
    # Process line by line for the second pattern
    lines = content.split('\n')
    processed_lines = [re.sub(pattern2, replacement2, line) for line in lines]
    return '\n'.join(processed_lines)

def mark_remaining_content(content):
    """
    Mark any text that isn't already enclosed in markers with |content start| and |content end|.
    Consecutive unmarked lines (including blank lines) are combined into a single content block.
    Only break content blocks when encountering lines with markers.
    """
    lines = content.split('\n')
    processed_lines = []
    current_content_block = []
    
    for line in lines:
        # If line contains markers, process any accumulated content block first
        if '|' in line:
            if current_content_block:
                # Join with newlines to preserve internal spacing
                combined_content = '\n'.join(current_content_block).strip()
                if combined_content:
                    processed_lines.append(f'|content start| {combined_content} |content end|')
                current_content_block = []
            processed_lines.append(line)
            continue
        
        # Accumulate unmarked content (including blank lines)
        current_content_block.append(line)
    
    # Process any remaining content block at the end
    if current_content_block:
        combined_content = '\n'.join(current_content_block).strip()
        if combined_content:
            processed_lines.append(f'|content start| {combined_content} |content end|')
    
    return '\n'.join(processed_lines)

def add_title_spacing(content):
    """
    Add exactly two newlines before each title marker, except for the first title.
    """
    # First, normalize any existing newlines before titles to a single newline
    content = re.sub(r'\n+\|title start\|', r'\n|title start|', content)
    
    # Split into first title and rest
    parts = content.split('|title start|', 1)
    if len(parts) == 1:  # No titles found
        return content
        
    # Handle first part and first title without extra newlines
    result = parts[0] + '|title start|'
    
    # Handle remaining content by adding two newlines before each title
    if len(parts) > 1:
        remaining = parts[1]
        remaining = re.sub(r'\n*\|title start\|', r'\n\n|title start|', remaining)
        result += remaining
    
    return result

def validate_record_sequence(content):
    """
    Validate that records follow the expected sequence:
    title -> [URL] -> address -> [phone] -> [hours] -> [cost] -> content -> (repeat)
    Square brackets indicate optional fields.
    """
    # Pattern to match any marked field
    field_pattern = r'\|(title|URL|address|phone|hours|cost|content) start\|\s*(.*?)\s*\|\1 end\|'
    
    # Expected sequence (None means field is optional)
    expected_sequence = ['title', 'URL', 'address', None, None, None, 'content']
    
    # Find all marked fields
    matches = re.finditer(field_pattern, content, re.DOTALL)
    current_sequence = []
    sequence_start_pos = 0
    
    for match in matches:
        field_type = match.group(1)
        field_content = match.group(2)
        
        # If we find a title and we're already in a sequence, validate the current sequence
        if field_type == 'title' and current_sequence:
            validate_current_sequence(current_sequence, sequence_start_pos)
            current_sequence = []
        
        # Start new sequence or add to existing
        if field_type == 'title':
            sequence_start_pos = match.start()
        current_sequence.append((field_type, field_content, match.start()))
    
    # Validate the last sequence
    if current_sequence:
        validate_current_sequence(current_sequence, sequence_start_pos)

def validate_current_sequence(sequence, start_pos):
    """Helper function to validate a single record sequence"""
    # Extract just the field types
    field_types = [field[0] for field in sequence]
    
    # Get the title content for error messages
    title_content = next((field[1] for field in sequence if field[0] == 'title'), 'Unknown Title')
    
    # Basic validation rules
    if field_types[0] != 'title':
        print(f"Error at position {start_pos}: Sequence doesn't start with title (Title: {title_content})")
        return
    
    if 'address' not in field_types[1:4]:  # Address should be in first few fields after title
        print(f"Error at position {start_pos}: Missing address after title (Title: {title_content})")
        return
    
    # Optional fields (URL, phone, hours, cost) can appear in any order after title
    # but before content
    
    if 'content' not in field_types:
        print(f"Error at position {start_pos}: Missing content field (Title: {title_content})")
        return
    
    # Check for duplicate fields
    for field_type in set(field_types):
        if field_types.count(field_type) > 1 and field_type != 'content':
            print(f"Error at position {start_pos}: Duplicate {field_type} field (Title: {title_content})")
            return

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
        processed_content = mark_cost(processed_content)
        processed_content = left_trim_lines(processed_content)
        processed_content = mark_remaining_content(processed_content)
        processed_content = left_trim_lines(processed_content)
        processed_content = add_title_spacing(processed_content)
        
        os.makedirs(os.path.dirname(PROCESSED_OUTPUT_FILE), exist_ok=True)
        with open(PROCESSED_OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write(processed_content)
        print(f"\nProcessed content saved to {PROCESSED_OUTPUT_FILE}")
        
        # Add validation after processing
        print("\nValidating record sequences...")
        validate_record_sequence(processed_content)
        
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