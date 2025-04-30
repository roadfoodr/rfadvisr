from langchain.tools import Tool
from typing import Optional, Dict, List
from pydantic import BaseModel, Field
import re

# --- Input Schemas for Tools ---
class StateInput(BaseModel):
    """Input schema for the state extractor tool."""
    query: str = Field(description="The user query text potentially containing US state names or abbreviations")

class RegionInput(BaseModel):
    """Input schema for the region extractor tool."""
    query: str = Field(description="The user query text potentially containing US geographic region names")

# --- Stub Tool Implementations ---

# Comprehensive mapping of state names and abbreviations (lowercase) to 2-letter codes (uppercase)
# common two-letter words are excluded to avoid unintentional matches when not used as state names
US_STATES = {
    "alabama": "AL", "al": "AL",
    "alaska": "AK", "ak": "AK",
    "arizona": "AZ", "az": "AZ",
    "arkansas": "AR", "ar": "AR",
    "california": "CA", "ca": "CA",
    "colorado": "CO", "co": "CO",
    "connecticut": "CT", "ct": "CT",
    "delaware": "DE", "de": "DE",
    "florida": "FL", "fl": "FL",
    "georgia": "GA", "ga": "GA",
    "hawaii": "HI", 
    "idaho": "ID", "id": "ID",
    "illinois": "IL", "il": "IL",
    "indiana": "IN",
    "iowa": "IA", "ia": "IA",
    "kansas": "KS", "ks": "KS",
    "kentucky": "KY", "ky": "KY",
    "louisiana": "LA", "la": "LA",
    "maine": "ME",
    "maryland": "MD", "md": "MD",
    "massachusetts": "MA", "ma": "MA",
    "michigan": "MI", "mi": "MI",
    "minnesota": "MN", "mn": "MN",
    "mississippi": "MS", "ms": "MS",
    "missouri": "MO", "mo": "MO",
    "montana": "MT", "mt": "MT",
    "nebraska": "NE", "ne": "NE",
    "nevada": "NV", "nv": "NV",
    "new hampshire": "NH", "nh": "NH",
    "new jersey": "NJ", "nj": "NJ",
    "new mexico": "NM", "nm": "NM",
    "new york": "NY", "ny": "NY",
    "north carolina": "NC", "nc": "NC",
    "north dakota": "ND", "nd": "ND",
    "ohio": "OH", "oh": "OH",
    "oklahoma": "OK", "ok": "OK",
    "oregon": "OR",
    "pennsylvania": "PA", "pa": "PA",
    "rhode island": "RI", "ri": "RI",
    "south carolina": "SC", "sc": "SC",
    "south dakota": "SD", "sd": "SD",
    "tennessee": "TN", "tn": "TN",
    "texas": "TX", "tx": "TX",
    "utah": "UT", "ut": "UT",
    "vermont": "VT", "vt": "VT",
    "virginia": "VA", "va": "VA",
    "washington": "WA", "wa": "WA",
    "west virginia": "WV", "wv": "WV",
    "wisconsin": "WI", "wi": "WI",
    "wyoming": "WY", "wy": "WY"
}

def extract_state_filter(query: str) -> Optional[Dict]:
    """
    Extracts state filters from a query by looking for US state names or abbreviations.
    Returns a ChromaDB filter condition dictionary if states are found, otherwise None.
    """
    print(f"  >> [extract_state_filter] Received query: '{query}'")
    query_lower = query.lower()
    # Use a set to store found state codes to avoid duplicates
    found_states_codes = set()

    # Check for matches in the query using the comprehensive mapping
    for key in US_STATES.keys():
        # Use word boundaries to avoid partial matches within words (e.g., 'or' in 'oregon')
        # Handle multi-word keys (like "new jersey") and single-word keys/abbreviations
        pattern = r'\b' + key + r'\b'
        if re.search(pattern, query_lower):
             state_code = US_STATES[key]
             print(f"  >> [extract_state_filter] Match found for '{key}', adding code: {state_code}")
             found_states_codes.add(state_code)

    print(f"  >> [extract_state_filter] Final found state codes (set): {found_states_codes}")
    if found_states_codes:
        # Convert set to list for JSON serialization in the filter
        found_states_list = list(found_states_codes)
        print(f"  >> [extract_state_filter] Found States List: {found_states_list}")
        return {"State": {"$in": found_states_list}}
    else:
        print("  >> [extract_state_filter] No states found, returning None.")
        return None

def extract_region_filter(query: str) -> Optional[Dict]:
    """
    Stub function to extract region filters.
    Looks for specific region names in the query.
    Returns a ChromaDB filter condition dictionary if regions are found, otherwise None.
    """
    query_lower = query.lower()
    found_regions = []

    # Simple keyword matching for stub implementation
    if "new england" in query_lower:
        found_regions.append("New England")
    if "west coast" in query_lower:
         found_regions.append("West Coast")
    if "mid-atlantic" in query_lower or "mid atlantic" in query_lower:
        found_regions.append("Mid-Atlantic")
    if "mid-south" in query_lower: # Be careful, "south" is broad
        found_regions.append("Mid-South")
    # Add more regions...

    if found_regions:
        print(f"  > (Stub Tool) Found Regions: {found_regions}")
        return {"Region": {"$in": found_regions}}
    else:
        return None

# --- Define LangChain Tools ---

state_tool = Tool.from_function(
    name="state_extractor",
    func=extract_state_filter,
    description="Use this tool *only* if the user query explicitly mentions one or more specific US state names (e.g., 'California') or 2-letter abbreviations (e.g., 'CA'). Do not use if no state is mentioned. Returns a filter condition for the 'State' metadata field.",
    args_schema=StateInput
)

region_tool = Tool.from_function(
    name="region_extractor",
    func=extract_region_filter,
    description="Use this tool *only* if the user query explicitly mentions a specific geographic region of the US (e.g., 'Northeast', 'West Coast', 'South', 'Midwest', 'Mid-Atlantic', 'New England'). Do not use if no region is mentioned. Returns a filter condition for the 'Region' metadata field.",
    args_schema=RegionInput
)

# List of tools to be imported by the main application
filter_tools: List[Tool] = [state_tool, region_tool]

# Potential future tools:
# cuisine_tool = Tool(...)
# price_range_tool = Tool(...)
# feature_tool = Tool(...) # e.g., "waterfront dining", "historic" 