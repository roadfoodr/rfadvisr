from langchain.tools import Tool
from langchain.tools import StructuredTool
from typing import Optional, Dict, List
from pydantic import BaseModel, Field
import re
from langsmith import traceable

# --- Input Schemas for Tools ---
class StateInput(BaseModel):
    """Input schema for the state extractor tool."""
    query: str = Field(description="The user query text potentially containing US state names or abbreviations")

class RegionInput(BaseModel):
    """Input schema for the region extractor tool."""
    query: str = Field(description="The user query text potentially containing US geographic region names")

class CityInput(BaseModel):
    """Input schema for the city extractor tool."""
    query: str = Field(description="The user query text potentially containing US city names")
    cities: List[str] = Field(description="List of city names identified in the query by the LLM")

# --- Stub Tool Implementations ---

# Comprehensive mapping of state names and abbreviations (lowercase) to 2-letter codes (uppercase)
# common two-letter words are excluded to avoid unintentional matches when not used as state names
US_STATES = {
    "alabama": "AL", "al": "AL",
    "bama": "AL",
    "alaska": "AK", "ak": "AK",
    "arizona": "AZ", "az": "AZ",
    "arkansas": "AR", "ar": "AR",
    "arkie": "AR",
    "california": "CA", "ca": "CA",
    "cali": "CA",
    "colorado": "CO", "co": "CO",
    "connecticut": "CT", "ct": "CT",
    "nutmeg state": "CT",
    "delaware": "DE", "de": "DE",
    "florida": "FL", "fl": "FL",
    "georgia": "GA", "ga": "GA",
    "hawaii": "HI", 
    "idaho": "ID", "id": "ID",
    "illinois": "IL", "il": "IL",
    "hoosier": "IN",
    "indiana": "IN",
    "iowa": "IA", "ia": "IA",
    "kansas": "KS", "ks": "KS",
    "kentucky": "KY", "ky": "KY",
    "louisiana": "LA", "la": "LA",
    "maine": "ME",
    "maryland": "MD", "md": "MD",
    "massachusetts": "MA", "ma": "MA",
    "mass": "MA",
    "michigan": "MI", "mi": "MI",
    "minnesota": "MN", "mn": "MN",
    "mississippi": "MS", "ms": "MS",
    "missouri": "MO", "mo": "MO",
    "montana": "MT", "mt": "MT",
    "show me state": "MO",
    "nebraska": "NE", "ne": "NE",
    "nevada": "NV", "nv": "NV",
    "new hampshire": "NH", "nh": "NH",
    "new jersey": "NJ", "nj": "NJ",
    "jersey": "NJ",
    "new mexico": "NM", "nm": "NM",
    "new york": "NY", "ny": "NY", "ny state": "NY", "new york state": "NY",
    "north carolina": "NC", "nc": "NC",
    "north dakota": "ND", "nd": "ND",
    "ohio": "OH", "oh": "OH",
    "oklahoma": "OK", "ok": "OK",
    "okie": "OK",
    "oregon": "OR",
    "pennsylvania": "PA", "pa": "PA",
    "rhode island": "RI", "ri": "RI",
    "south carolina": "SC", "sc": "SC",
    "south dakota": "SD", "sd": "SD",
    "tennessee": "TN", "tn": "TN",
    "texas": "TX", "tx": "TX",
    "lone star": "TX", "lone star state": "TX",
    "utah": "UT", "ut": "UT",
    "vermont": "VT", "vt": "VT",
    "virginia": "VA", "va": "VA",
    "washington": "WA", "wa": "WA",
    "west virginia": "WV", "wv": "WV",
    "wisconsin": "WI", "wi": "WI",
    "cheesehead": "WI",
    "wyoming": "WY", "wy": "WY"
}

@traceable(run_type="tool", name="State Extractor Tool")
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

# Mapping of region variations/aliases (lowercase) to canonical region names
US_REGIONS = {
    # Deep South
    "deep south": "Deep South",
    # Great Plains
    "great plains": "Great Plains",
    # Mid-Atlantic
    "mid-atlantic": "Mid-Atlantic",
    "mid atlantic": "Mid-Atlantic",
    # Mid-South
    "mid-south": "Mid-South",
    "mid south": "Mid-South",
    # Midwest
    "midwest": "Midwest",
    "mid-west": "Midwest",
    "Mid-west": "Midwest",
    "Mid-West": "Midwest",
    # New England
    "new england": "New England",
    "northeast": "New England", # Alias
    # Southwest
    "southwest": "Southwest",
    # West Coast
    "west coast": "West Coast",
    "pacific coast": "West Coast" # Alias
    # Note: Broad terms like "South" are intentionally omitted unless they map to a specific defined region (like Mid-South).
    # You could add more specific sub-regions or states mapping to regions if needed.
}

# Mapping of broader region terms (lowercase) to lists of state codes
# Used when no specific region from US_REGIONS is matched.
BROAD_REGIONS_TO_STATES = {
    "east coast": ["ME", "NH", "MA", "RI", "CT", "NY", "NJ", "PA", "DE", "MD", "VA", "NC", "SC", "GA", "FL"],
    "eastern seaboard": ["ME", "NH", "MA", "RI", "CT", "NY", "NJ", "PA", "DE", "MD", "VA", "NC", "SC", "GA", "FL"], # Alias
    "the south": ["VA", "WV", "KY", "NC", "SC", "GA", "FL", "AL", "MS", "TN", "AR", "LA", "TX", "OK"], # Southeast + South Central commonly
    "southern states": ["VA", "WV", "KY", "NC", "SC", "GA", "FL", "AL", "MS", "TN", "AR", "LA", "TX", "OK"], # Alias
    "southeast": ["AL", "GA", "FL", "SC", "KY", "TN", "MS", "LA"],
    "northwest": ["WA", "OR", "ID"],
    "pacific northwest": ["WA", "OR"],
    # Add other broad regions like "west", "mountain west" etc. if needed
}

@traceable(run_type="tool", name="Region Extractor Tool")
def extract_region_filter(query: str) -> Optional[Dict]:
    """
    Extracts filters from a query based on US geographic regions.
    - If specific regions (e.g., "New England", "Mid-Atlantic") or their aliases are found,
      returns a filter for the 'Region' metadata field.
    - If broader regions (e.g., "East Coast", "South") are found *and* no specific regions are,
      returns a filter for the 'State' metadata field based on the states in that broad region.
    - Returns None if no recognized regions are found.
    """
    print(f"  >> [extract_region_filter] Received query: '{query}'")
    query_lower = query.lower()
    found_regions_canonical = set()
    found_states_from_broad_regions = set()

    # 1. Check for specific regions (mapped to Region field)
    for key in US_REGIONS.keys():
        pattern = r'\b' + re.escape(key) + r'\b'
        if re.search(pattern, query_lower):
            region_canonical = US_REGIONS[key]
            print(f"  >> [extract_region_filter] Match found for specific region term '{key}', adding canonical region: {region_canonical}")
            found_regions_canonical.add(region_canonical)

    # 2. If specific regions found, prioritize and return Region filter
    if found_regions_canonical:
        found_regions_list = list(found_regions_canonical)
        print(f"  >> [extract_region_filter] Prioritizing specific regions. Found Regions List (canonical): {found_regions_list}")
        return {"Region": {"$in": found_regions_list}}

    # 3. If no specific regions, check for broad regions (mapped to State field)
    print("  >> [extract_region_filter] No specific regions found. Checking for broad regions...")
    for key in BROAD_REGIONS_TO_STATES.keys():
        pattern = r'\b' + re.escape(key) + r'\b'
        if re.search(pattern, query_lower):
            states = BROAD_REGIONS_TO_STATES[key]
            print(f"  >> [extract_region_filter] Match found for broad region term '{key}', adding states: {states}")
            found_states_from_broad_regions.update(states) # Use update for set union

    # 4. If broad region states found, return State filter
    if found_states_from_broad_regions:
        found_states_list = sorted(list(found_states_from_broad_regions)) # Sort for consistency
        print(f"  >> [extract_region_filter] Found states from broad regions: {found_states_list}")
        return {"State": {"$in": found_states_list}}
    else:
        # 5. No regions (specific or broad) found
        print("  >> [extract_region_filter] No specific or broad regions found, returning None.")
        return None

@traceable(run_type="tool", name="City Extractor Tool")
def extract_city_filter(query: str, cities: List[str]) -> Optional[Dict]:
    """
    Creates a ChromaDB filter condition for the identified cities.
    Returns a ChromaDB filter condition dictionary if cities are provided, otherwise None.
    """
    print(f"  >> [extract_city_filter] Received query: '{query}'")
    print(f"  >> [extract_city_filter] Received cities: {cities}")
    
    if cities:
        # Remove duplicates while preserving order
        found_cities = list(dict.fromkeys(cities))
        print(f"  >> [extract_city_filter] Using cities: {found_cities}")
        return {"City": {"$in": found_cities}}
    else:
        print("  >> [extract_city_filter] No cities provided, returning None.")
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

city_tool = StructuredTool.from_function(
    name="city_extractor",
    func=extract_city_filter,
    description="Use this tool *only* if the user query explicitly mentions one or more specific US city names (e.g., 'New York', 'Los Angeles'). The LLM should identify the city names and pass them as a list. Do not use if no city is mentioned. Returns a filter condition for the 'City' metadata field.",
    args_schema=CityInput
)

# List of tools to be imported by the main application
filter_tools: List[Tool] = [state_tool, region_tool, city_tool]

# Potential future tools:
# cuisine_tool = Tool(...)
# price_range_tool = Tool(...)
# feature_tool = Tool(...) # e.g., "waterfront dining", "historic" 