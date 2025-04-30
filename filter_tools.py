from langchain.tools import Tool
from typing import Optional, Dict, List
from pydantic import BaseModel, Field

# --- Input Schemas for Tools ---
class StateInput(BaseModel):
    """Input schema for the state extractor tool."""
    query: str = Field(description="The user query text potentially containing US state names or abbreviations")

class RegionInput(BaseModel):
    """Input schema for the region extractor tool."""
    query: str = Field(description="The user query text potentially containing US geographic region names")

# --- Stub Tool Implementations ---

def extract_state_filter(query: str) -> Optional[Dict]:
    """
    Stub function to extract state filters.
    Looks for specific state names or abbreviations in the query.
    Returns a ChromaDB filter condition dictionary if states are found, otherwise None.
    """
    print(f"  >> [extract_state_filter] Received query: '{query}'")
    query_lower = query.lower()
    print(f"  >> [extract_state_filter] Query lowercased: '{query_lower}'")
    found_states = []

    # Simple keyword matching for stub implementation
    print("  >> [extract_state_filter] Checking for 'new jersey'...")
    if "new jersey" in query_lower or " nj " in query_lower or query_lower.endswith(" nj"):
        print("  >> [extract_state_filter] Match found for NJ!")
        found_states.append("NJ")

    print("  >> [extract_state_filter] Checking for 'california'...")
    if "california" in query_lower or " ca " in query_lower or query_lower.endswith(" ca"):
        print("  >> [extract_state_filter] Match found for CA!")
        found_states.append("CA")

    print("  >> [extract_state_filter] Checking for 'texas'...")
    if "texas" in query_lower or " tx " in query_lower or query_lower.endswith(" tx"):
        print("  >> [extract_state_filter] Match found for TX!")
        found_states.append("TX")

    print("  >> [extract_state_filter] Checking for 'arkansas'...")
    if "arkansas" in query_lower or " ar " in query_lower or query_lower.endswith(" ar"):
        print("  >> [extract_state_filter] Match found for AR!")
        found_states.append("AR")
    # Add more states or use regex later...

    print(f"  >> [extract_state_filter] Final found_states list: {found_states}")
    if found_states:
        print(f"  >> [extract_state_filter] Found States (original print): {found_states}")
        return {"State": {"$in": found_states}}
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