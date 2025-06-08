from typing import List, Optional
from langchain.schema import Document
from app.supabase_client import SupabaseManager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Supabase client
supabase_manager = SupabaseManager()

def store_search_scores(
    search_docs: List[Document],
    query: str,
    langsmith_run_id: Optional[str] = None,
    filter_applied: Optional[dict] = None,
    search_type: str = 'detailed'
) -> Optional[str]:
    """
    Store search results and scores in Supabase.
    
    Args:
        search_docs: List of Document objects from ChromaDB search
        query: The search query
        langsmith_run_id: Optional LangSmith run ID for tracing
        filter_applied: Optional dictionary of applied filters
        search_type: Type of search ('detailed' or 'summary')
        
    Returns:
        search_id: UUID of the created search record, or None if failed
    """
    try:
        # Convert search docs to the format expected by store_search_results
        results_for_storage = []
        for doc in search_docs:
            result_dict = {
                'restaurant_id': doc.metadata.get('ID', ''),
                'restaurant_name': doc.metadata.get('Restaurant', ''),
                'city': doc.metadata.get('City', ''),
                'state': doc.metadata.get('State', ''),
                'region': doc.metadata.get('Region', ''),
                'honor_roll': doc.metadata.get('Honor Roll', 'n'),
                'recommend': doc.metadata.get('Recommend', 'n'),
                'longitude': doc.metadata.get('long'),
                'latitude': doc.metadata.get('lat'),
                'sig_item': doc.metadata.get('sig_item', ''),
                'url': doc.metadata.get('URL', ''),
                'similarity_score': doc.metadata.get('similarity_score'),
                'metadata': doc.metadata
            }
            results_for_storage.append(result_dict)
        
        # Store results in Supabase
        search_id = supabase_manager.store_search_results(
            query=query,
            results=results_for_storage,
            langsmith_run_id=langsmith_run_id,
            filter_applied=filter_applied,
            search_type=search_type
        )
        
        if search_id:
            logger.info(f"Successfully stored search results with ID: {search_id}")
            return search_id
        else:
            logger.warning("Failed to store search results in Supabase")
            return None
            
    except Exception as e:
        logger.error(f"Error storing search results in Supabase: {str(e)}")
        return None 