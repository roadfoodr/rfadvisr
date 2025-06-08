import os
from typing import Optional, Dict, List
from supabase import create_client, Client
import yaml
import logging
from datetime import datetime
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupabaseManager:
    def __init__(self):
        """Initialize the Supabase manager with connection handling."""
        self.client = self._initialize_client()
        
    def _initialize_client(self) -> Optional[Client]:
        """Initialize Supabase client with credentials from environment or yaml file."""
        try:
            # Try to get credentials from environment variables first
            supabase_url = os.environ.get("SUPABASE_URL")
            supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
            
            # If not in environment, try credentials.yml
            if not supabase_url or not supabase_key:
                logger.info("Supabase credentials not found in environment, checking credentials.yml")
                try:
                    credentials = yaml.safe_load(open('credentials.yml'))
                    supabase_config = credentials.get('supabase-rfadvisr_result_scores', {})
                    supabase_url = supabase_config.get('url')
                    supabase_key = supabase_config.get('service_role_key')
                except Exception as e:
                    logger.error(f"Error loading credentials.yml: {str(e)}")
                    return None
            
            if not supabase_url or not supabase_key:
                logger.error("Supabase credentials not found in environment or credentials.yml")
                return None
                
            # Initialize the client
            client = create_client(supabase_url, supabase_key)
            logger.info("Supabase client initialized successfully")
            return client
            
        except Exception as e:
            logger.error(f"Error initializing Supabase client: {str(e)}")
            return None
    
    def store_search_results(self, query: str, results: List[Dict], user_id: Optional[str] = None, 
                           langsmith_run_id: Optional[str] = None, filter_applied: Optional[Dict] = None,
                           search_type: str = 'detailed') -> Optional[str]:
        """
        Store search results and scores in Supabase.
        
        Args:
            query: The search query
            results: List of result dictionaries containing restaurant info and scores
            user_id: Optional user identifier
            langsmith_run_id: Optional LangSmith run ID for tracing
            filter_applied: Optional dictionary of applied filters
            search_type: Type of search ('detailed' or 'summary')
            
        Returns:
            search_id: UUID of the created search record, or None if failed
        """
        if not self.client:
            logger.error("Cannot store results: Supabase client not initialized")
            return None
            
        try:
            # Generate a unique search ID
            search_id = str(uuid.uuid4())
            
            # Prepare search results record
            search_data = {
                'search_id': search_id,
                'query': query,
                'timestamp': datetime.utcnow().isoformat(),
                'user_id': user_id,
                'langsmith_run_id': langsmith_run_id,
                'filter_applied': filter_applied,
                'num_results': len(results),
                'search_type': search_type
            }
            
            # Insert search results
            self.client.table('search_results').insert(search_data).execute()
            
            # Prepare and insert result scores
            score_records = []
            for rank, result in enumerate(results, 1):
                score_record = {
                    'search_id': search_id,
                    'restaurant_id': result.get('restaurant_id'),
                    'restaurant_name': result.get('restaurant_name'),
                    'city': result.get('city'),
                    'state': result.get('state'),
                    'region': result.get('region'),
                    'honor_roll': result.get('honor_roll', 'n'),
                    'recommend': result.get('recommend', 'n'),
                    'url': result.get('url', ''),
                    'longitude': result.get('longitude'),
                    'latitude': result.get('latitude'),
                    'sig_item': result.get('sig_item'),
                    'similarity_score': result.get('similarity_score'),
                    'rank': rank,
                    'metadata': result.get('metadata', {})
                }
                score_records.append(score_record)
            
            # Batch insert score records
            if score_records:
                self.client.table('result_scores').insert(score_records).execute()
            
            logger.info(f"Successfully stored search results with ID: {search_id}")
            return search_id
            
        except Exception as e:
            logger.error(f"Error storing search results: {str(e)}")
            return None
    
    def get_search_results(self, search_id: str) -> Optional[Dict]:
        """Retrieve search results and scores for a given search ID."""
        if not self.client:
            logger.error("Cannot retrieve results: Supabase client not initialized")
            return None
            
        try:
            # Get search metadata
            search_data = self.client.table('search_results')\
                .select('*')\
                .eq('search_id', search_id)\
                .execute()
                
            if not search_data.data:
                logger.warning(f"No search results found for ID: {search_id}")
                return None
                
            # Get result scores
            scores_data = self.client.table('result_scores')\
                .select('*')\
                .eq('search_id', search_id)\
                .order('rank')\
                .execute()
                
            return {
                'search': search_data.data[0],
                'scores': scores_data.data
            }
            
        except Exception as e:
            logger.error(f"Error retrieving search results: {str(e)}")
            return None 