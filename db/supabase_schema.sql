-- Supabase Schema for Roadfood Advisor ChromaDB Scores
-- Run this script in your Supabase SQL editor to create the required tables

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Drop dependent objects first
DROP MATERIALIZED VIEW IF EXISTS search_analytics;
DROP VIEW IF EXISTS search_results_with_scores;
DROP FUNCTION IF EXISTS get_top_restaurants(INTEGER);

-- Drop existing tables if they exist
DROP TABLE IF EXISTS result_scores;
DROP TABLE IF EXISTS search_results;

-- Main search results table
CREATE TABLE IF NOT EXISTS search_results (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    search_id UUID NOT NULL UNIQUE,
    query TEXT NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    user_id TEXT,
    langsmith_run_id TEXT,
    filter_applied JSONB,
    num_results INTEGER,
    search_type TEXT CHECK (search_type IN ('detailed', 'summary')),
    feedback_success TEXT CHECK (feedback_success IN ('y', 'n')),
    feedback_reason TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- Individual result scores table
CREATE TABLE IF NOT EXISTS result_scores (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    search_id UUID NOT NULL REFERENCES search_results(search_id) ON DELETE CASCADE,
    restaurant_id TEXT NOT NULL,
    restaurant_name TEXT NOT NULL,
    city TEXT,
    state TEXT,
    region TEXT,
    honor_roll TEXT CHECK (honor_roll IN ('y', 'n')) DEFAULT 'n',
    recommend TEXT CHECK (recommend IN ('y', 'n')) DEFAULT 'n',
    relevant TEXT CHECK (relevant IN ('y', 'n')),  -- NULL by default, must be explicitly set by reviewer
    adjusted_score FLOAT,
    longitude FLOAT,
    latitude FLOAT,
    sig_item TEXT,
    similarity_score FLOAT NOT NULL CHECK (similarity_score >= 0),
    rank INTEGER NOT NULL CHECK (rank > 0),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_search_results_timestamp ON search_results(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_search_results_query ON search_results(query);
CREATE INDEX IF NOT EXISTS idx_search_results_langsmith_run_id ON search_results(langsmith_run_id);
CREATE INDEX IF NOT EXISTS idx_search_results_feedback_success ON search_results(feedback_success);
CREATE INDEX IF NOT EXISTS idx_result_scores_search_id ON result_scores(search_id);
CREATE INDEX IF NOT EXISTS idx_result_scores_restaurant_id ON result_scores(restaurant_id);
CREATE INDEX IF NOT EXISTS idx_result_scores_similarity_score ON result_scores(similarity_score);
CREATE INDEX IF NOT EXISTS idx_result_scores_adjusted_score ON result_scores(adjusted_score);
CREATE INDEX IF NOT EXISTS idx_result_scores_honor_roll ON result_scores(honor_roll);
CREATE INDEX IF NOT EXISTS idx_result_scores_recommend ON result_scores(recommend);
CREATE INDEX IF NOT EXISTS idx_result_scores_region ON result_scores(region);
CREATE INDEX IF NOT EXISTS idx_result_scores_relevant ON result_scores(relevant);

-- Create a view for easy querying of search results with scores
CREATE OR REPLACE VIEW search_results_with_scores
WITH (security_invoker = on) AS
SELECT 
    sr.id,
    sr.search_id,
    sr.query,
    sr.timestamp,
    sr.user_id,
    sr.langsmith_run_id,
    sr.filter_applied,
    sr.num_results,
    sr.search_type,
    sr.feedback_success,
    sr.feedback_reason,
    rs.restaurant_id,
    rs.restaurant_name,
    rs.city,
    rs.state,
    rs.region,
    rs.honor_roll,
    rs.recommend,
    rs.relevant,
    rs.adjusted_score,
    rs.longitude,
    rs.latitude,
    rs.sig_item,
    rs.similarity_score,
    rs.rank
FROM search_results sr
LEFT JOIN result_scores rs ON sr.search_id = rs.search_id
ORDER BY sr.timestamp DESC, rs.rank ASC;

-- Create a materialized view for analytics (refresh periodically)
CREATE MATERIALIZED VIEW IF NOT EXISTS search_analytics AS
SELECT 
    DATE_TRUNC('day', timestamp) as search_date,
    COUNT(DISTINCT search_id) as total_searches,
    COUNT(DISTINCT query) as unique_queries,
    AVG(num_results) as avg_results_requested,
    COUNT(CASE WHEN search_type = 'summary' THEN 1 END) as summary_searches,
    COUNT(CASE WHEN search_type = 'detailed' THEN 1 END) as detailed_searches,
    COUNT(CASE WHEN feedback_success = 'y' THEN 1 END) as successful_feedback,
    COUNT(CASE WHEN feedback_success = 'n' THEN 1 END) as unsuccessful_feedback
FROM search_results
GROUP BY DATE_TRUNC('day', timestamp);

-- Create index on materialized view
CREATE INDEX IF NOT EXISTS idx_search_analytics_date ON search_analytics(search_date);

-- Function to get top restaurants by search frequency
CREATE OR REPLACE FUNCTION get_top_restaurants(limit_count INTEGER DEFAULT 10)
RETURNS TABLE (
    restaurant_id TEXT,
    restaurant_name TEXT,
    city TEXT,
    state TEXT,
    search_count BIGINT,
    avg_similarity_score FLOAT,
    avg_rank FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        rs.restaurant_id,
        rs.restaurant_name,
        rs.city,
        rs.state,
        COUNT(DISTINCT sr.search_id) as search_count,
        AVG(rs.similarity_score) as avg_similarity_score,
        AVG(rs.rank) as avg_rank
    FROM result_scores rs
    JOIN search_results sr ON rs.search_id = sr.search_id
    GROUP BY rs.restaurant_id, rs.restaurant_name, rs.city, rs.state
    ORDER BY search_count DESC
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- Row Level Security (RLS) policies
-- Enable RLS on tables
ALTER TABLE search_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE result_scores ENABLE ROW LEVEL SECURITY;

-- Create policies (adjust based on your authentication strategy)
-- For now, we'll create a simple policy that allows all authenticated users to insert
-- and allows reading all data (adjust as needed)

-- Policy for inserting search results (authenticated users only)
CREATE POLICY "Enable insert for authenticated users only" ON search_results
    FOR INSERT
    TO authenticated
    WITH CHECK (true);

-- Policy for reading search results (public read for analytics)
DROP POLICY IF EXISTS "Enable read access for all users" ON search_results;
CREATE POLICY "Enable read access for all users" ON search_results
    FOR SELECT
    TO public
    USING (true);

-- Similar policies for result_scores
CREATE POLICY "Enable insert for authenticated users only" ON result_scores
    FOR INSERT
    TO authenticated
    WITH CHECK (true);

DROP POLICY IF EXISTS "Enable read access for all users" ON result_scores;
CREATE POLICY "Enable read access for all users" ON result_scores
    FOR SELECT
    TO public
    USING (true);

-- Grant necessary permissions to the anon and authenticated roles
GRANT SELECT ON search_results TO anon;
GRANT SELECT ON result_scores TO anon;
GRANT SELECT ON search_results_with_scores TO anon;
GRANT SELECT ON search_analytics TO anon;

GRANT INSERT ON search_results TO authenticated;
GRANT INSERT ON result_scores TO authenticated;

-- Comments for documentation
COMMENT ON TABLE search_results IS 'Stores metadata about each search performed in the Roadfood Advisor app';
COMMENT ON TABLE result_scores IS 'Stores individual restaurant results and their similarity scores for each search';
COMMENT ON VIEW search_results_with_scores IS 'Combined view of search results and their scores for easy querying';
COMMENT ON MATERIALIZED VIEW search_analytics IS 'Aggregated search analytics by day - refresh periodically';
COMMENT ON FUNCTION get_top_restaurants IS 'Returns the most frequently searched restaurants with their average scores'; 