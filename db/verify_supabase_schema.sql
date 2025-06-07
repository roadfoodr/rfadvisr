-- Verification queries for Supabase schema
-- Run this to get a consolidated view of the schema

WITH table_info AS (
    -- Get table structure
    SELECT 
        table_name,
        'column' as object_type,
        column_name as object_name,
        data_type || 
        CASE 
            WHEN is_nullable = 'NO' THEN ' NOT NULL'
            ELSE ''
        END ||
        CASE 
            WHEN column_default IS NOT NULL THEN ' DEFAULT ' || column_default
            ELSE ''
        END as definition
    FROM information_schema.columns 
    WHERE table_name IN ('search_results', 'result_scores')
),
index_info AS (
    -- Get indexes
    SELECT 
        tablename as table_name,
        'index' as object_type,
        indexname as object_name,
        indexdef as definition
    FROM pg_indexes 
    WHERE tablename IN ('search_results', 'result_scores')
),
constraint_info AS (
    -- Get constraints
    SELECT 
        tc.table_name,
        'constraint' as object_type,
        tc.constraint_name as object_name,
        tc.constraint_type || 
        CASE 
            WHEN cc.check_clause IS NOT NULL THEN ': ' || cc.check_clause
            ELSE ''
        END as definition
    FROM information_schema.table_constraints tc
    LEFT JOIN information_schema.check_constraints cc 
        ON tc.constraint_name = cc.constraint_name
    WHERE tc.table_name IN ('search_results', 'result_scores')
),
view_info AS (
    -- Get regular views
    SELECT 
        table_name,
        'view' as object_type,
        table_name as object_name,
        view_definition as definition
    FROM information_schema.views 
    WHERE table_name = 'search_results_with_scores'
    UNION ALL
    -- Get materialized views
    SELECT 
        'public' as table_name,
        'materialized view' as object_type,
        matviewname as object_name,
        'Has indexes: ' || hasindexes || ', Is populated: ' || ispopulated as definition
    FROM pg_matviews 
    WHERE matviewname = 'search_analytics'
)
-- Combine all results
SELECT 
    table_name,
    object_type,
    object_name,
    definition
FROM (
    SELECT * FROM table_info
    UNION ALL
    SELECT * FROM index_info
    UNION ALL
    SELECT * FROM constraint_info
    UNION ALL
    SELECT * FROM view_info
) all_objects
ORDER BY 
    table_name,
    CASE object_type
        WHEN 'column' THEN 1
        WHEN 'constraint' THEN 2
        WHEN 'index' THEN 3
        WHEN 'view' THEN 4
        WHEN 'materialized view' THEN 5
    END,
    object_name; 