import pandas as pd
import os
import re

# Get the current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (project root)
project_root = os.path.dirname(current_dir)

# Define file paths
excel_file_path = os.path.join(project_root, 'data', 'Roadfood_MDP_010225_GEO.xlsx')
csv_file_path = os.path.join(project_root, 'data', 'Roadfood_10th_edition_processed.csv')

print("=== Reading Excel File ===")
# Check if the Excel file exists
if os.path.exists(excel_file_path):
    print(f"Reading Excel file: {excel_file_path}")
    
    # Read the Excel file into a pandas DataFrame
    df_excel = pd.read_excel(excel_file_path)
    
    # Print the column names
    print("\nColumn names in the Excel DataFrame:")
    for i, col in enumerate(df_excel.columns):
        print(f"{i+1}. {col}")
    
    # Print the shape of the DataFrame
    print(f"\nExcel DataFrame shape: {df_excel.shape} (rows, columns)")
    
    # Print the first few rows to get a glimpse of the data
    print("\nFirst 3 rows of the Excel DataFrame:")
    print(df_excel.head(3))
else:
    print(f"\nError: Excel file not found at {excel_file_path}")

print("\n\n=== Reading CSV File ===")
# Check if the CSV file exists
if os.path.exists(csv_file_path):
    print(f"Reading CSV file: {csv_file_path}")
    
    # Read the CSV file into a pandas DataFrame
    df_csv = pd.read_csv(csv_file_path)
    
    # Print the column names
    print("\nColumn names in the CSV DataFrame:")
    for i, col in enumerate(df_csv.columns):
        print(f"{i+1}. {col}")
    
    # Print the shape of the DataFrame
    print(f"\nCSV DataFrame shape: {df_csv.shape} (rows, columns)")
    
    # Print the first few rows to get a glimpse of the data
    print("\nFirst 3 rows of the CSV DataFrame:")
    print(df_csv.head(3))
else:
    print(f"\nError: CSV file not found at {csv_file_path}")

# If both files were loaded successfully, process and merge
if 'df_excel' in locals() and 'df_csv' in locals():
    print("\n\n=== Processing and Merging Data ===")
    
    # Check if 'address' column exists in the CSV DataFrame
    if 'address' in df_csv.columns:
        print("Extracting state from address in CSV file...")
        
        # Function to extract state from address
        def extract_state(address):
            if pd.isna(address):
                return None
            
            # Pattern to match state abbreviation (2 uppercase letters at the end or followed by zip code)
            state_pattern = r'([A-Z]{2})(?:$|\s+\d{5})'
            
            # Alternative pattern to match state before a comma
            alt_pattern = r',\s*([A-Z]{2})(?:$|\s)'
            
            # Try the first pattern
            match = re.search(state_pattern, address)
            if match:
                return match.group(1)
            
            # Try the alternative pattern
            match = re.search(alt_pattern, address)
            if match:
                return match.group(1)
            
            return None
        
        # Apply the function to extract state
        df_csv['State'] = df_csv['address'].apply(extract_state)
        
        # Count how many states were successfully extracted
        state_count = df_csv['State'].notna().sum()
        print(f"Successfully extracted state for {state_count} out of {len(df_csv)} entries")
        
        # Show a sample of addresses and extracted states
        print("\nSample of addresses and extracted states:")
        sample_df = df_csv[['address', 'State']].head(5)
        print(sample_df)
    else:
        print("Warning: 'address' column not found in CSV file")
    
    # Prepare for merging
    print("\nMerging data from Excel into CSV...")
    
    # Check if required columns exist for merging
    if 'title' in df_csv.columns and 'State' in df_csv.columns and 'Restaurant' in df_excel.columns and 'State' in df_excel.columns:
        # Create copies of the DataFrames to avoid modifying the originals
        df_csv_copy = df_csv.copy()
        df_excel_copy = df_excel.copy()
        
        # Rename columns for merging
        df_csv_copy.rename(columns={'title': 'Restaurant_csv'}, inplace=True)
        df_excel_copy.rename(columns={'Restaurant': 'Restaurant_excel'}, inplace=True)
        
        # Create merge keys
        df_csv_copy['merge_key'] = df_csv_copy['Restaurant_csv'].str.strip().str.lower()
        df_excel_copy['merge_key'] = df_excel_copy['Restaurant_excel'].str.strip().str.lower()
        
        # Perform the merge on lowercase restaurant name and state
        print("\nMerging on restaurant name and state...")
        merged_df = pd.merge(
            df_csv_copy,
            df_excel_copy[['merge_key', 'ID', 'State', 'City', 'Address', 'Region', 'Crossout', 
                           'Honor Roll', 'Recommend', 'long', 'lat', 'geohash', 'Restaurant_excel']],
            on=['merge_key', 'State'],
            how='left'
        )
        
        # Count successful matches
        match_count = merged_df['Restaurant_excel'].notna().sum()
        print(f"Successfully matched {match_count} out of {len(df_csv)} entries")
        
        # Rename columns back but keep both title and Restaurant
        merged_df.rename(columns={'Restaurant_csv': 'title', 'Restaurant_excel': 'Restaurant'}, inplace=True)
        
        # Drop temporary merge key but keep both title and Restaurant columns
        merged_df.drop(['merge_key'], axis=1, inplace=True)
        
        # Show a sample of the merged data
        print("\nSample of merged data:")
        columns_to_show = ['title', 'Restaurant', 'State', 'City', 'Address', 'Region', 'Honor Roll', 'Recommend', 'long', 'lat']
        sample_merged = merged_df[columns_to_show].head(3)
        print(sample_merged)
        
        # Print statistics about the merged data
        print("\nMerge Statistics:")
        for col in ['City', 'Address', 'Region', 'Crossout', 'Honor Roll', 'Recommend', 'long', 'lat', 'geohash']:
            if col in merged_df.columns:
                non_null_count = merged_df[col].notna().sum()
                percentage = (non_null_count / len(merged_df)) * 100
                print(f"- {col}: {non_null_count} non-null values ({percentage:.2f}%)")
        
        # Reorder columns in the specified order
        print("\nReordering columns before export...")
        
        # Define the desired column order
        desired_order = [
            'ID', 'title', 'Restaurant', 'URL', 'address', 'Address', 'City', 'State', 'Region',
            'phone', 'hours', 'cost', 'content', 'Crossout', 'Honor Roll', 'Recommend',
            'long', 'lat', 'geohash'
        ]
        
        # Get the actual columns in the DataFrame
        actual_columns = merged_df.columns.tolist()
        
        # Create a new order with columns that exist in the DataFrame
        new_order = [col for col in desired_order if col in actual_columns]
        
        # Add any remaining columns that weren't in the desired order
        remaining_columns = [col for col in actual_columns if col not in desired_order]
        if remaining_columns:
            print(f"\nNote: The following columns were not in the specified order and will be added at the end:")
            for col in remaining_columns:
                print(f"- {col}")
            new_order.extend(remaining_columns)
        
        # Reorder the DataFrame
        merged_df = merged_df[new_order]
                
        # Save the merged DataFrame with reordered columns
        output_path = os.path.join(project_root, 'data', 'Roadfood_10th_reprocessed.csv')
        merged_df.to_csv(output_path, index=False)
        print(f"\nMerged CSV file with reordered columns saved to: {output_path}")
        
        # Second pass: Fuzzy matching for unmatched records
        print("\n\n=== Second Pass: Fuzzy Matching for Unmatched Records ===")
        
        # Import rapidfuzz for fuzzy string matching
        from rapidfuzz import process, fuzz
        
        # Filter unmatched records from the merged DataFrame
        unmatched_df = merged_df[merged_df['ID'].isna()].copy()
        print(f"Number of unmatched records after first pass: {len(unmatched_df)}")
        
        # Filter unused records from the Excel DataFrame
        used_ids = merged_df['ID'].dropna().unique()
        unused_excel_df = df_excel_copy[~df_excel_copy['ID'].isin(used_ids)].copy()
        
        # Further filter to only include rows where "10th Ed." is "y"
        if '10th Ed.' in unused_excel_df.columns:
            unused_excel_df = unused_excel_df[unused_excel_df['10th Ed.'] == 'y'].copy()
            print(f"Number of unused Excel records (10th Edition only): {len(unused_excel_df)}")
        else:
            print(f"Warning: '10th Ed.' column not found in Excel file")
            print(f"Number of unused Excel records (all): {len(unused_excel_df)}")
        
        if len(unmatched_df) > 0 and len(unused_excel_df) > 0:
            # Create dictionaries of unused Excel records for faster lookup, grouped by state
            excel_dict_by_state = {}
            for _, row in unused_excel_df.iterrows():
                state = row['State']
                if state not in excel_dict_by_state:
                    excel_dict_by_state[state] = {}
                excel_dict_by_state[state][row['Restaurant_excel'].lower()] = row
            
            # Set a similarity threshold for second pass
            threshold = 85  # Higher threshold for second pass
            
            # Track matches for second pass
            second_pass_matches = 0
            
            # Create a copy of the unmatched DataFrame to store matches
            matched_in_second_pass = []
            
            # Store best match info for all unmatched records (even below threshold)
            best_match_info = {}
            
            # Process each unmatched record
            print(f"\nPerforming fuzzy matching with threshold {threshold}...")
            for idx, row in unmatched_df.iterrows():
                restaurant_name = row['title'].lower()
                state = row['State']
                
                # Skip if state is missing or not in our Excel data
                if pd.isna(state) or state not in excel_dict_by_state:
                    continue
                
                # Get only Excel records for this state
                state_excel_dict = excel_dict_by_state[state]
                state_excel_names = list(state_excel_dict.keys())
                
                # Skip if no Excel records for this state
                if not state_excel_names:
                    continue
                
                # Find the best match using token_sort_ratio (handles word order differences)
                matches = process.extract(
                    restaurant_name, 
                    state_excel_names, 
                    scorer=fuzz.token_sort_ratio, 
                    limit=1
                )
                
                # Store best match info regardless of threshold (but only if states match)
                if matches and len(matches) > 0:
                    match_tuple = matches[0]
                    best_match = match_tuple[0]
                    score = match_tuple[1]
                    excel_row = state_excel_dict[best_match]
                    
                    # Store the best match info for this record
                    best_match_info[idx] = {
                        'Best_Match': excel_row['Restaurant_excel'],
                        'Match_Score': score,
                        'Excel_State': excel_row['State'],
                        'Excel_ID': excel_row['ID']
                    }
                
                # Check if we have a match above the threshold
                if matches and len(matches) > 0 and matches[0][1] >= threshold:
                    match_tuple = matches[0]  # This is a tuple with (match_string, score, index)
                    best_match = match_tuple[0]  # Get the matched string
                    score = match_tuple[1]  # Get the score
                    
                    excel_row = state_excel_dict[best_match]
                    
                    # Update the original merged DataFrame with the match
                    for col in ['ID', 'City', 'Address', 'Region', 'Crossout', 
                               'Honor Roll', 'Recommend', 'long', 'lat', 'geohash', 'Restaurant']:
                        if col in excel_row and col in merged_df.columns:
                            merged_df.at[idx, col] = excel_row[col]
                    
                    second_pass_matches += 1
                    matched_in_second_pass.append({
                        'CSV_Name': row['title'],
                        'Excel_Name': excel_row['Restaurant_excel'],
                        'State': row['State'],
                        'Score': score
                    })
            
            # Report results of second pass
            print(f"\nSecond pass results:")
            print(f"- Additional matches found: {second_pass_matches}")
            print(f"- Total match rate after second pass: {(match_count + second_pass_matches) / len(df_csv) * 100:.2f}%")
            
            if second_pass_matches > 0:
                # Show sample of matches from second pass
                print("\nSample of matches from second pass:")
                match_df = pd.DataFrame(matched_in_second_pass[:5] if len(matched_in_second_pass) > 5 else matched_in_second_pass)
                print(match_df)
            
            # Third pass: Lower threshold fuzzy matching for remaining unmatched records
            print("\n\n=== Third Pass: Lower Threshold Fuzzy Matching ===")
            
            # Filter unmatched records after second pass
            still_unmatched_df = merged_df[merged_df['ID'].isna()].copy()
            print(f"Number of unmatched records after second pass: {len(still_unmatched_df)}")
            
            # Filter unused Excel records after second pass
            used_ids_after_second_pass = merged_df['ID'].dropna().unique()
            unused_excel_df_third_pass = unused_excel_df[~unused_excel_df['ID'].isin(used_ids_after_second_pass)].copy()
            print(f"Number of unused Excel records for third pass: {len(unused_excel_df_third_pass)}")
            
            if len(still_unmatched_df) > 0 and len(unused_excel_df_third_pass) > 0:
                # Create dictionaries of unused Excel records for third pass
                excel_dict_by_state_third_pass = {}
                for _, row in unused_excel_df_third_pass.iterrows():
                    state = row['State']
                    if state not in excel_dict_by_state_third_pass:
                        excel_dict_by_state_third_pass[state] = {}
                    excel_dict_by_state_third_pass[state][row['Restaurant_excel'].lower()] = row
                
                # Set a lower similarity threshold for third pass
                lower_threshold = 25  # Lower threshold for third pass
                
                # Track matches for third pass
                third_pass_matches = 0
                
                # Create a copy of the unmatched DataFrame to store matches
                matched_in_third_pass = []
                
                # Process each still unmatched record
                print(f"\nPerforming third pass fuzzy matching with threshold {lower_threshold}...")
                for idx, row in still_unmatched_df.iterrows():
                    restaurant_name = row['title'].lower()
                    state = row['State']
                    
                    # Skip if state is missing or not in our Excel data
                    if pd.isna(state) or state not in excel_dict_by_state_third_pass:
                        continue
                    
                    # Get only Excel records for this state
                    state_excel_dict = excel_dict_by_state_third_pass[state]
                    state_excel_names = list(state_excel_dict.keys())
                    
                    # Skip if no Excel records for this state
                    if not state_excel_names:
                        continue
                    
                    # Find the best match using token_sort_ratio
                    matches = process.extract(
                        restaurant_name, 
                        state_excel_names, 
                        scorer=fuzz.token_sort_ratio, 
                        limit=1
                    )
                    
                    # Check if we have a match above the lower threshold
                    if matches and len(matches) > 0 and matches[0][1] >= lower_threshold:
                        match_tuple = matches[0]
                        best_match = match_tuple[0]
                        score = match_tuple[1]
                        
                        excel_row = state_excel_dict[best_match]
                        
                        # Update the original merged DataFrame with the match
                        for col in ['ID', 'City', 'Address', 'Region', 'Crossout', 
                                   'Honor Roll', 'Recommend', 'long', 'lat', 'geohash', 'Restaurant']:
                            if col in excel_row and col in merged_df.columns:
                                merged_df.at[idx, col] = excel_row[col]
                        
                        third_pass_matches += 1
                        matched_in_third_pass.append({
                            'CSV_Name': row['title'],
                            'Excel_Name': excel_row['Restaurant_excel'],
                            'State': row['State'],
                            'Score': score
                        })
                        
                        # Also update the best match info for this record
                        best_match_info[idx] = {
                            'Best_Match': excel_row['Restaurant_excel'],
                            'Match_Score': score,
                            'Excel_State': excel_row['State'],
                            'Excel_ID': excel_row['ID']
                        }
                
                # Report results of third pass
                print(f"\nThird pass results:")
                print(f"- Additional matches found: {third_pass_matches}")
                total_matches = match_count + second_pass_matches + third_pass_matches
                print(f"- Total match rate after all passes: {(total_matches) / len(df_csv) * 100:.2f}%")
                
                if third_pass_matches > 0:
                    # Show sample of matches from third pass
                    print("\nSample of matches from third pass:")
                    match_df = pd.DataFrame(matched_in_third_pass[:5] if len(matched_in_third_pass) > 5 else matched_in_third_pass)
                    print(match_df)
                    
                    # Save the improved merged DataFrame after all passes
                    final_output_path = os.path.join(project_root, 'data', 'Roadfood_10th_reprocessed_final.csv')
                    merged_df.to_csv(final_output_path, index=False)
                    print(f"\nFinal merged CSV file saved to: {final_output_path}")
            else:
                print("No records to process in third pass.")
                
                # Save the improved merged DataFrame after second pass only
                improved_output_path = os.path.join(project_root, 'data', 'Roadfood_10th_reprocessed_improved.csv')
                merged_df.to_csv(improved_output_path, index=False)
                print(f"\nImproved merged CSV file saved to: {improved_output_path}")
            
            # Export remaining unmatched records to a separate CSV
            final_unmatched_df = merged_df[merged_df['ID'].isna()].copy()
            unmatched_count = len(final_unmatched_df)
            print(f"\nRemaining unmatched records after all passes: {unmatched_count} ({unmatched_count / len(df_csv) * 100:.2f}%)")
            
            if unmatched_count > 0:
                # Add best match information to unmatched records
                for idx in final_unmatched_df.index:
                    if idx in best_match_info:
                        for key, value in best_match_info[idx].items():
                            final_unmatched_df.at[idx, key] = value
                
                # Reorder columns to put match info first
                unmatched_cols = final_unmatched_df.columns.tolist()
                match_info_cols = ['Match_Score', 'Best_Match', 'Excel_State', 'Excel_ID']
                other_cols = [col for col in unmatched_cols if col not in match_info_cols]
                new_col_order = match_info_cols + other_cols
                
                # Only include columns that exist
                new_col_order = [col for col in new_col_order if col in final_unmatched_df.columns]
                final_unmatched_df = final_unmatched_df[new_col_order]
                
                unmatched_output_path = os.path.join(project_root, 'data', 'Roadfood_10th_unmatched.csv')
                final_unmatched_df.to_csv(unmatched_output_path, index=False)
                print(f"Unmatched records saved to: {unmatched_output_path}")
            
            # Separately check for unused Excel records from the 10th edition
            print("\n=== Checking for Unused 10th Edition Excel Records ===")
            
            # Get all IDs that were used in the merged DataFrame
            used_ids_final = merged_df['ID'].dropna().unique()
            
            # Filter the original Excel DataFrame to find 10th edition records that weren't used
            if '10th Ed.' in df_excel.columns:
                tenth_edition_unused = df_excel[(df_excel['10th Ed.'] == 'y') & (~df_excel['ID'].isin(used_ids_final))].copy()
                unused_count = len(tenth_edition_unused)
                print(f"Number of unused 10th edition Excel records: {unused_count}")
                
                if unused_count > 0:
                    # Export the unused 10th edition Excel records
                    unused_excel_output_path = os.path.join(project_root, 'data', 'Roadfood_10th_excel_unused.csv')
                    tenth_edition_unused.to_csv(unused_excel_output_path, index=False)
                    print(f"Unused 10th edition Excel records saved to: {unused_excel_output_path}")
                    
                    # Show sample of unused records
                    print("\nSample of unused 10th edition Excel records:")
                    print(tenth_edition_unused[['ID', 'Restaurant', 'State', 'City']].head(5))
            else:
                print("Warning: '10th Ed.' column not found in Excel file")
        else:
            print("No unmatched records to process in second pass.")
    else:
        print("Warning: Required columns for merging not found in one or both DataFrames")
        missing_cols = []
        if 'title' not in df_csv.columns:
            missing_cols.append("'title' in CSV")
        if 'State' not in df_csv.columns:
            missing_cols.append("'State' in CSV")
        if 'Restaurant' not in df_excel.columns:
            missing_cols.append("'Restaurant' in Excel")
        if 'State' not in df_excel.columns:
            missing_cols.append("'State' in Excel")
        print(f"Missing columns: {', '.join(missing_cols)}")
    
