import pandas as pd
import numpy as np
import requests
import time
from math import radians, sin, cos, sqrt, atan2

def get_distance_matrix(origins, destination, api_key):
    """
    Use Google Maps Distance Matrix API to get actual driving distances and times
    
    Args:
        origins: List of (lat, lng) tuples
        destination: (lat, lng) tuple for destination
        api_key: Google Maps API key
        
    Returns:
        List of (distance_miles, duration_minutes) tuples
    """
    # Format origins for API request
    origins_str = "|".join([f"{lat},{lng}" for lat, lng in origins])
    dest_str = f"{destination[0]},{destination[1]}"
    
    # Build URL
    base_url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params = {
        "origins": origins_str,
        "destinations": dest_str,
        "mode": "driving",
        "units": "imperial",
        "key": api_key
    }
    
    # Make request
    response = requests.get(base_url, params=params)
    data = response.json()
    
    # Parse results
    results = []
    if data["status"] == "OK":
        for row in data["rows"]:
            element = row["elements"][0]
            if element["status"] == "OK":
                distance_miles = element["distance"]["value"] / 1609.34  # Convert meters to miles
                duration_minutes = element["duration"]["value"] / 60  # Convert seconds to minutes
                results.append((distance_miles, duration_minutes))
            else:
                results.append((None, None))
    
    return results

def calculate_drive_time(lat1, lon1, lat2, lon2, use_api=False, api_key=None):
    """
    Calculate drive time between two coordinates
    If use_api=True, uses Google Maps API, otherwise uses distance-based estimation
    """
    if use_api and api_key:
        results = get_distance_matrix([(lat1, lon1)], (lat2, lon2), api_key)
        if results and results[0][0] is not None:
            return results[0]
    
    # Fallback to estimation if API not used or failed
    # Convert coordinates from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula to calculate distance
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance_km = 6371 * c  # Earth radius in km
    distance_miles = distance_km * 0.621371
    
    # Improved time estimate for mountain roads in Aspen area
    # Aspen has winding mountain roads with slower speeds
    if distance_miles < 2:
        avg_speed = 15  # mph (slower for in-town)
    elif distance_miles < 10:
        avg_speed = 25  # mph (mountain roads)
    else:
        avg_speed = 35  # mph (highway portions)
    
    # Add 30% to account for elevation, winding roads, and traffic
    drive_time_minutes = (distance_miles / avg_speed) * 60 * 1.3
    
    return distance_miles, drive_time_minutes

def extract_property_condition(df):
    """Extract property condition from features text and calculate remodel metrics"""
    # Extract condition from features text
    def extract_condition(features_text):
        if pd.isna(features_text):
            return "Not Specified"
        
        features_text = str(features_text).lower()
        
        if "under construction" in features_text:
            return "Under Construction"
        elif any(term in features_text for term in ["condition|new", "new build", "new construction"]):
            return "New Build" 
        elif "condition|excellent" in features_text or "excellent condition" in features_text:
            return "Excellent Condition"
        elif "condition|good" in features_text or "good condition" in features_text:
            return "Good Condition"
        elif "condition|average" in features_text or "average condition" in features_text:
            return "Average Condition"
        elif "fixer" in features_text or "needs work" in features_text or "as is" in features_text:
            return "Needs Work"
        else:
            return "Not Specified"
    
    # Apply condition extraction
    if 'features' in df.columns:
        df['property_condition'] = df['features'].apply(extract_condition)
        print(f"Property condition distribution:\n{df['property_condition'].value_counts()}")
    else:
        print("WARNING: 'features' column not found, skipping condition extraction")
    
    # Calculate remodel metrics if year_remodeled exists
    if 'year_remodeled' in df.columns and 'year_built' in df.columns:
        current_year = pd.to_datetime('today').year
        
        # Handle missing values
        df['year_remodeled'] = pd.to_numeric(df['year_remodeled'], errors='coerce')
        df['year_built'] = pd.to_numeric(df['year_built'], errors='coerce')
        
        # Calculate years since remodel
        mask = df['year_remodeled'].notna() & (df['year_remodeled'] > 0)
        df.loc[mask, 'years_since_remodel'] = current_year - df.loc[mask, 'year_remodeled']
        
        # Calculate if recently remodeled
        df['recently_remodeled'] = (df['years_since_remodel'] <= 5) & df['years_since_remodel'].notna()
    
    # Calculate price_per_sqft for active listings
    if 'asking_price' in df.columns and 'concession_amount' in df.columns and 'total_sqft' in df.columns:
        # Identify active listings (those without a sold_date)
        active_mask = df['status'].isin(['Active', 'New', 'Pending', 'Under Contract']) | df['sold_date'].isna()
        
        # Convert columns to numeric to ensure proper calculation
        df['asking_price'] = pd.to_numeric(df['asking_price'], errors='coerce')
        df['concession_amount'] = pd.to_numeric(df['concession_amount'], errors='coerce').fillna(0)
        df['total_sqft'] = pd.to_numeric(df['total_sqft'], errors='coerce')
        
        # Calculate price_per_sqft for active listings
        valid_mask = active_mask & (df['total_sqft'] > 0) & (df['asking_price'].notna())
        df.loc[valid_mask, 'price_per_sqft'] = (df.loc[valid_mask, 'asking_price'] - df.loc[valid_mask, 'concession_amount']) / df.loc[valid_mask, 'total_sqft']
        
        # Log the calculation
        count = valid_mask.sum()
        print(f"Calculated price_per_sqft for {count} active listings")
    
    # Calculate drive time to Mill St Aspen (39.189128, -106.819765)
    if 'latitude' in df.columns and 'longitude' in df.columns:
        # Target locations
        destinations = {
            'mill_st': (39.189128, -106.819765, "Mill St"),
            'gondola': (39.186779, -106.818051, "Gondola Plaza")
        }
        
        # Convert latitude and longitude to numeric
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        
        # Create mask for valid coordinates
        valid_coords = df['latitude'].notna() & df['longitude'].notna()
        
        # Initialize columns for each destination
        for dest_key, (_, _, dest_name) in destinations.items():
            df[f'distance_to_{dest_key}'] = np.nan
            df[f'drive_time_to_{dest_key}_min'] = np.nan
            df[f'walk_time_to_{dest_key}_min'] = np.nan
        
        # Ask for API key (optional)
        use_api = False
        api_key = None
        try:
            api_key_input = input("Enter Google Maps API key for accurate times (or press Enter to skip): ").strip()
            if api_key_input:
                api_key = api_key_input
                use_api = True
                print("Using Google Maps API for time calculations")
            else:
                print("Using distance-based estimation for times")
        except:
            print("Using distance-based estimation for times")
        
        # Process calculations for each destination
        for dest_key, (dest_lat, dest_lon, dest_name) in destinations.items():
            # Process in batches if using API (to avoid exceeding rate limits)
            if use_api and api_key:
                batch_size = 10  # Google allows up to 25 origins in a single request
                valid_indices = df[valid_coords].index
                
                for i in range(0, len(valid_indices), batch_size):
                    batch_indices = valid_indices[i:i+batch_size]
                    batch_rows = df.loc[batch_indices]
                    
                    # Prepare origins for batch API call
                    origins = [(row['latitude'], row['longitude']) for _, row in batch_rows.iterrows()]
                    
                    # Get drive times for batch
                    drive_results = get_distance_matrix(origins, (dest_lat, dest_lon), api_key)
                    
                    # Get walk times (different API call with mode=walking)
                    params = {
                        "origins": "|".join([f"{lat},{lng}" for lat, lng in origins]),
                        "destinations": f"{dest_lat},{dest_lon}",
                        "mode": "walking",
                        "units": "imperial",
                        "key": api_key
                    }
                    base_url = "https://maps.googleapis.com/maps/api/distancematrix/json"
                    response = requests.get(base_url, params=params)
                    walk_data = response.json()
                    
                    walk_results = []
                    if walk_data["status"] == "OK":
                        for row in walk_data["rows"]:
                            element = row["elements"][0]
                            if element["status"] == "OK":
                                distance_miles = element["distance"]["value"] / 1609.34
                                duration_minutes = element["duration"]["value"] / 60
                                walk_results.append((distance_miles, duration_minutes))
                            else:
                                walk_results.append((None, None))
                    
                    # Update DataFrame with results
                    for j, idx in enumerate(batch_indices):
                        # Update drive time
                        if j < len(drive_results) and drive_results[j][0] is not None:
                            df.at[idx, f'distance_to_{dest_key}'] = round(drive_results[j][0], 2)
                            df.at[idx, f'drive_time_to_{dest_key}_min'] = round(drive_results[j][1], 1)
                        
                        # Update walk time
                        if j < len(walk_results) and walk_results[j][0] is not None:
                            df.at[idx, f'walk_time_to_{dest_key}_min'] = round(walk_results[j][1], 1)
                    
                    # Add delay to avoid hitting API rate limits
                    if i + batch_size < len(valid_indices):
                        time.sleep(1)
            else:
                # Calculate individually using estimation formula
                for idx, row in df[valid_coords].iterrows():
                    # Calculate drive time
                    distance, drive_time = calculate_drive_time(
                        row['latitude'], row['longitude'], dest_lat, dest_lon,
                        use_api=False, api_key=None
                    )
                    df.at[idx, f'distance_to_{dest_key}'] = round(distance, 2)
                    df.at[idx, f'drive_time_to_{dest_key}_min'] = round(drive_time, 1)
                    
                    # Estimate walk time (average walking speed is about 3 mph)
                    # Using straight-line distance with a winding factor of 1.3
                    walk_time = (distance * 1.3) / 3 * 60
                    df.at[idx, f'walk_time_to_{dest_key}_min'] = round(walk_time, 1)
            
            print(f"Calculated times to {dest_name} for {valid_coords.sum()} properties")
    
    return df

if __name__ == "__main__":
    # Load the main dataset 
    main_file = "cleaned_real_estate_dataset.csv"
    print(f"Loading main dataset from {main_file}...")
    main_df = pd.read_csv(main_file)
    
    # Extract property condition directly from the main dataset
    main_df = extract_property_condition(main_df)
    
    # Now also load the adjusted analysis if it exists
    try:
        adjusted_file = "adjusted_price_analysis.csv"
        print(f"Loading {adjusted_file}...")
        adjusted_df = pd.read_csv(adjusted_file)
        
        # Identify the key column to use for merging
        possible_keys = ['list_number', 'sub_loc', 'area']
        merge_key = None
        for key in possible_keys:
            if key in adjusted_df.columns and key in main_df.columns:
                merge_key = key
                break
        
        if merge_key:
            print(f"Merging datasets on '{merge_key}'...")
            # Keep only the condition columns from main_df to avoid duplicates
            condition_cols = ['property_condition', 'years_since_remodel', 'recently_remodeled', 
                             'price_per_sqft', 'distance_to_mill_st', 'drive_time_to_mill_st_min',
                             'walk_time_to_mill_st_min', 'distance_to_gondola', 'drive_time_to_gondola_min',
                             'walk_time_to_gondola_min']
            cols_to_merge = [merge_key] + [col for col in condition_cols if col in main_df.columns]
            
            # Merge with adjusted price data
            result_df = adjusted_df.merge(main_df[cols_to_merge], on=merge_key, how='left')
            print(f"Merged condition data. New shape: {result_df.shape}")
        else:
            print("No common key found. Using main dataset with condition added.")
            result_df = main_df
    except FileNotFoundError:
        print(f"Adjusted price file not found. Using main dataset with condition added.")
        result_df = main_df
    
    # Save enhanced dataset
    output_file = "real_estate_with_condition.csv"
    result_df.to_csv(output_file, index=False)
    print(f"Saved enhanced dataset to {output_file}")