#!/usr/bin/env python3
"""
Analysis of premium addresses for condominiums in the central core
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path):
    """Load the real estate dataset"""
    logger.info(f"Loading data from {file_path}")
    return pd.read_csv(file_path)

def filter_central_core_condos(df):
    """Filter dataset to only include condos in central core"""
    # Check for required columns
    if 'property_type' not in df.columns:
        logger.error("Required column missing: property_type")
        return df.head(0)  # Return empty dataframe
    
    # Identify condos based on property type
    condo_mask = False
    
    # Check if explicit condo type exists
    if 'resolved_property_type' in df.columns:
        condo_mask = df['resolved_property_type'].str.lower().str.contains('condo', na=False)
    
    # Fallback to checking features for condo keywords
    if not any(condo_mask) and 'features' in df.columns:
        condo_mask = df['features'].str.lower().str.contains('condo|condominium', na=False)
    
    # If still no condos, check if units have HOA fees (likely condos)
    if not any(condo_mask) and 'hoa_fee' in df.columns:
        condo_mask = df['hoa_fee'] > 0
    
    # Identify central core properties
    central_core_mask = False
    
    # Check area column first
    if 'area' in df.columns:
        central_core_mask = df['area'].str.lower().str.contains('central core', na=False)
    
    # If no central core in area, check sub_loc for downtown indicators
    if not any(central_core_mask) and 'sub_loc' in df.columns:
        downtown_keywords = ['core', 'downtown', 'central', 'main st', 'mill st', 'cooper', 'hopkins', 'hyman', 'galena']
        # Check each downtown keyword against sub_loc
        for keyword in downtown_keywords:
            central_core_mask = central_core_mask | df['sub_loc'].str.lower().str.contains(keyword, na=False)
    
    # Combine masks
    filtered_df = df[condo_mask & central_core_mask].copy()
    logger.info(f"Found {len(filtered_df)} condos in central core/downtown area")
    
    return filtered_df

def extract_street_name(address):
    """Extract standardized street name from address"""
    if pd.isna(address):
        return None
    
    # Standardize address format
    address = str(address).lower()
    
    # First try with full pattern
    main_streets = ['main', 'cooper', 'hopkins', 'hyman', 'galena', 'mill', 'monarch', 'durant', 'spring', 
                   'aspen', 'bleeker', 'hallam', 'francis', 'garmisch', 'gibson', 'smuggler']
    
    for street in main_streets:
        if street in address:
            return street
    
    # Handle numbered streets
    numbered_pattern = r'\d+(?:st|nd|rd|th)?\s+(?:street|st)\b'
    if re.search(numbered_pattern, address):
        match = re.search(r'(\d+)', address)
        if match:
            return f"{match.group(1)}th street"
    
    # Look for directional + street combinations
    dir_street_pattern = r'\b([nsew])[.\s]+([a-z]+)'
    dir_match = re.search(dir_street_pattern, address)
    if dir_match:
        direction = dir_match.group(1)
        street = dir_match.group(2)
        return f"{direction} {street}"
    
    # Extract any street name using regex
    street_pattern = r'\d+\s+(?:[nsew][.\s]+)?([a-z]+)'
    street_match = re.search(street_pattern, address)
    
    if street_match:
        return street_match.group(1).strip()
    
    return None

def analyze_premium_addresses(df):
    """Analyze which addresses command premium prices"""
    logger.info("Analyzing premium addresses")
    
    # Ensure we have price per sqft
    if 'price_per_sqft' not in df.columns:
        if 'asking_price' in df.columns and 'total_sqft' in df.columns:
            df['price_per_sqft'] = df['asking_price'] / df['total_sqft']
        elif 'sold_price' in df.columns and 'total_sqft' in df.columns:
            df['price_per_sqft'] = df['sold_price'] / df['total_sqft']
        else:
            logger.error("Cannot calculate price per sqft - missing data")
            return df, None
    
    # Extract street name from different address fields
    if 'full_address' in df.columns:
        df['street_name'] = df['full_address'].apply(extract_street_name)
    elif 'street_name' in df.columns and 'street_number' in df.columns:
        # Create temporary address for extraction
        df['temp_address'] = df['street_number'].astype(str) + ' ' + df['street_name'].astype(str)
        df['street_name'] = df['temp_address'].apply(extract_street_name)
        df.drop('temp_address', axis=1, inplace=True)
    elif 'street_name' in df.columns:
        # Standardize existing street name
        df['street_name'] = df['street_name'].str.lower()
    else:
        logger.error("No address column found")
        return df, None
    
    # Calculate overall average price per sqft
    avg_price = df['price_per_sqft'].mean()
    logger.info(f"Average price per sqft in central core: ${avg_price:.2f}")
    
    # Check data quality
    street_counts = df['street_name'].value_counts()
    identified_streets = len(street_counts)
    logger.info(f"Identified {identified_streets} unique streets with at least one property")
    
    # Filter out None values
    df = df[df['street_name'].notna()].copy()
    
    # Group by street name and calculate stats
    street_stats = df.groupby('street_name').agg(
        count=('price_per_sqft', 'count'),
        avg_price=('price_per_sqft', 'mean'),
        median_price=('price_per_sqft', 'median'),
        min_price=('price_per_sqft', 'min'),
        max_price=('price_per_sqft', 'max'),
        std_price=('price_per_sqft', 'std')
    ).reset_index()
    
    # Calculate premium percentage compared to overall average
    street_stats['premium_pct'] = ((street_stats['avg_price'] - avg_price) / avg_price) * 100
    
    # Filter for streets with at least 3 properties (changed from 5 to get more results)
    street_stats = street_stats[street_stats['count'] >= 3].copy()
    
    # Sort by premium percentage
    street_stats = street_stats.sort_values('premium_pct', ascending=False)
    
    # Mark premium streets (>15% above average)
    street_stats['is_premium'] = street_stats['premium_pct'] > 15
    premium_streets = street_stats[street_stats['is_premium']]['street_name'].tolist()
    
    logger.info(f"Identified {len(premium_streets)} premium streets")
    logger.info(f"Top premium streets: {premium_streets[:5]}")
    
    return df, street_stats

def analyze_premium_buildings(df):
    """Analyze which specific buildings command premium prices"""
    logger.info("Analyzing premium buildings")
    
    # Try to identify building names
    building_col = None
    for col in ['sub_loc', 'building_name', 'complex']:
        if col in df.columns:
            building_col = col
            break
    
    if not building_col:
        logger.warning("No building identifier column found")
        return df, None
    
    # Calculate overall average price per sqft
    avg_price = df['price_per_sqft'].mean()
    
    # Group by building and calculate stats
    building_stats = df.groupby(building_col).agg(
        count=('price_per_sqft', 'count'),
        avg_price=('price_per_sqft', 'mean'),
        median_price=('price_per_sqft', 'median'),
        min_price=('price_per_sqft', 'min'),
        max_price=('price_per_sqft', 'max'),
        std_price=('price_per_sqft', 'std')
    ).reset_index()
    
    # Calculate premium percentage
    building_stats['premium_pct'] = ((building_stats['avg_price'] - avg_price) / avg_price) * 100
    
    # Filter for buildings with at least 3 properties
    building_stats = building_stats[building_stats['count'] >= 3].copy()
    
    # Sort by premium percentage
    building_stats = building_stats.sort_values('premium_pct', ascending=False)
    
    # Mark premium buildings (>20% above average)
    building_stats['is_premium'] = building_stats['premium_pct'] > 20
    premium_buildings = building_stats[building_stats['is_premium']][building_col].tolist()
    
    # Mark properties in premium buildings
    df['in_premium_building'] = df[building_col].isin(premium_buildings)
    
    logger.info(f"Identified {len(premium_buildings)} premium buildings")
    if len(premium_buildings) > 0:
        logger.info(f"Top premium buildings: {premium_buildings[:5]}")
    
    return df, building_stats

def visualize_results(street_stats, building_stats=None):
    """Create visualizations to illustrate premium addresses"""
    # Set seaborn style
    sns.set(style="whitegrid")
    
    # Visualize top premium streets
    plt.figure(figsize=(12, 8))
    top_streets = street_stats.head(15)
    ax = sns.barplot(x='premium_pct', y='street_name', data=top_streets)
    plt.title('Premium Streets in Central Core (% Above Average Price)', fontsize=14)
    plt.xlabel('Premium Percentage', fontsize=12)
    plt.ylabel('Street Name', fontsize=12)
    plt.axvline(x=0, color='red', linestyle='--')
    
    # Add value labels
    for i, p in enumerate(ax.patches):
        width = p.get_width()
        plt.text(width + 1, p.get_y() + p.get_height()/2, 
                f'{width:.1f}%', ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig('premium_streets.png')
    
    # Visualize premium buildings if available
    if building_stats is not None and not building_stats.empty:
        plt.figure(figsize=(12, 8))
        top_buildings = building_stats.head(15)
        ax = sns.barplot(x='premium_pct', y=building_stats.columns[0], data=top_buildings)
        plt.title('Premium Buildings in Central Core (% Above Average Price)', fontsize=14)
        plt.xlabel('Premium Percentage', fontsize=12)
        plt.ylabel('Building Name', fontsize=12)
        plt.axvline(x=0, color='red', linestyle='--')
        
        # Add value labels
        for i, p in enumerate(ax.patches):
            width = p.get_width()
            plt.text(width + 1, p.get_y() + p.get_height()/2, 
                    f'{width:.1f}%', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig('premium_buildings.png')
    
    logger.info("Saved visualizations")

def main():
    try:
        # Load data
        df = load_data('real_estate_with_condition.csv')
        
        # Filter to central core condos
        central_core_condos = filter_central_core_condos(df)
        
        # Skip analysis if insufficient data
        if len(central_core_condos) < 20:
            logger.error("Insufficient data - fewer than 20 central core condos found")
            return
        
        # Analyze premium streets
        results_df, street_stats = analyze_premium_addresses(central_core_condos)
        
        # Analyze premium buildings
        results_df, building_stats = analyze_premium_buildings(results_df)
        
        # Visualize results
        if street_stats is not None:
            visualize_results(street_stats, building_stats)
        
        # Save results to CSV
        street_stats.to_csv('premium_streets_analysis.csv', index=False)
        if building_stats is not None:
            building_stats.to_csv('premium_buildings_analysis.csv', index=False)
        results_df.to_csv('premium_address_results.csv', index=False)
        
        logger.info("Analysis complete. Results saved to CSV files.")
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()
