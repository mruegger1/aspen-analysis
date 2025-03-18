#!/usr/bin/env python3
"""
Analysis of STR-eligible condos with short walk times to gondola vs other central core condos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path):
    """Load the real estate dataset"""
    logger.info(f"Loading data from {file_path}")
    return pd.read_csv(file_path)

def filter_central_core_condos(df):
    """Filter dataset to only include condos in central core"""
    # Check columns to ensure they exist
    if 'property_type' not in df.columns or ('area' not in df.columns and 'sub_loc' not in df.columns):
        logger.error("Required columns missing: need property_type and either area or sub_loc")
        return df.head(0)  # Return empty dataframe
    
    # Identify condos based on property type - check both Residential with condo in features
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

def analyze_gondola_proximity(df):
    """Analyze the impact of gondola proximity on condo prices"""
    # Check for necessary columns
    if 'walk_time_to_gondola_min' not in df.columns:
        logger.warning("walk_time_to_gondola_min column not found")
        # Estimate based on distance if available
        if 'distance_to_gondola' in df.columns:
            logger.info("Estimating walk time from distance (assuming 3 mph walking speed)")
            # Average walking speed is about 3 mph
            df['walk_time_to_gondola_min'] = (df['distance_to_gondola'] / 3) * 60
        else:
            logger.error("Unable to determine gondola proximity - no time or distance columns")
            return df, pd.DataFrame(), (None, None)
    
    # Check for STR eligibility column
    if 'short_term_rental' in df.columns:
        df['str_eligible'] = df['short_term_rental'].astype(str).str.lower().isin(['y', 'yes', 'true', '1', 'eligible', 'allowed'])
    elif 'str_eligible' not in df.columns:
        # Try to infer from features
        if 'features' in df.columns:
            logger.info("Inferring STR eligibility from features")
            str_keywords = ['str ', 'short term rental', 'airbnb', 'vrbo', 'vacation rental']
            df['str_eligible'] = df['features'].astype(str).str.lower().apply(
                lambda x: any(kw in x for kw in str_keywords)
            )
        else:
            logger.warning("No STR eligibility info found, assuming all are eligible")
            df['str_eligible'] = True
    
    # Define short walk time to gondola (under 4 minutes)
    df['short_gondola_walk'] = df['walk_time_to_gondola_min'] < 4
    
    # Group condos by STR eligibility and gondola proximity
    df['condo_category'] = 'Other Downtown/Core Condos'
    str_close_gondola_mask = df['str_eligible'] & df['short_gondola_walk']
    df.loc[str_close_gondola_mask, 'condo_category'] = 'STR-Eligible, Close to Gondola'
    
    # Check if price_per_sqft column exists
    if 'price_per_sqft' not in df.columns:
        # Try to calculate it
        if 'asking_price' in df.columns and 'total_sqft' in df.columns:
            df['price_per_sqft'] = df['asking_price'] / df['total_sqft']
        elif 'sold_price' in df.columns and 'total_sqft' in df.columns:
            df['price_per_sqft'] = df['sold_price'] / df['total_sqft']
    
    # Calculate price per sqft statistics by category
    price_stats = df.groupby('condo_category')['price_per_sqft'].agg([
        'count', 'mean', 'median', 'std'
    ]).reset_index()
    
    logger.info(f"Price per sqft statistics by category:\n{price_stats}")
    
    # Extract the two groups for statistical testing
    group1 = df[df['condo_category'] == 'STR-Eligible, Close to Gondola']['price_per_sqft']
    group2 = df[df['condo_category'] == 'Other Downtown/Core Condos']['price_per_sqft']
    
    t_stat, p_value = None, None
    
    # Perform t-test if we have sufficient data
    if len(group1) >= 5 and len(group2) >= 5:
        # Perform t-test to check if difference is statistically significant
        t_stat, p_value = stats.ttest_ind(
            group1.dropna(), 
            group2.dropna(), 
            equal_var=False  # Using Welch's t-test (doesn't assume equal variance)
        )
        
        logger.info(f"T-test results: t-statistic={t_stat:.4f}, p-value={p_value:.4f}")
        
        # Calculate price premium percentage
        premium_pct = ((group1.mean() - group2.mean()) / group2.mean()) * 100
        logger.info(f"Price premium: {premium_pct:.2f}%")
    else:
        logger.warning("Insufficient data for statistical testing - need at least 5 properties in each group")
    
    return df, price_stats, (t_stat, p_value)

def visualize_results(df, price_stats):
    """Create visualizations to illustrate the findings"""
    # Set seaborn style
    sns.set(style="whitegrid")
    
    # Box plot of price per sqft by category
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='condo_category', y='price_per_sqft', data=df)
    plt.title('Price per Sqft: STR-Eligible Gondola-Proximate Condos vs Other Central Core Condos')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('central_core_condos_boxplot.png')
    
    # Bar chart of average price per sqft
    plt.figure(figsize=(10, 6))
    sns.barplot(x='condo_category', y='mean', data=price_stats)
    plt.title('Average Price per Sqft by Condo Category')
    plt.xticks(rotation=45)
    plt.ylabel('Average Price per Sqft ($)')
    plt.tight_layout()
    plt.savefig('central_core_condos_avg_price.png')
    
    logger.info("Saved visualizations to central_core_condos_boxplot.png and central_core_condos_avg_price.png")

def main():
    try:
        # Load data
        df = load_data('real_estate_with_condition.csv')
        
        # Filter to central core condos
        central_core_condos = filter_central_core_condos(df)
        
        # Skip analysis if insufficient data
        if len(central_core_condos) < 5:
            logger.error("Insufficient data - fewer than 5 central core condos found")
            return
        
        # Analyze gondola proximity impact
        results_df, price_stats, (t_stat, p_value) = analyze_gondola_proximity(central_core_condos)
        
        # Create visualizations
        visualize_results(results_df, price_stats)
        
        # Print key findings
        sig_level = 0.05
        if p_value < sig_level:
            logger.info(f"FINDING: There is a statistically significant difference in price per sqft (p={p_value:.4f})")
        else:
            logger.info(f"FINDING: No statistically significant difference in price per sqft (p={p_value:.4f})")
        
        # Save results to CSV
        results_df.to_csv('central_core_condo_analysis.csv', index=False)
        logger.info("Saved analysis results to central_core_condo_analysis.csv")
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()
