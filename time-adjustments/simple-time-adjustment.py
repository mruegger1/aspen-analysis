#!/usr/bin/env python3
"""
Simple time-based analysis focusing only on sold properties
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path):
    """Load the real estate dataset"""
    logger.info(f"Loading data from {file_path}")
    return pd.read_csv(file_path)

def filter_sold_properties(df):
    """Filter properties with sold dates only"""
    if 'sold_date' not in df.columns:
        logger.error("No sold_date column found in dataset")
        return df.head(0)
        
    # Convert sold_date to datetime
    df['sold_date'] = pd.to_datetime(df['sold_date'], errors='coerce')
    
    # Filter for properties with sold dates
    sold_mask = df['sold_date'].notna()
    sold_df = df[sold_mask].copy()
    
    logger.info(f"Found {len(sold_df)} properties with sold dates")
    return sold_df

def calculate_quarterly_averages(df, price_col='price_per_sqft'):
    """Calculate quarterly price averages"""
    # Check price column exists
    if price_col not in df.columns:
        # Try to calculate it
        if 'sold_price' in df.columns and 'total_sqft' in df.columns:
            df[price_col] = df['sold_price'] / df['total_sqft']
            logger.info(f"Calculated {price_col} from sold_price and total_sqft")
        else:
            logger.error(f"Cannot find or calculate {price_col}")
            return df, None
    
    # Extract quarter from sold date
    df['quarter'] = df['sold_date'].dt.to_period('Q')
    df['quarter_str'] = df['quarter'].astype(str)
    
    # Calculate quarterly averages
    quarterly_avg = df.groupby('quarter_str')[price_col].agg(
        ['mean', 'median', 'count', 'std']
    ).reset_index()
    
    # Sort chronologically
    quarterly_avg = quarterly_avg.sort_values('quarter_str')
    
    logger.info(f"Calculated averages for {len(quarterly_avg)} quarters")
    return df, quarterly_avg

def calculate_rolling_averages(quarterly_avg, window=4):
    """Calculate rolling averages"""
    if len(quarterly_avg) == 0:
        logger.error("No quarterly data available")
        return quarterly_avg
        
    # Calculate rolling averages
    quarterly_avg['rolling_mean'] = quarterly_avg['mean'].rolling(window=window, min_periods=1).mean()
    quarterly_avg['rolling_median'] = quarterly_avg['median'].rolling(window=window, min_periods=1).mean()
    
    # Calculate percentage changes
    quarterly_avg['pct_change'] = quarterly_avg['mean'].pct_change() * 100
    quarterly_avg['rolling_pct_change'] = quarterly_avg['rolling_mean'].pct_change() * 100
    
    logger.info(f"Calculated {window}-quarter rolling averages")
    return quarterly_avg

def calculate_adjustment_factors(quarterly_avg):
    """Calculate time adjustment factors relative to most recent quarter"""
    if len(quarterly_avg) == 0:
        logger.error("No quarterly data available")
        return quarterly_avg
        
    # Get most recent quarter
    latest_quarter = quarterly_avg['quarter_str'].max()
    logger.info(f"Using {latest_quarter} as base period")
    
    # Get rolling average for most recent quarter
    base_avg = quarterly_avg.loc[quarterly_avg['quarter_str'] == latest_quarter, 'rolling_mean'].values[0]
    
    # Calculate adjustment factors
    quarterly_avg['adjustment_factor'] = base_avg / quarterly_avg['rolling_mean']
    
    # Cap extreme adjustments
    MAX_ADJUSTMENT = 3.0
    quarterly_avg['adjustment_factor'] = quarterly_avg['adjustment_factor'].clip(upper=MAX_ADJUSTMENT)
    
    return quarterly_avg

def apply_time_adjustments(df, quarterly_avg, price_col='price_per_sqft'):
    """Apply time adjustment factors"""
    if len(quarterly_avg) == 0:
        logger.error("No quarterly data available")
        return df
        
    # Create lookup dictionary
    adjustment_dict = dict(zip(quarterly_avg['quarter_str'], quarterly_avg['adjustment_factor']))
    
    # Apply adjustment factors
    df['time_adjustment_factor'] = df['quarter_str'].map(adjustment_dict)
    df[f'time_adjusted_{price_col}'] = df[price_col] * df['time_adjustment_factor']
    
    logger.info(f"Applied time adjustments to {price_col}")
    return df

def visualize_results(quarterly_avg):
    """Create visualization of price trends"""
    if len(quarterly_avg) < 2:
        logger.error("Not enough data for visualization")
        return
        
    plt.figure(figsize=(12, 8))
    
    # Price trends
    plt.subplot(2, 1, 1)
    plt.plot(quarterly_avg['quarter_str'], quarterly_avg['mean'], 'o-', alpha=0.6, label='Quarterly Average')
    plt.plot(quarterly_avg['quarter_str'], quarterly_avg['rolling_mean'], 'r-', linewidth=3, label='4-Quarter Rolling Average')
    
    plt.title('Price per Square Foot Trends', fontsize=14)
    plt.ylabel('Price per Square Foot ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Price changes
    plt.subplot(2, 1, 2)
    plt.bar(quarterly_avg['quarter_str'], quarterly_avg['pct_change'], alpha=0.6, label='Quarterly % Change')
    plt.plot(quarterly_avg['quarter_str'], quarterly_avg['rolling_pct_change'], 'g-', linewidth=2, label='Rolling % Change')
    
    plt.title('Price Changes by Quarter', fontsize=14)
    plt.xlabel('Quarter', fontsize=12)
    plt.ylabel('Percent Change (%)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('price_trends.png')
    logger.info("Saved visualization to price_trends.png")

def main():
    try:
        # Load data
        df = load_data('real_estate_with_condition.csv')
        
        # Filter for sold properties
        df = filter_sold_properties(df)
        
        if len(df) == 0:
            logger.error("No sold properties found")
            return
        
        # Calculate quarterly averages
        df, quarterly_avg = calculate_quarterly_averages(df)
        
        # Calculate rolling averages
        quarterly_avg = calculate_rolling_averages(quarterly_avg)
        
        # Calculate adjustment factors
        quarterly_avg = calculate_adjustment_factors(quarterly_avg)
        
        # Apply time adjustments
        df = apply_time_adjustments(df, quarterly_avg)
        
        # Create visualization
        visualize_results(quarterly_avg)
        
        # Save results
        quarterly_avg.to_csv('quarterly_price_trends.csv', index=False)
        df.to_csv('time_adjusted_properties.csv', index=False)
        
        logger.info("Time adjustment analysis complete")
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()
