#!/usr/bin/env python3
"""
Time-based analysis for real estate data with quarterly rolling averages
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path):
    """Load the real estate dataset"""
    logger.info(f"Loading data from {file_path}")
    return pd.read_csv(file_path)

def convert_dates(df):
    """Convert date columns to datetime format"""
    date_columns = ['listing_date', 'sold_date', 'under_contract_date']
    
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Create quarter column based on sold date if available, otherwise listing date
    if 'sold_date' in df.columns:
        mask = df['sold_date'].notna()
        df.loc[mask, 'quarter'] = df.loc[mask, 'sold_date'].dt.to_period('Q')
    
    if 'listing_date' in df.columns:
        mask = (df['quarter'].isna()) & (df['listing_date'].notna())
        df.loc[mask, 'quarter'] = df.loc[mask, 'listing_date'].dt.to_period('Q')
    
    logger.info("Converted dates and created quarter column")
    return df

def calculate_rolling_averages(df, metric='price_per_sqft', window=4):
    """Calculate rolling averages by quarter"""
    logger.info(f"Calculating {window}-quarter rolling averages for {metric}")
    
    # Ensure we have quarter and price data
    if 'quarter' not in df.columns or metric not in df.columns:
        logger.error(f"Missing required columns: quarter or {metric}")
        return df
    
    # Convert PeriodIndex to string for groupby
    df['quarter_str'] = df['quarter'].astype(str)
    
    # Calculate quarterly averages
    quarterly_avg = df.groupby('quarter_str')[metric].agg(['mean', 'median', 'count']).reset_index()
    quarterly_avg = quarterly_avg.sort_values('quarter_str')
    
    # Calculate rolling averages
    quarterly_avg['rolling_mean'] = quarterly_avg['mean'].rolling(window=window, min_periods=1).mean()
    quarterly_avg['rolling_median'] = quarterly_avg['median'].rolling(window=window, min_periods=1).mean()
    
    # Calculate quarterly percentage changes
    quarterly_avg['pct_change'] = quarterly_avg['mean'].pct_change() * 100
    quarterly_avg['rolling_pct_change'] = quarterly_avg['rolling_mean'].pct_change() * 100
    
    logger.info(f"Calculated rolling averages across {len(quarterly_avg)} quarters")
    
    # Create lookup dictionary for mapping back to original dataframe
    rolling_avg_dict = dict(zip(quarterly_avg['quarter_str'], quarterly_avg['rolling_mean']))
    
    # Map the rolling average back to the original dataframe
    df['rolling_avg'] = df['quarter_str'].map(rolling_avg_dict)
    
    # Calculate price relative to rolling average
    df['pct_vs_rolling_avg'] = ((df[metric] / df['rolling_avg']) - 1) * 100
    
    return df, quarterly_avg

def calculate_time_adjustment_factors(quarterly_avg, base_period=None):
    """Calculate time adjustment factors relative to a base period"""
    # If no base period provided, use the most recent quarter
    if base_period is None:
        base_period = quarterly_avg['quarter_str'].max()
    
    # Get the rolling average for the base period
    base_avg = quarterly_avg.loc[quarterly_avg['quarter_str'] == base_period, 'rolling_mean'].values[0]
    
    # Calculate adjustment factors
    quarterly_avg['adjustment_factor'] = base_avg / quarterly_avg['rolling_mean']
    
    logger.info(f"Calculated time adjustment factors relative to {base_period}")
    return quarterly_avg

def apply_time_adjustments(df, quarterly_avg, metric='price_per_sqft'):
    """Apply time adjustment factors to prices"""
    # Create lookup for adjustment factors
    adjustment_dict = dict(zip(quarterly_avg['quarter_str'], quarterly_avg['adjustment_factor']))
    
    # Apply adjustment factors
    df['time_adjustment_factor'] = df['quarter_str'].map(adjustment_dict)
    df[f'time_adjusted_{metric}'] = df[metric] * df['time_adjustment_factor']
    
    logger.info(f"Applied time adjustments to {metric}")
    return df

def visualize_time_trends(quarterly_avg):
    """Visualize price trends over time"""
    plt.figure(figsize=(12, 8))
    
    # Plot quarterly averages and rolling average
    plt.subplot(2, 1, 1)
    plt.plot(quarterly_avg['quarter_str'], quarterly_avg['mean'], 'o-', alpha=0.6, label='Quarterly Average')
    plt.plot(quarterly_avg['quarter_str'], quarterly_avg['rolling_mean'], 'r-', linewidth=3, label='4-Quarter Rolling Average')
    
    # Format the chart
    plt.title('Price per Square Foot Trends Over Time', fontsize=14)
    plt.ylabel('Price per Square Foot ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot percentage changes
    plt.subplot(2, 1, 2)
    plt.bar(quarterly_avg['quarter_str'], quarterly_avg['pct_change'], alpha=0.6, label='Quarterly % Change')
    plt.plot(quarterly_avg['quarter_str'], quarterly_avg['rolling_pct_change'], 'g-', linewidth=3, label='Rolling % Change')
    
    # Format the chart
    plt.title('Quarterly Price Changes', fontsize=14)
    plt.xlabel('Quarter', fontsize=12)
    plt.ylabel('Percent Change (%)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    # Save the chart
    plt.savefig('price_trends_by_quarter.png')
    logger.info("Saved price trend visualization")

def main():
    try:
        # Load data
        df = load_data('real_estate_with_condition.csv')
        
        # Convert dates
        df = convert_dates(df)
        
        # Skip analysis if insufficient date data
        if df['quarter'].isna().sum() > len(df) * 0.8:
            logger.error("Insufficient date data for time-based analysis")
            return
        
        # Calculate rolling averages
        df, quarterly_avg = calculate_rolling_averages(df, metric='price_per_sqft', window=4)
        
        # Calculate time adjustment factors (relative to most recent quarter)
        quarterly_avg = calculate_time_adjustment_factors(quarterly_avg)
        
        # Apply time adjustments to prices
        df = apply_time_adjustments(df, quarterly_avg)
        
        # Visualize time trends
        visualize_time_trends(quarterly_avg)
        
        # Save results
        quarterly_avg.to_csv('quarterly_price_trends.csv', index=False)
        df.to_csv('time_adjusted_properties.csv', index=False)
        
        logger.info("Time-based analysis complete. Results saved to CSV files.")
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()
