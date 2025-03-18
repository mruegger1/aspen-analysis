#!/usr/bin/env python3
"""
Location Premium Analysis for Real Estate
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LocationPremiumAnalyzer:
    def __init__(self, data_path):
        """
        Initialize analyzer with data path
        
        Args:
            data_path (str): Path to CSV file with real estate data
        """
        self.df = pd.read_csv(data_path)
        
    def calculate_neighborhood_premiums(self):
        """
        Calculate price premiums for different neighborhoods
        
        Returns:
            pd.DataFrame: Neighborhood price statistics
        """
        # Group by neighborhood and calculate statistics
        neighborhood_stats = self.df.groupby('area').agg({
            'time_adjusted_price_per_sqft': [
                'mean', 'median', 'std', 
                lambda x: x.quantile(0.75) - x.quantile(0.25)
            ]
        })
        
        # Rename columns for clarity
        neighborhood_stats.columns = [
            'avg_price_per_sqft', 
            'median_price_per_sqft', 
            'price_std', 
            'price_iqr'
        ]
        
        # Calculate relative premium compared to overall mean
        overall_mean = self.df['time_adjusted_price_per_sqft'].mean()
        neighborhood_stats['premium_percentage'] = (
            (neighborhood_stats['avg_price_per_sqft'] - overall_mean) / overall_mean * 100
        )
        
        return neighborhood_stats.sort_values('premium_percentage', ascending=False)
    
    def analyze_short_term_rental_impact(self):
        """
        Analyze impact of short-term rental eligibility on property prices
        
        Returns:
            dict: STR impact metrics
        """
        # Ensure STR columns exist
        str_columns = ['short_term_rental', 'str_eligible']
        str_col = next((col for col in str_columns if col in self.df.columns), None)
        
        if not str_col:
            logger.warning("No short-term rental column found")
            return None
        
        # Compare prices
        str_price_comparison = {
            'str_eligible_mean': self.df[self.df[str_col]]['time_adjusted_price_per_sqft'].mean(),
            'non_str_mean': self.df[~self.df[str_col]]['time_adjusted_price_per_sqft'].mean()
        }
        
        # Calculate percentage difference
        str_price_comparison['premium_percentage'] = (
            (str_price_comparison['str_eligible_mean'] - str_price_comparison['non_str_mean']) 
            / str_price_comparison['non_str_mean'] * 100
        )
        
        return str_price_comparison
    
    def visualize_neighborhood_premiums(self, neighborhood_stats):
        """
        Create visualization of neighborhood price premiums
        
        Args:
            neighborhood_stats (pd.DataFrame): Neighborhood price statistics
        """
        plt.figure(figsize=(12, 6))
        neighborhood_stats['premium_percentage'].plot(kind='bar')
        plt.title('Neighborhood Price Premiums')
        plt.xlabel('Neighborhood')
        plt.ylabel('Price Premium (%)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('neighborhood_premiums.png')
    
    def run_analysis(self):
        """
        Execute full location premium analysis
        
        Returns:
            dict: Analysis results
        """
        try:
            # Neighborhood premiums
            neighborhood_stats = self.calculate_neighborhood_premiums()
            self.visualize_neighborhood_premiums(neighborhood_stats)
            
            # Short-term rental impact
            str_impact = self.analyze_short_term_rental_impact()
            
            return {
                'neighborhood_premiums': neighborhood_stats,
                'str_impact': str_impact
            }
        
        except Exception as e:
            logger.error(f"Location premium analysis failed: {e}")
            return None

def main():
    analyzer = LocationPremiumAnalyzer('time_adjusted_properties.csv')
    results = analyzer.run_analysis()
    
    if results:
        # Print neighborhood premiums
        print("Neighborhood Premiums:")
        print(results['neighborhood_premiums'])
        
        # Print STR impact
        if results['str_impact']:
            print("\nShort-Term Rental Impact:")
            print(results['str_impact'])

if __name__ == "__main__":
    main()
