#!/usr/bin/env python3
"""
Feature Engineering and Correlation Analysis for Real Estate Data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureEngineeringAnalyzer:
    def __init__(self, data_path):
        """Initialize analyzer with data path"""
        self.df = pd.read_csv(data_path)
        
    def engineer_features(self):
        """
        Create advanced features with historical normalization
        
        Returns:
            pd.DataFrame: Dataframe with engineered features
        """
        df = self.df.copy()
        
        # Determine appropriate date column
        date_columns = ['sale_date', 'list_date', 'sold_date', 'end_date']
        date_col = next((col for col in date_columns if col in df.columns), None)
        
        if date_col:
            df['sale_date'] = pd.to_datetime(df[date_col], errors='coerce')
        else:
            logger.warning("No date column found for temporal analysis")
            df['sale_date'] = pd.Timestamp.now()
        
        # Historical price normalization
        historical_reference_year = 2015
        df['years_since_reference'] = df['sale_date'].dt.year - historical_reference_year
        
        # Inflation and market adjustment
        if 'time_adjusted_price_per_sqft' in df.columns:
            df['historical_price_index'] = df['time_adjusted_price_per_sqft'] / (1 + 0.03 * df['years_since_reference'])
        
        # Bedroom-related features
        if all(col in df.columns for col in ['bedrooms', 'total_sqft']):
            df['sqft_per_bedroom'] = df['total_sqft'] / (df['bedrooms'] + 1)
            df['bedroom_density'] = df['bedrooms'] / df['total_sqft']
        
        # Property age and valuation features
        if 'year_built' in df.columns:
            df['property_age'] = df['sale_date'].dt.year - df['year_built']
            df['age_depreciation_factor'] = 1 / np.log(df['property_age'] + 1)
        
        # Location-based interaction features
        if all(col in df.columns for col in ['latitude', 'longitude', 'area']):
            df['location_encoded'] = df['area'].astype('category').cat.codes
        
        # Bathrooms to bedrooms ratio
        if all(col in df.columns for col in ['bedrooms', 'total_baths']):
            df['bath_to_bed_ratio'] = df['total_baths'] / (df['bedrooms'] + 1)
        
        return df
    
    def compute_correlation_matrix(self, target_column='time_adjusted_price_per_sqft'):
        """
        Compute and visualize correlation matrix
        
        Args:
            target_column (str): Column to focus correlation with
        
        Returns:
            pd.DataFrame: Correlation matrix
        """
        # Select numeric columns
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        
        # Compute correlation matrix
        correlation_matrix = self.df[numeric_cols].corr()
        
        # Visualize correlation with target column
        plt.figure(figsize=(12, 8))
        target_correlations = correlation_matrix[target_column].sort_values(ascending=False)
        
        sns.barplot(x=target_correlations.index, y=target_correlations.values)
        plt.title(f'Feature Correlations with {target_column}')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig('feature_correlations.png')
        
        return correlation_matrix
    
    def identify_key_features(self, correlation_matrix, target_column='time_adjusted_price_per_sqft', threshold=0.3):
        """
        Identify key features based on correlation strength
        
        Args:
            correlation_matrix (pd.DataFrame): Correlation matrix
            target_column (str): Column to analyze correlations with
            threshold (float): Absolute correlation threshold
        
        Returns:
            list: Important features
        """
        # Get correlations with target column
        target_correlations = correlation_matrix[target_column].abs()
        
        # Select features above threshold
        important_features = target_correlations[
            (target_correlations >= threshold) & 
            (target_correlations < 1.0)  # Exclude perfect correlation
        ]
        
        return list(important_features.index)
    
    def run_analysis(self):
        """
        Execute full feature engineering and correlation analysis
        
        Returns:
            dict: Analysis results
        """
        try:
            # Engineer features
            engineered_df = self.engineer_features()
            
            # Compute correlation matrix
            correlation_matrix = self.compute_correlation_matrix()
            
            # Identify key features
            key_features = self.identify_key_features(correlation_matrix)
            
            logger.info(f"Identified {len(key_features)} key features")
            
            return {
                'engineered_features': engineered_df,
                'correlation_matrix': correlation_matrix,
                'key_features': key_features
            }
        
        except Exception as e:
            logger.error(f"Feature engineering analysis failed: {e}")
            return None

def main():
    analyzer = FeatureEngineeringAnalyzer('time_adjusted_properties.csv')
    results = analyzer.run_analysis()
    
    if results:
        print("Key Features:", results['key_features'])
        print("\nCorrelation Matrix:\n", results['correlation_matrix'])

if __name__ == "__main__":
    main()