import pandas as pd
import numpy as np
import os
import warnings
import traceback
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

class AspenDataPrep:
    """
    Data loading and preparation for Aspen real estate analysis
    """
    
    def __init__(self, data_path='combined_aspen_real_estate_with_str.csv', output_dir='output'):
        """Initialize with data path and output directory"""
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None
        self.property_type_medians = None
        self.target = 'Price per SqFt'  # Default target
        
        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)
        
        print("Initialized Aspen Data Preparation module")
    
    def load_data(self):
        """Load the dataset from file"""
        print(f"Loading data from: {self.data_path}")
        
        try:
            # Load the dataset
            self.df = pd.read_csv(self.data_path, encoding='cp1252')
            print(f"Dataset loaded successfully! Shape: {self.df.shape}")
            return True
        except Exception as e:
            print(f"Error loading dataset: {e}")
            traceback.print_exc()
            return False
    
    def prepare_data(self):
        """Prepare the dataset with filtering based on specifications"""
        if self.df is None:
            print("No data loaded. Run load_data() first.")
            return None
            
        print("\n===== DATA PREPARATION =====")
        print(f"Original dataset: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        
        # Identify target column
        self._find_target_column()
        
        # Filter properties
        self._filter_properties()
        
        # Handle outliers and missing values
        self._handle_missing_values()
        self._remove_outliers()
        
        # Calculate property type calibration
        self._calculate_property_type_calibration()
        
        # Save processed data
        self._save_processed_data()
        
        return self.df
    
    def _find_target_column(self):
        """Identify or create target column for price per square foot"""
        potential_target_columns = ['$/SqFt', 'Price/SqFt', 'Sold Price/SqFt', 'Price per SqFt']
        
        # Find the first matching column
        for column in potential_target_columns:
            if column in self.df.columns:
                self.target = column
                print(f"Found target column: {self.target}")
                return
        
        # Create target if needed
        if ('Sold Price' in self.df.columns and 'Total SqFt' in self.df.columns):
            self.df['Price per SqFt'] = self.df['Sold Price'] / self.df['Total SqFt']
            self.target = 'Price per SqFt'
            print(f"Created target column: {self.target}")
    
    def _filter_properties(self):
        """Filter to only sold properties and handle special categories"""
        # Filter to only sold properties
        if 'Status' in self.df.columns:
            self.df = self.df[self.df['Status'] == 'C']
            print(f"Filtering to only status 'C' (Closed) listings: {self.df.shape[0]} rows")
        elif 'listing_status' in self.df.columns:
            self.df = self.df[self.df['listing_status'] == 'Sold']
            print(f"Filtering to only 'Sold' listings: {self.df.shape[0]} rows")
        
        # Handle property types
        if 'Book Section' in self.df.columns:
            # Exclude mobile homes
            before_excl = self.df.shape[0]
            self.df = self.df[~self.df['Book Section'].isin(['Mobile Home'])]
            print(f"After excluding mobile homes: {self.df.shape[0]} rows")
            
            # Flag hotel condominiums
            hotel_condo_mask = (self.df['Book Section'] == 'Hotel Condominium')
            hotel_condo_count = hotel_condo_mask.sum()
            self.df['Is_Hotel_Condo'] = hotel_condo_mask.astype(int)
            print(f"Flagged {hotel_condo_count} hotel condominiums")
    
    def _handle_missing_values(self):
        """Handle missing values in key columns"""
        key_columns = ['Total SqFt', 'Bedrooms', 'Total Baths', self.target]
        columns_to_check = [col for col in key_columns if col in self.df.columns]
        
        # Report missing values
        for col in columns_to_check:
            missing_count = self.df[col].isna().sum()
            if missing_count > 0:
                print(f"Column {col} has {missing_count} missing values")
        
        # Drop rows with missing values in one operation
        self.df = self.df.dropna(subset=columns_to_check)
        print(f"After dropping rows with missing values: {self.df.shape[0]} rows")
    
    def _remove_outliers(self):
        """Remove price outliers using IQR method"""
        # Calculate bounds using percentiles
        q1 = self.df[self.target].quantile(0.01)
        q3 = self.df[self.target].quantile(0.99)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Filter out extreme outliers
        outlier_mask = (self.df[self.target] < lower_bound) | (self.df[self.target] > upper_bound)
        outlier_count = outlier_mask.sum()
        print(f"Identified {outlier_count} price outliers (${lower_bound:.2f} to ${upper_bound:.2f})")
        self.df = self.df[~outlier_mask]
        print(f"After removing outliers: {self.df.shape[0]} rows")
    
    def _calculate_property_type_calibration(self):
        """Calculate property type medians and calibration factors"""
        if 'Book Section' in self.df.columns:
            self.property_type_medians = self.df.groupby('Book Section')[self.target].median()
            
            print("\nMedian Price per SqFt by Property Type:")
            for prop_type, median in self.property_type_medians.items():
                print(f"- {prop_type}: ${median:.2f}")
            
            # Create calibration factor
            overall_median = self.df[self.target].median()
            self.df['Property_Type_Calibration'] = self.df['Book Section'].map(
                lambda x: self.property_type_medians.get(x, overall_median) / overall_median)
    
    def _save_processed_data(self):
        """Save the processed dataframe to file"""
        try:
            output_path = os.path.join(self.output_dir, 'processed_data.csv')
            self.df.to_csv(output_path, index=False)
            print(f"Processed data saved to {output_path}")
            
            # Also save property type medians
            if self.property_type_medians is not None:
                medians_path = os.path.join(self.output_dir, 'property_type_medians.csv')
                self.property_type_medians.to_csv(medians_path)
                print(f"Property type medians saved to {medians_path}")
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def get_data(self):
        """Return the prepared dataframe"""
        return self.df
    
    def get_property_type_medians(self):
        """Return the property type medians"""
        return self.property_type_medians