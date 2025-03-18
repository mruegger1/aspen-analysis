import pandas as pd
import numpy as np
from datetime import datetime
import os
from functools import lru_cache

class AspenFeatureEngineering:
    """
    Feature engineering for Aspen real estate analysis
    """
    
    def __init__(self, output_dir='output'):
        """Initialize feature engineering module"""
        self.output_dir = output_dir
        self.df = None
        os.makedirs(output_dir, exist_ok=True)
        print("Initialized Aspen Feature Engineering module")
    
    def load_data(self, df=None, load_from_file=False):
        """
        Load data either from a DataFrame or from processed data file
        
        Parameters:
        -----------
        df : DataFrame, optional
            DataFrame to use for feature engineering
        load_from_file : bool, default=False
            Whether to load data from processed_data.csv
        """
        if df is not None:
            self.df = df.copy()
            print(f"Using provided DataFrame with shape: {self.df.shape}")
            return True
            
        elif load_from_file:
            try:
                processed_path = os.path.join(self.output_dir, 'processed_data.csv')
                self.df = pd.read_csv(processed_path)
                print(f"Loaded processed data from {processed_path}, shape: {self.df.shape}")
                return True
            except Exception as e:
                print(f"Error loading processed data: {e}")
                return False
        else:
            print("No data provided. Either pass a DataFrame or set load_from_file=True")
            return False
    
    @lru_cache(maxsize=128)
    def categorize_bed_bath(self, beds, baths):
        """
        Create a category string based on bedroom and bathroom counts
        
        Parameters:
        -----------
        beds : int or float
            Number of bedrooms
        baths : int or float
            Number of bathrooms
        """
        try:
            # Convert to appropriate types
            beds = int(beds) if pd.notna(beds) else 0
            baths = float(baths) if pd.notna(baths) else 0
            
            # Cap at 7+ bedrooms to avoid sparse categories
            if beds > 7:
                beds = "7+"
            
            # Create bath categories
            if baths < 1:
                bath_cat = "LT1Bath"
            elif baths == 1:
                bath_cat = "1Bath"
            elif baths < 2:
                bath_cat = "1_5Bath"
            elif baths == 2:
                bath_cat = "2Bath"
            elif baths < 3:
                bath_cat = "2_5Bath"
            elif baths == 3:
                bath_cat = "3Bath"
            elif baths < 4:
                bath_cat = "3_5Bath"
            elif baths >= 4:
                bath_cat = "4PlusBath"
                
            return f"{beds}Bed_{bath_cat}"
        except:
            return "Unknown"
    
    def create_basic_features(self):
        """Create basic property features"""
        if self.df is None:
            print("No data loaded. Run load_data() first.")
            return False
            
        print("\n----- Creating Basic Features -----")
        
        # Basic property features
        self.df['Property Age'] = datetime.now().year - self.df['Year Built']
        self.df['Bath_Bedroom_Ratio'] = self.df['Total Baths'] / (self.df['Bedrooms'] + 1)
        self.df['Log_Total_SqFt'] = np.log1p(self.df['Total SqFt'])
        
        # Hotel condo interactions
        if 'Is_Hotel_Condo' in self.df.columns:
            self.df['Hotel_SqFt_Interaction'] = self.df['Is_Hotel_Condo'] * self.df['Total SqFt']
            self.df['Hotel_Price_Factor'] = self.df['Is_Hotel_Condo'] * 0.5
        
        # Interaction terms
        self.df['SqFt_Bedrooms_Interaction'] = self.df['Total SqFt'] * self.df['Bedrooms']
        self.df['Age_Baths_Interaction'] = self.df['Property Age'] * self.df['Total Baths']
        
        # Price trend features if available
        if '4-Quarter Rolling Median $/SqFt' in self.df.columns and '30-Day Rolling Median $/SqFt' in self.df.columns:
            self.df['Price_Trend_Ratio'] = self.df['4-Quarter Rolling Median $/SqFt'] / self.df['30-Day Rolling Median $/SqFt']
        
        print("Basic features created successfully")
        return True
    
    def create_bed_bath_features(self):
        """Create features related to bedroom and bathroom configurations"""
        if self.df is None:
            print("No data loaded. Run load_data() first.")
            return False
            
        print("\n----- Creating Bed-Bath Features -----")
        
        # Create bed-bath combination categories
        self.df['Bed_Bath_Category'] = self.df.apply(
            lambda row: self.categorize_bed_bath(
                row['Bedrooms'] if pd.notna(row['Bedrooms']) else 0,
                row['Total Baths'] if pd.notna(row['Total Baths']) else 0
            ), axis=1
        )
        
        # Create one-hot encoding for categories
        bed_bath_dummies = pd.get_dummies(self.df['Bed_Bath_Category'], prefix='BedBath')
        self.df = pd.concat([self.df, bed_bath_dummies], axis=1)
        
        # Log the counts of each category
        bed_bath_counts = bed_bath_dummies.sum().sort_values(ascending=False)
        print("\nBed-Bath Category Counts:")
        for category, count in bed_bath_counts.items():
            if count > 0:
                category_name = category.replace('BedBath_', '')
                print(f"- {category_name}: {int(count)}")
        
        # Bath premium indicators
        self.df['Extra_Baths'] = self.df['Total Baths'] - self.df['Bedrooms'].apply(lambda x: min(x, 1))
        self.df['Has_Extra_Baths'] = (self.df['Extra_Baths'] > 0).astype(int)
        
        # Bath per bedroom metrics
        self.df['Baths_Per_Bedroom'] = self.df['Total Baths'] / self.df['Bedrooms'].apply(lambda x: max(x, 1))
        
        # Bath richness categories
        conditions = [
            (self.df['Baths_Per_Bedroom'] < 1),
            (self.df['Baths_Per_Bedroom'] == 1),
            (self.df['Baths_Per_Bedroom'] > 1) & (self.df['Baths_Per_Bedroom'] <= 1.5),
            (self.df['Baths_Per_Bedroom'] > 1.5) & (self.df['Baths_Per_Bedroom'] <= 2),
            (self.df['Baths_Per_Bedroom'] > 2)
        ]
        
        values = ['LowBath', 'EqualBath', 'ModerateExtraBath', 'HighExtraBath', 'VeryHighExtraBath']
        self.df['Bath_Richness'] = np.select(conditions, values, default='Unknown')
        
        # Bath premium interaction with property type
        if 'Book Section' in self.df.columns:
            self.df['Type_Bath_Interaction'] = self.df['Book Section'] + '_' + self.df['Bath_Richness']
        
        # Bath score if components are available
        if 'Bath Score' not in self.df.columns and all(col in self.df.columns for col in ['Baths - Full', 'Baths - Half', 'Baths - 3/4']):
            self.df['Bath Score'] = (
                self.df['Baths - Full'].fillna(0) + 
                0.5 * self.df['Baths - Half'].fillna(0) + 
                0.75 * self.df['Baths - 3/4'].fillna(0)
            )
        
        print("Bed-bath features created successfully")
        return True
    
    def create_str_features(self):
        """Create features related to Short-Term Rental (STR) eligibility"""
        if self.df is None:
            print("No data loaded. Run load_data() first.")
            return False
            
        if 'Short Termable' not in self.df.columns:
            print("Short Termable column not found in dataset")
            return False
            
        print("\n----- Creating STR Features -----")
        
        # Binary indicator for STR eligibility
        self.df['Is_STR_Eligible'] = self.df['Short Termable'].map({'Yes': 1, 'No': 0, np.nan: np.nan})
        
        # STR interaction features
        self.df['STR_SqFt_Interaction'] = self.df['Is_STR_Eligible'].fillna(0) * self.df['Total SqFt']
        self.df['STR_Bedroom_Interaction'] = self.df['Is_STR_Eligible'].fillna(0) * self.df['Bedrooms']
        self.df['STR_Bath_Interaction'] = self.df['Is_STR_Eligible'].fillna(0) * self.df['Total Baths']
        
        # Flag for unknown STR status
        self.df['STR_Status_Known'] = (~self.df['Short Termable'].isna()).astype(int)
        
        # Furnished status if available
        if 'Furnished' in self.df.columns:
            self.df['Is_Furnished'] = self.df['Furnished'].map({'Yes': 1, 'Partial': 0.5, 'No': 0, np.nan: np.nan})
            self.df['Is_Furnished'] = self.df['Is_Furnished'].fillna(0)
            
            # Furnished * STR interaction
            self.df['STR_Furnished_Interaction'] = self.df['Is_STR_Eligible'].fillna(0) * self.df['Is_Furnished']
        
        # Location features
        if 'Gondola Walk Time (minutes)' in self.df.columns:
            # Inverse relationship (closer is better)
            self.df['Gondola_Proximity'] = 1 / (self.df['Gondola Walk Time (minutes)'].fillna(30) + 1)
            
            # Proximity * STR interaction
            self.df['STR_Gondola_Interaction'] = self.df['Is_STR_Eligible'].fillna(0) * self.df['Gondola_Proximity']
        
        print("STR features created successfully")
        return True
    
    def create_all_features(self):
        """Create all features in sequence"""
        print("\n===== FEATURE ENGINEERING =====")
        
        success = True
        success &= self.create_basic_features()
        success &= self.create_bed_bath_features()
        str_success = self.create_str_features()  # This might legitimately fail if no STR data
        
        # Save the featured data
        self.save_featured_data()
        
        if success:
            print("All required features created successfully")
        return self.df
    
    def save_featured_data(self):
        """Save the featured data to file"""
        if self.df is None:
            print("No data to save")
            return False
            
        try:
            featured_path = os.path.join(self.output_dir, 'featured_data.csv')
            self.df.to_csv(featured_path, index=False)
            print(f"Featured data saved to {featured_path}")
            return True
        except Exception as e:
            print(f"Error saving featured data: {e}")
            return False
    
    def get_data(self):
        """Return the featured dataframe"""
        return self.df
