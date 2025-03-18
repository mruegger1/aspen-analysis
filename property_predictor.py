import pandas as pd
import numpy as np
import joblib
import traceback
import os
from datetime import datetime

class PropertyPredictor:
    """
    Lightweight class for making predictions with the trained model
    """
    
    def __init__(self, model_path=None, output_dir='output'):
        """
        Initialize the prediction class and load the trained model
        
        Parameters:
        -----------
        model_path : str, optional
            Path to the model file (if None, searches in output_dir)
        output_dir : str, default='output'
            Directory where models and outputs are stored
        """
        self.model = None
        self.feature_names = []
        self.numeric_features = []
        self.categorical_features = []
        self.target = 'Price per SqFt'
        self.output_dir = output_dir
        
        # Try to load model
        if model_path is None:
            # Look for model in output directory
            for model_type in ['random_forest', 'linear_regression']:
                potential_path = os.path.join(output_dir, f"{model_type}_model.pkl")
                if os.path.exists(potential_path):
                    model_path = potential_path
                    break
        
        if model_path is not None:
            try:
                self.load_model(model_path)
            except Exception as e:
                print(f"Error loading model: {e}")
                traceback.print_exc()
        else:
            print("No model specified. Use load_model() to load a trained model.")
    
    def load_model(self, filename):
        """
        Load the trained model and metadata
        
        Parameters:
        -----------
        filename : str
            Path to the model file
            
        Returns:
        --------
        bool: Success status
        """
        try:
            print(f"Loading model from {filename}")
            data = joblib.load(filename)
            
            # Extract model and metadata
            self.model = data['model']
            self.feature_names = data['feature_names']
            self.numeric_features = data.get('numeric_features', [])
            self.categorical_features = data.get('categorical_features', [])
            self.target = data.get('target', 'Price per SqFt')
            
            print(f"Model loaded successfully with {len(self.feature_names)} features")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            traceback.print_exc()
            return False
    
    def create_features(self, property_data):
        """
        Apply feature engineering to prepare for prediction
        
        Parameters:
        -----------
        property_data : dict or DataFrame
            Property data to engineer features for
            
        Returns:
        --------
        DataFrame: Data with engineered features
        """
        # Ensure input is a DataFrame
        df = pd.DataFrame([property_data]) if isinstance(property_data, dict) else property_data.copy()
        
        # Compute derived features - only those needed for the model
        if 'Year Built' in df.columns and 'Property Age' in self.feature_names:
            df['Property Age'] = datetime.now().year - df['Year Built']
        
        if all(col in df.columns for col in ['Bedrooms', 'Total Baths']) and 'Bath_Bedroom_Ratio' in self.feature_names:
            df['Bath_Bedroom_Ratio'] = df['Total Baths'] / (df['Bedrooms'] + 1)
        
        if 'Total SqFt' in df.columns:
            # Log transform
            if 'Log_Total_SqFt' in self.feature_names:
                df['Log_Total_SqFt'] = np.log1p(df['Total SqFt'])
            
            # Interaction with bedrooms
            if 'SqFt_Bedrooms_Interaction' in self.feature_names and 'Bedrooms' in df.columns:
                df['SqFt_Bedrooms_Interaction'] = df['Total SqFt'] * df['Bedrooms']
        
        # Age and baths interaction
        if all(col in df.columns for col in ['Property Age', 'Total Baths']) and 'Age_Baths_Interaction' in self.feature_names:
            df['Age_Baths_Interaction'] = df['Property Age'] * df['Total Baths']
        
        # Handle STR eligibility
        if 'Short Termable' in df.columns:
            if 'Is_STR_Eligible' in self.feature_names:
                df['Is_STR_Eligible'] = df['Short Termable'].map({'Yes': 1, 'No': 0, np.nan: 0})
            
            if 'STR_Status_Known' in self.feature_names:
                df['STR_Status_Known'] = (~df['Short Termable'].isna()).astype(int)
            
            # STR interactions
            if 'Total SqFt' in df.columns and 'STR_SqFt_Interaction' in self.feature_names:
                df['STR_SqFt_Interaction'] = df['Is_STR_Eligible'] * df['Total SqFt']
            
            if 'Bedrooms' in df.columns and 'STR_Bedroom_Interaction' in self.feature_names:
                df['STR_Bedroom_Interaction'] = df['Is_STR_Eligible'] * df['Bedrooms']
            
            if 'Total Baths' in df.columns and 'STR_Bath_Interaction' in self.feature_names:
                df['STR_Bath_Interaction'] = df['Is_STR_Eligible'] * df['Total Baths']
        
        # Handle furnished status
        if 'Furnished' in df.columns and 'Is_Furnished' in self.feature_names:
            df['Is_Furnished'] = df['Furnished'].map({'Yes': 1, 'Partial': 0.5, 'No': 0, np.nan: 0})
            
            if 'Is_STR_Eligible' in df.columns and 'STR_Furnished_Interaction' in self.feature_names:
                df['STR_Furnished_Interaction'] = df['Is_STR_Eligible'] * df['Is_Furnished']
        
        # Handle hotel condominiums
        if 'Book Section' in df.columns and 'Is_Hotel_Condo' in self.feature_names:
            df['Is_Hotel_Condo'] = (df['Book Section'] == 'Hotel Condominium').astype(int)
            
            if 'Total SqFt' in df.columns and 'Hotel_SqFt_Interaction' in self.feature_names:
                df['Hotel_SqFt_Interaction'] = df['Is_Hotel_Condo'] * df['Total SqFt']
            
            if 'Hotel_Price_Factor' in self.feature_names:
                df['Hotel_Price_Factor'] = df['Is_Hotel_Condo'] * 0.5
        
        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in df.columns:
                if feature in self.numeric_features:
                    df[feature] = 0  # Default numeric value
                else:
                    df[feature] = 'Other'  # Default categorical value
        
        return df[self.feature_names]
    
    def predict_property_value(self, property_data):
        """
        Predict the price per square foot and total value of a property
        
        Parameters:
        -----------
        property_data : dict or DataFrame
            Property data to predict on
            
        Returns:
        --------
        dict: Prediction results including price per square foot and total value
        """
        if self.model is None:
            print("Model not loaded. Run load_model() first.")
            return None
        
        try:
            # Apply feature engineering
            property_df = self.create_features(property_data)
            
            # Make prediction
            predicted_price_per_sqft = self.model.predict(property_df)[0]
            
            # Calculate total value
            total_value = None
            if isinstance(property_data, dict) and 'Total SqFt' in property_data:
                total_value = predicted_price_per_sqft * property_data['Total SqFt']
            elif isinstance(property_data, pd.DataFrame) and 'Total SqFt' in property_data.columns:
                total_value = predicted_price_per_sqft * property_data['Total SqFt'].iloc[0]
            
            # Get property details for the result
            if isinstance(property_data, dict):
                property_type = property_data.get('Book Section', 'Unknown')
                str_eligible = property_data.get('Short Termable', 'Unknown')
            else:
                property_type = property_data['Book Section'].iloc[0] if 'Book Section' in property_data.columns else 'Unknown'
                str_eligible = property_data['Short Termable'].iloc[0] if 'Short Termable' in property_data.columns else 'Unknown'
            
            return {
                'price_per_sqft': predicted_price_per_sqft,
                'total_value': total_value,
                'property_type': property_type,
                'str_eligible': str_eligible
            }
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            traceback.print_exc()
            return None
    
    def batch_predict(self, properties):
        """
        Make predictions for multiple properties
        
        Parameters:
        -----------
        properties : DataFrame or list of dicts
            Properties to predict on
            
        Returns:
        --------
        DataFrame: Prediction results for all properties
        """
        if self.model is None:
            print("Model not loaded. Run load_model() first.")
            return None
        
        try:
            # Convert list of dicts to DataFrame if needed
            if isinstance(properties, list) and all(isinstance(p, dict) for p in properties):
                properties_df = pd.DataFrame(properties)
            else:
                properties_df = properties.copy()
            
            # Apply feature engineering
            features_df = self.create_features(properties_df)
            
            # Make predictions
            predictions = self.model.predict(features_df)
            
            # Create results DataFrame
            results = properties_df.copy()
            results['Predicted_Price_Per_SqFt'] = predictions
            
            # Calculate total value if square footage is available
            if 'Total SqFt' in results.columns:
                results['Predicted_Total_Value'] = results['Predicted_Price_Per_SqFt'] * results['Total SqFt']
            
            return results
            
        except Exception as e:
            print(f"Error in batch prediction: {e}")
            traceback.print_exc()
            return None
    
    def compare_property_scenarios(self, base_property, scenarios):
        """
        Compare different property scenarios to see impact on value
        
        Parameters:
        -----------
        base_property : dict
            Base property configuration
        scenarios : list of tuples
            List of (scenario_name, property_changes) tuples
            
        Returns:
        --------
        DataFrame: Comparison of scenarios
        """
        if self.model is None:
            print("Model not loaded. Run load_model() first.")
            return None
        
        results = []
        
        try:
            # Predict base property value
            base_prediction = self.predict_property_value(base_property)
            base_result = {
                'Scenario': 'Base Property',
                'Price_Per_SqFt': base_prediction['price_per_sqft'],
                'Total_Value': base_prediction['total_value']
            }
            results.append(base_result)
            
            # Predict each scenario
            for scenario_name, property_changes in scenarios:
                # Create scenario by updating base property
                scenario_property = base_property.copy()
                scenario_property.update(property_changes)
                
                # Make prediction
                scenario_prediction = self.predict_property_value(scenario_property)
                
                # Calculate differences
                price_diff = scenario_prediction['price_per_sqft'] - base_prediction['price_per_sqft']
                price_diff_pct = (price_diff / base_prediction['price_per_sqft']) * 100
                
                value_diff = None
                value_diff_pct = None
                if scenario_prediction['total_value'] is not None and base_prediction['total_value'] is not None:
                    value_diff = scenario_prediction['total_value'] - base_prediction['total_value']
                    value_diff_pct = (value_diff / base_prediction['total_value']) * 100
                
                # Add to results
                scenario_result = {
                    'Scenario': scenario_name,
                    'Price_Per_SqFt': scenario_prediction['price_per_sqft'],
                    'Total_Value': scenario_prediction['total_value'],
                    'Price_Difference': price_diff,
                    'Price_Difference_Pct': price_diff_pct,
                    'Value_Difference': value_diff,
                    'Value_Difference_Pct': value_diff_pct
                }
                results.append(scenario_result)
            
            # Convert to DataFrame
            return pd.DataFrame(results)
            
        except Exception as e:
            print(f"Error comparing scenarios: {e}")
            traceback.print_exc()
            return None