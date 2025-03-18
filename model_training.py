import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class RealEstateModel:
    """
    Training and evaluation of predictive models for real estate values
    """
    
    def __init__(self, df=None, target_column='Price per SqFt', output_dir='output'):
        """
        Initialize model class with dataset
        
        Parameters:
        -----------
        df : DataFrame, optional
            The dataset to train on
        target_column : str, default='Price per SqFt'
            The column containing price per square foot data
        output_dir : str, default='output'
            Directory for saving outputs
        """
        self.df = df
        self.target = target_column
        self.output_dir = output_dir
        self.model = None
        self.feature_names = []
        self.numeric_features = []
        self.categorical_features = []
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        print("Initialized Real Estate Model module")
    
    def load_data(self, df=None, load_from_file=False):
        """
        Load data either from a DataFrame or from file
        
        Parameters:
        -----------
        df : DataFrame, optional
            DataFrame to use for training
        load_from_file : bool, default=False
            Whether to load from featured_data.csv
        """
        if df is not None:
            self.df = df.copy()
            print(f"Using provided DataFrame with shape: {self.df.shape}")
            return True
            
        elif load_from_file:
            try:
                # Try to load featured data first, fall back to processed data
                featured_path = os.path.join(self.output_dir, 'featured_data.csv')
                if os.path.exists(featured_path):
                    self.df = pd.read_csv(featured_path)
                    print(f"Loaded featured data from {featured_path}, shape: {self.df.shape}")
                else:
                    processed_path = os.path.join(self.output_dir, 'processed_data.csv')
                    self.df = pd.read_csv(processed_path)
                    print(f"Loaded processed data from {processed_path}, shape: {self.df.shape}")
                return True
            except Exception as e:
                print(f"Error loading data: {e}")
                return False
        else:
            print("No data provided. Either pass a DataFrame or set load_from_file=True")
            return False
    
    def select_features(self, feature_list=None):
        """
        Identify numeric and categorical features for the model
        
        Parameters:
        -----------
        feature_list : list, optional
            Custom list of features to use (if None, uses default feature lists)
            
        Returns:
        --------
        list: Selected feature names
        """
        if self.df is None:
            print("No data loaded. Run load_data() first.")
            return None
            
        print("\n===== FEATURE SELECTION =====")
        
        if feature_list is not None:
            # Use provided feature list, separating numeric and categorical
            self.feature_names = [f for f in feature_list if f in self.df.columns]
            self.numeric_features = [f for f in self.feature_names if self.df[f].dtype in ['int64', 'float64', int, float]]
            self.categorical_features = [f for f in self.feature_names if f not in self.numeric_features]
            
            print(f"Using provided feature list: {len(self.feature_names)} features")
        else:
            # Default numeric features
            self.numeric_features = [
                'Total SqFt', 'Bedrooms', 'Total Baths', 'Year Built', 
                'Property Age', 'Bath_Bedroom_Ratio', 'Log_Total_SqFt',
                'SqFt_Bedrooms_Interaction', 'Age_Baths_Interaction'
            ]
            
            # Add hotel condo features if available
            hotel_features = ['Is_Hotel_Condo', 'Hotel_SqFt_Interaction', 'Hotel_Price_Factor']
            self.numeric_features.extend([f for f in hotel_features if f in self.df.columns])
            
            # STR-related features
            str_features = [
                'Is_STR_Eligible', 'STR_SqFt_Interaction', 'STR_Bedroom_Interaction', 
                'STR_Bath_Interaction', 'STR_Status_Known', 'STR_Furnished_Interaction',
                'STR_Gondola_Interaction'
            ]
            self.numeric_features.extend([f for f in str_features if f in self.df.columns])
            
            # Additional features if available
            extra_features = [
                'Is_Furnished', 'Gondola_Proximity', 'Bath Score', 'Property_Type_Calibration',
                'Extra_Baths', 'Has_Extra_Baths', 'Baths_Per_Bedroom', 'Price_Trend_Ratio'
            ]
            self.numeric_features.extend([f for f in extra_features if f in self.df.columns])
            
            # Keep only columns that actually exist in the dataframe
            self.numeric_features = [f for f in self.numeric_features if f in self.df.columns]
            
            # Categorical Features
            self.categorical_features = ['Book Section'] if 'Book Section' in self.df.columns else []
            
            # Combine feature lists
            self.feature_names = self.numeric_features + self.categorical_features
        
        print(f"Selected {len(self.feature_names)} features:")
        print(f"- {len(self.numeric_features)} numeric features")
        print(f"- {len(self.categorical_features)} categorical features")
        
        # Log feature counts by type
        str_count = len([f for f in self.numeric_features if 'STR' in f])
        bath_count = len([f for f in self.numeric_features if 'Bath' in f])
        
        if str_count > 0:
            print(f"- {str_count} STR-related features")
        if bath_count > 0:
            print(f"- {bath_count} bath-related features")
        
        # Save feature list
        feature_df = pd.DataFrame({
            'FeatureName': self.feature_names,
            'Type': ['Numeric' if f in self.numeric_features else 'Categorical' for f in self.feature_names]
        })
        feature_path = os.path.join(self.output_dir, 'model_features.csv')
        feature_df.to_csv(feature_path, index=False)
        print(f"Feature list saved to '{feature_path}'")
        
        return self.feature_names
    
    def handle_missing_values(self):
        """
        Impute missing values in the features
        
        Returns:
        --------
        bool: Success status
        """
        if self.df is None:
            print("No data loaded. Run load_data() first.")
            return False
            
        if not self.feature_names:
            print("No features selected. Run select_features() first.")
            return False
            
        print("\n===== HANDLING MISSING VALUES =====")
        
        # Check for missing values in selected features
        missing_values = {col: self.df[col].isna().sum() for col in self.feature_names if col in self.df.columns}
        missing_values = {col: count for col, count in missing_values.items() if count > 0}
        
        if missing_values:
            print(f"Found missing values in {len(missing_values)} features:")
            for col, count in missing_values.items():
                print(f"- {col}: {count} missing values")
                
                # Fill missing values
                if col in self.numeric_features:
                    median_value = self.df[col].median()
                    self.df[col] = self.df[col].fillna(median_value)
                    print(f"  Filled with median: {median_value:.2f}")
                else:
                    mode_value = self.df[col].mode().iloc[0] if not self.df[col].mode().empty else 'Other'
                    self.df[col] = self.df[col].fillna(mode_value)
                    print(f"  Filled with mode: {mode_value}")
        else:
            print("No missing values found in the selected features.")
            
        return True
    
    def build_model(self, model_type='random_forest', hyperparams=None, test_size=0.2, save_model=True):
        """
        Train and evaluate a predictive model
        
        Parameters:
        -----------
        model_type : str, default='random_forest'
            Type of model to train ('random_forest' or 'linear_regression')
        hyperparams : dict, optional
            Hyperparameters for the model
        test_size : float, default=0.2
            Proportion of data to use for testing
        save_model : bool, default=True
            Whether to save the trained model
            
        Returns:
        --------
        dict: Model performance metrics
        """
        if self.df is None:
            print("No data loaded. Run load_data() first.")
            return None
            
        print("\n===== MODEL TRAINING =====")
        
        # Default hyperparameters if none provided
        if hyperparams is None:
            if model_type == 'random_forest':
                hyperparams = {
                    'n_estimators': 200,
                    'max_depth': 20,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'max_features': 'sqrt'
                }
            else:
                hyperparams = {}
        
        # Prepare features and target
        if not self.feature_names:
            self.select_features()
            
        self.handle_missing_values()
        
        # Extract features and target
        X = self.df[self.feature_names]
        y = self.df[self.target]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        print(f"Training on {X_train.shape[0]} samples, Testing on {X_test.shape[0]} samples")
        
        # Create preprocessing pipeline
        start_time = time.time()
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Only include categorical pipeline if we have categorical features
        if self.categorical_features:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, self.numeric_features),
                    ('cat', categorical_transformer, self.categorical_features)
                ]
            )
        else:
            # Only numeric features
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, self.numeric_features)
                ]
            )
        
        # Select model type
        if model_type == 'random_forest':
            print(f"Training Random Forest with hyperparameters: {hyperparams}")
            model = RandomForestRegressor(
                n_estimators=hyperparams.get('n_estimators', 200),
                max_depth=hyperparams.get('max_depth', 20),
                min_samples_split=hyperparams.get('min_samples_split', 5),
                min_samples_leaf=hyperparams.get('min_samples_leaf', 2),
                max_features=hyperparams.get('max_features', 'sqrt'),
                random_state=42,
                n_jobs=-1  # Use all cores
            )
        elif model_type == 'linear_regression':
            print("Training Linear Regression model")
            model = LinearRegression()
        else:
            print(f"Unknown model type: {model_type}. Using Random Forest.")
            model = RandomForestRegressor(random_state=42)
        
        # Create pipeline
        self.model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        # Train model
        print("Training model...")
        self.model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        print(f"Model training completed in {training_time:.1f} seconds")
        
        # Evaluate performance
        print("\n===== MODEL EVALUATION =====")
        
        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Calculate metrics
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)
        
        print(f"Training set - MAE: ${train_mae:.2f}, RMSE: ${train_rmse:.2f}, R²: {train_r2:.4f}")
        print(f"Testing set - MAE: ${test_mae:.2f}, RMSE: ${test_rmse:.2f}, R²: {test_r2:.4f}")
        
        # Save model if requested
        if save_model:
            model_filename = os.path.join(self.output_dir, f"{model_type}_model.pkl")
            
            # Save model with metadata
            joblib.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'numeric_features': self.numeric_features,
                'categorical_features': self.categorical_features,
                'target': self.target,
                'model_type': model_type,
                'hyperparams': hyperparams,
                'training_time': training_time,
                'metrics': {
                    'train_mae': train_mae,
                    'train_rmse': train_rmse,
                    'train_r2': train_r2,
                    'test_mae': test_mae,
                    'test_rmse': test_rmse,
                    'test_r2': test_r2
                }
            }, model_filename)
            
            print(f"Model saved to '{model_filename}'")
            
            # Create evaluation visualization
            self._create_evaluation_plots(y_test, y_test_pred)
        
        # Return metrics
        return {
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_r2': train_r2,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'training_time': training_time
        }
    
    def _create_evaluation_plots(self, y_true, y_pred):
        """
        Create evaluation plots for model performance
        
        Parameters:
        -----------
        y_true : array
            True target values
        y_pred : array
            Predicted target values
        """
        try:
            plt.figure(figsize=(12, 8))
            
            # Create scatter plot of actual vs predicted values
            plt.subplot(1, 2, 1)
            plt.scatter(y_true, y_pred, alpha=0.5)
            
            # Add perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            plt.xlabel('Actual Price per SqFt')
            plt.ylabel('Predicted Price per SqFt')
            plt.title('Actual vs. Predicted Prices')
            plt.grid(True, alpha=0.3)
            
            # Create residual plot
            residuals = y_pred - y_true
            
            plt.subplot(1, 2, 2)
            plt.scatter(y_pred, residuals, alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted Price per SqFt')
            plt.ylabel('Residuals')
            plt.title('Residual Plot')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save the figure
            eval_path = os.path.join(self.output_dir, 'model_evaluation.png')
            plt.savefig(eval_path)
            plt.close()
            print(f"Model evaluation visualizations saved to '{eval_path}'")
            
        except Exception as e:
            print(f"Error creating evaluation plots: {e}")
    
    def analyze_feature_importance(self):
        """
        Analyze and visualize feature importance
        
        Returns:
        --------
        DataFrame: Feature importance results
        """
        if self.model is None:
            print("Model not trained yet. Run build_model() first.")
            return None
            
        print("\n===== FEATURE IMPORTANCE ANALYSIS =====")
        
        try:
            # Check if model has feature importances (RandomForest does)
            if hasattr(self.model.named_steps['regressor'], 'feature_importances_'):
                importances = self.model.named_steps['regressor'].feature_importances_
                
                # Get feature names after preprocessing if possible
                feature_names = self.feature_names
                
                # Create importance dataframe
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                })
                
                # Sort by importance and select top features
                top_features = importance_df.sort_values('Importance', ascending=False)
                
                print("\nTop 15 Feature Importances:")
                for i, row in top_features.head(15).iterrows():
                    print(f"- {row['Feature']}: {row['Importance']:.4f}")
                
                # Save feature importance
                importance_path = os.path.join(self.output_dir, 'feature_importance.csv')
                importance_df.to_csv(importance_path, index=False)
                print(f"Feature importance saved to '{importance_path}'")
                
                # Create visualization
                plt.figure(figsize=(12, 8))
                sns.barplot(x='Importance', y='Feature', data=top_features.head(15))
                plt.title('Top 15 Feature Importances')
                plt.tight_layout()
                
                # Save visualization
                viz_path = os.path.join(self.output_dir, 'feature_importance.png')
                plt.savefig(viz_path)
                plt.close()
                print(f"Feature importance visualization saved to '{viz_path}'")
                
                return importance_df
                
            elif hasattr(self.model.named_steps['regressor'], 'coef_'):
                # Linear model coefficients
                coefficients = self.model.named_steps['regressor'].coef_
                
                # Map to feature names if possible
                feature_names = self.feature_names[:len(coefficients)]
                
                # Create coefficient dataframe
                coef_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Coefficient': coefficients
                })
                
                # Sort by absolute coefficient value
                coef_df['Abs_Coefficient'] = np.abs(coef_df['Coefficient'])
                top_features = coef_df.sort_values('Abs_Coefficient', ascending=False)
                
                print("\nTop 15 Feature Coefficients:")
                for i, row in top_features.head(15).iterrows():
                    print(f"- {row['Feature']}: {row['Coefficient']:.4f}")
                
                # Save feature coefficients
                coef_path = os.path.join(self.output_dir, 'feature_coefficients.csv')
                coef_df.to_csv(coef_path, index=False)
                print(f"Feature coefficients saved to '{coef_path}'")
                
                # Create visualization
                plt.figure(figsize=(12, 8))
                sns.barplot(x='Coefficient', y='Feature', data=top_features.head(15))
                plt.title('Top 15 Feature Coefficients')
                plt.tight_layout()
                
                # Save visualization
                viz_path = os.path.join(self.output_dir, 'feature_coefficients.png')
                plt.savefig(viz_path)
                plt.close()
                print(f"Feature coefficients visualization saved to '{viz_path}'")
                
                return coef_df
            else:
                print("Model doesn't provide feature importance or coefficients")
                return None
        except Exception as e:
            print(f"Error analyzing feature importance: {e}")
            return None
            
    def load_model(self, model_path=None):
        """
        Load a trained model from file
        
        Parameters:
        -----------
        model_path : str, optional
            Path to model file (if None, uses default path)
            
        Returns:
        --------
        bool: Success status
        """
        if model_path is None:
            # Try to find a model in the output directory
            for model_type in ['random_forest', 'linear_regression']:
                potential_path = os.path.join(self.output_dir, f"{model_type}_model.pkl")
                if os.path.exists(potential_path):
                    model_path = potential_path
                    break
                    
            if model_path is None:
                print("No model file found in output directory")
                return False
        
        try:
            print(f"Loading model from {model_path}")
            model_data = joblib.load(model_path)
            
            # Load model and metadata
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.numeric_features = model_data['numeric_features']
            self.categorical_features = model_data['categorical_features']
            self.target = model_data.get('target', self.target)
            
            # Show model info
            model_type = model_data.get('model_type', 'unknown')
            metrics = model_data.get('metrics', {})
            
            print(f"Loaded {model_type} model with {len(self.feature_names)} features")
            if metrics:
                print(f"Test set performance - MAE: ${metrics.get('test_mae', 0):.2f}, R²: {metrics.get('test_r2', 0):.4f}")
                
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, property_data):
        """
        Make predictions for a new property
        
        Parameters:
        -----------
        property_data : dict or DataFrame
            Property features to predict on
            
        Returns:
        --------
        dict: Prediction results
        """
        if self.model is None:
            print("No model loaded. Run build_model() or load_model() first.")
            return None
        
        try:
            # Convert to DataFrame if dict
            if isinstance(property_data, dict):
                property_df = pd.DataFrame([property_data])
            else:
                property_df = property_data.copy()
                
            # Check that required features are present
            missing_features = [f for f in self.feature_names if f not in property_df.columns]
            
            # Fill missing features with defaults
            for feature in missing_features:
                if feature in self.numeric_features:
                    # Try to use median from training data if available
                    if self.df is not None and feature in self.df.columns:
                        property_df[feature] = self.df[feature].median()
                    else:
                        property_df[feature] = 0  # Default numeric value
                else:
                    # For categorical features, use the most common value or 'Other'
                    if self.df is not None and feature in self.df.columns:
                        property_df[feature] = self.df[feature].mode().iloc[0]
                    else:
                        property_df[feature] = 'Other'
            
            # Make prediction
            predicted_price = self.model.predict(property_df[self.feature_names])
            
            # Calculate total value if square footage is available
            total_value = None
            if 'Total SqFt' in property_df.columns:
                total_value = predicted_price * property_df['Total SqFt'].values
                
            # Create result dictionary
            results = {}
            for i in range(len(predicted_price)):
                results[i] = {
                    'price_per_sqft': predicted_price[i],
                    'total_value': total_value[i] if total_value is not None else None,
                }
                # Add key property attributes to results
                for key in ['Bedrooms', 'Total Baths', 'Total SqFt', 'Year Built', 'Book Section']:
                    if key in property_df.columns:
                        results[i][key] = property_df[key].iloc[i]
            
            return results
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None