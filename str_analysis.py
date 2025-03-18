import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
import os

class STRAnalysis:
    """
    Analysis of Short-Term Rental (STR) impact on property values
    """
    
    def __init__(self, df=None, target_column='Price per SqFt', output_dir='output'):
        """
        Initialize STR Analysis with dataset and target column
        
        Parameters:
        -----------
        df : DataFrame, optional
            The dataset to analyze
        target_column : str, default='Price per SqFt'
            The column containing price per square foot data
        output_dir : str, default='output'
            Directory for saving outputs
        """
        self.df = df
        self.target = target_column
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        print("Initialized STR Analysis module")
    
    def load_data(self, df=None, load_from_file=False):
        """
        Load data either from a DataFrame or from file
        
        Parameters:
        -----------
        df : DataFrame, optional
            DataFrame to use for analysis
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
    
    def analyze_basic_str_impact(self, save_results=True):
        """
        Perform a basic analysis of STR impact on property prices
        
        Parameters:
        -----------
        save_results : bool, default=True
            Whether to save results and visualizations
            
        Returns:
        --------
        DataFrame: Statistics on STR vs non-STR properties
        """
        if self.df is None:
            print("No data loaded. Run load_data() first.")
            return None
            
        print("\n===== BASIC STR IMPACT ANALYSIS =====")

        if 'Short Termable' not in self.df.columns:
            print("No STR eligibility data found in 'Short Termable' column.")
            return None

        # Filter valid STR properties
        str_df = self.df[self.df['Short Termable'].notna()]

        if len(str_df) < 10:
            print("Not enough STR data for analysis (need at least 10 properties with STR status).")
            return None

        # Compare STR vs. non-STR price per SqFt
        str_stats = str_df.groupby('Short Termable')[self.target].agg(['count', 'mean', 'median', 'std'])
        print("\nPrice per SqFt by STR Status:")
        print(str_stats)

        # Compute STR premium
        if 'Yes' in str_stats.index and 'No' in str_stats.index:
            yes_median = str_stats.loc['Yes', 'median']
            no_median = str_stats.loc['No', 'median']
            premium_pct = ((yes_median / no_median) - 1) * 100
            premium_amount = yes_median - no_median

            print(f"\nSTR Premium:")
            print(f"STR 'Yes' median price: ${yes_median:.2f}/sqft")
            print(f"STR 'No' median price: ${no_median:.2f}/sqft")
            print(f"Premium: ${premium_amount:.2f}/sqft (+{premium_pct:.1f}%)")
            
            # Save premium data if requested
            if save_results:
                premium_data = pd.DataFrame({
                    'STR_Status': ['Yes', 'No', 'Premium', 'Premium_Pct'],
                    'Value': [yes_median, no_median, premium_amount, premium_pct]
                })
                premium_path = os.path.join(self.output_dir, 'str_premium_basic.csv')
                premium_data.to_csv(premium_path, index=False)
                print(f"STR premium data saved to '{premium_path}'")

        # Create visualization
        if save_results:
            try:
                plt.figure(figsize=(10, 6))
                
                # Create boxplot
                sns.boxplot(x='Short Termable', y=self.target, data=str_df)
                plt.title(f'{self.target} by Short Termable Status')
                plt.grid(True, alpha=0.3, axis='y')
                
                # Save visualization
                viz_path = os.path.join(self.output_dir, 'str_price_impact.png')
                plt.savefig(viz_path)
                plt.close()
                print(f"Saved STR impact visualization to '{viz_path}'")
                
                # Save the stats
                stats_path = os.path.join(self.output_dir, 'str_stats.csv')
                str_stats.to_csv(stats_path)
                print(f"STR statistics saved to '{stats_path}'")
            except Exception as e:
                print(f"Error creating visualization: {e}")

        return str_stats
    
    def run_linear_regression_analysis(self, save_results=True):
        """
        Perform a simple linear regression to isolate STR impact 
        while controlling for basic property characteristics
        
        Parameters:
        -----------
        save_results : bool, default=True
            Whether to save results
            
        Returns:
        --------
        dict: Regression results
        """
        if self.df is None:
            print("No data loaded. Run load_data() first.")
            return None
            
        if 'Short Termable' not in self.df.columns:
            print("No STR eligibility data found.")
            return None
            
        print("\n===== LINEAR REGRESSION STR ANALYSIS =====")
        
        # Filter to properties with STR data
        str_df = self.df[self.df['Short Termable'].notna()]
        
        if len(str_df) < 10:
            print("Not enough STR data for regression analysis.")
            return None
            
        try:
            # Create X and y for regression
            X = pd.get_dummies(str_df[['Total SqFt', 'Bedrooms', 'Total Baths', 'Short Termable']], 
                              columns=['Short Termable'], drop_first=True)
            y = str_df[self.target]
            
            # Add property type if available for better control
            if 'Book Section' in str_df.columns:
                property_type_dummies = pd.get_dummies(str_df['Book Section'], prefix='PropType', drop_first=True)
                X = pd.concat([X, property_type_dummies], axis=1)
            
            # Fit linear model
            model = LinearRegression()
            model.fit(X, y)
            
            # Get coefficient for STR Yes
            str_yes_col = 'Short Termable_Yes'
            if str_yes_col in X.columns:
                str_coef = model.coef_[X.columns.get_loc(str_yes_col)]
                str_effect = (str_coef / y.mean()) * 100  # Percentage effect
                
                print(f"\nControlling for size, bedrooms, bathrooms" + 
                      (" and property type:" if 'Book Section' in str_df.columns else ":"))
                print(f"STR eligibility adds ${str_coef:.2f}/sqft")
                print(f"This represents approximately a {str_effect:.1f}% premium")
                
                # Create coefficient summary
                coef_df = pd.DataFrame({
                    'Feature': X.columns,
                    'Coefficient': model.coef_,
                    'Pct_Impact': model.coef_ / y.mean() * 100
                })
                
                # Show key coefficients
                print("\nKey Regression Coefficients:")
                print(coef_df[coef_df['Feature'] == str_yes_col].to_string(index=False))
                
                for feature in ['Total SqFt', 'Bedrooms', 'Total Baths']:
                    if feature in coef_df['Feature'].values:
                        print(coef_df[coef_df['Feature'] == feature].to_string(index=False))
                
                # Save results if requested
                if save_results:
                    coef_path = os.path.join(self.output_dir, 'str_regression_coefficients.csv')
                    coef_df.to_csv(coef_path, index=False)
                    print(f"Regression coefficients saved to '{coef_path}'")
                    
                    # Save model summary stats
                    r2 = model.score(X, y)
                    n = len(X)
                    k = len(X.columns)
                    summary_df = pd.DataFrame({
                        'Metric': ['R-squared', 'Sample Size', 'Features', 'STR_Coefficient', 'STR_Impact_Pct'],
                        'Value': [r2, n, k, str_coef, str_effect]
                    })
                    summary_path = os.path.join(self.output_dir, 'str_regression_summary.csv')
                    summary_df.to_csv(summary_path, index=False)
                    print(f"Regression summary saved to '{summary_path}'")
                
                return {
                    'STR_Coefficient': str_coef,
                    'STR_Impact_Pct': str_effect,
                    'R_squared': r2
                }
            else:
                print(f"Could not find '{str_yes_col}' in regression columns")
                return None
                
        except Exception as e:
            print(f"Error in regression analysis: {e}")
            return None
    
    def compute_model_based_str_premium(self, model=None, feature_names=None, model_path=None):
        """
        Use a trained model to estimate STR impact on a median property
        
        Parameters:
        -----------
        model : trained model, optional
            Pretrained model to use for predictions
        feature_names : list, optional
            List of feature names for prediction
        model_path : str, optional
            Path to load saved model from
            
        Returns:
        --------
        dict: STR premium results
        """
        print("\n===== MODEL-BASED STR IMPACT ANALYSIS =====")
        
        if self.df is None:
            print("No data loaded. Run load_data() first.")
            return None
        
        # Load model if provided path
        if model is None and model_path is not None:
            try:
                import joblib
                model_data = joblib.load(model_path)
                model = model_data['model']
                feature_names = model_data['feature_names']
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                return None
        
        if model is None:
            print("No model provided. Pass a trained model or model_path.")
            return None
            
        if feature_names is None:
            print("No feature names provided.")
            return None
        
        if 'Is_STR_Eligible' not in self.df.columns:
            print("STR eligibility data missing in dataset.")
            return None

        try:
            # Generate synthetic median property for prediction
            median_property = {}
            for feature in feature_names:
                if feature in self.df.columns and feature != 'Is_STR_Eligible' and not feature.startswith('STR_'):
                    if self.df[feature].dtype in [np.int64, np.float64, int, float]:
                        median_property[feature] = self.df[feature].median()
                    else:
                        # For categorical features, use the most common value
                        median_property[feature] = self.df[feature].mode().iloc[0]
            
            # Create two versions: one STR-eligible, one not
            property_no_str = median_property.copy()
            property_str = median_property.copy()
            
            # Configure for non-STR
            property_no_str['Is_STR_Eligible'] = 0
            for col in feature_names:
                if col.startswith('STR_') and col != 'STR_Status_Known':
                    property_no_str[col] = 0
            property_no_str['STR_Status_Known'] = 1
            
            # Configure for STR
            property_str['Is_STR_Eligible'] = 1
            for col in feature_names:
                if col.startswith('STR_') and col != 'STR_Status_Known':
                    base_feature = col.replace('STR_', '').replace('_Interaction', '')
                    if base_feature in property_str:
                        property_str[col] = property_str[base_feature]
                    else:
                        property_str[col] = 0
            property_str['STR_Status_Known'] = 1
            
            # Convert to DataFrame
            combined_df = pd.DataFrame([property_no_str, property_str])
            
            # Fill in any missing features
            for feature in feature_names:
                if feature not in combined_df.columns:
                    if feature in self.df.columns and self.df[feature].dtype in [np.int64, np.float64, int, float]:
                        combined_df[feature] = self.df[feature].median()
                    else:
                        combined_df[feature] = 0  # Default value
            
            # Make predictions
            predicted_prices = model.predict(combined_df[feature_names])
            price_no_str, price_str = predicted_prices
            
            # Compute STR premium
            str_premium = price_str - price_no_str
            str_premium_pct = (price_str / price_no_str - 1) * 100
            
            print(f"\nModel-predicted STR premium for median property:")
            print(f"- Non-STR property: ${price_no_str:.2f}/sqft")
            print(f"- STR-eligible property: ${price_str:.2f}/sqft")
            print(f"- STR premium: ${str_premium:.2f}/sqft (+{str_premium_pct:.1f}%)")
            
            # Save results
            results = {
                'Non_STR_Price': price_no_str,
                'STR_Price': price_str,
                'STR_Premium': str_premium,
                'STR_Premium_Pct': str_premium_pct
            }
            
            result_df = pd.DataFrame([results])
            result_path = os.path.join(self.output_dir, 'str_model_premium.csv')
            result_df.to_csv(result_path, index=False)
            print(f"Model-based STR premium saved to '{result_path}'")
            
            return results
            
        except Exception as e:
            print(f"Error in model-based STR analysis: {e}")
            return None
    
    def bootstrap_str_premium(self, model=None, feature_names=None, n_bootstraps=1000, model_path=None):
        """
        Compute confidence intervals for STR premium using bootstrapping
        
        Parameters:
        -----------
        model : trained model, optional
            Pretrained model to use for predictions
        feature_names : list, optional  
            List of feature names for prediction
        n_bootstraps : int, default=1000
            Number of bootstrap iterations
        model_path : str, optional
            Path to load saved model from
            
        Returns:
        --------
        dict: Bootstrap results with confidence intervals
        """
        print(f"\n===== BOOTSTRAP STR PREMIUM ANALYSIS (n={n_bootstraps}) =====")
        
        if self.df is None:
            print("No data loaded. Run load_data() first.")
            return None
            
        # Load model if provided path
        if model is None and model_path is not None:
            try:
                import joblib
                model_data = joblib.load(model_path)
                model = model_data['model']
                feature_names = model_data['feature_names']
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                return None
        
        if model is None:
            print("No model provided. Pass a trained model or model_path.")
            return None
            
        if feature_names is None:
            print("No feature names provided.")
            return None
        
        if 'Is_STR_Eligible' not in self.df.columns:
            print("STR eligibility data missing in dataset.")
            return None
            
        try:
            # Check that necessary features exist
            missing_features = [f for f in feature_names if f not in self.df.columns]
            if missing_features:
                print(f"Missing features for bootstrapping: {missing_features}")
                return None
                
            print(f"Running {n_bootstraps} bootstrap iterations...")
            
            # Store bootstrap results
            premiums = []
            premium_pcts = []
            
            # Run bootstrap iterations
            for i in range(n_bootstraps):
                # Status update for long-running operations
                if i % 100 == 0 and i > 0:
                    print(f"Completed {i} bootstrap iterations...")
                    
                # Resample with replacement
                boot_sample = resample(self.df, replace=True, n_samples=len(self.df))
                
                # Create median property from this bootstrap sample
                median_property = {}
                for feature in feature_names:
                    if feature != 'Is_STR_Eligible' and not feature.startswith('STR_'):
                        if boot_sample[feature].dtype in [np.int64, np.float64, int, float]:
                            median_property[feature] = boot_sample[feature].median()
                        else:
                            # For categorical features, use most common value
                            median_property[feature] = boot_sample[feature].mode().iloc[0]
                
                # Create non-STR and STR versions
                property_no_str = median_property.copy()
                property_str = median_property.copy()
                
                # Configure for non-STR
                property_no_str['Is_STR_Eligible'] = 0
                for col in feature_names:
                    if col.startswith('STR_') and col != 'STR_Status_Known':
                        property_no_str[col] = 0
                property_no_str['STR_Status_Known'] = 1
                
                # Configure for STR
                property_str['Is_STR_Eligible'] = 1
                for col in feature_names:
                    if col.startswith('STR_') and col != 'STR_Status_Known':
                        base_feature = col.replace('STR_', '').replace('_Interaction', '')
                        if base_feature in property_str:
                            property_str[col] = property_str[base_feature]
                        else:
                            property_str[col] = 0
                property_str['STR_Status_Known'] = 1
                
                # Add any missing features with default values
                for feature in feature_names:
                    if feature not in property_no_str:
                        property_no_str[feature] = 0
                    if feature not in property_str:
                        property_str[feature] = 0
                
                # Convert to DataFrame
                combined_df = pd.DataFrame([property_no_str, property_str])
                
                # Make predictions
                predicted_prices = model.predict(combined_df[feature_names])
                price_no_str, price_str = predicted_prices
                
                # Calculate premium
                premium = price_str - price_no_str
                premium_pct = (price_str / price_no_str - 1) * 100
                
                # Store results
                premiums.append(premium)
                premium_pcts.append(premium_pct)
            
            # Compute confidence intervals
            lower_bound = np.percentile(premiums, 2.5)
            upper_bound = np.percentile(premiums, 97.5)
            mean_premium = np.mean(premiums)
            
            lower_bound_pct = np.percentile(premium_pcts, 2.5)
            upper_bound_pct = np.percentile(premium_pcts, 97.5)
            mean_premium_pct = np.mean(premium_pcts)
            
            print(f"\nBootstrap Results (95% Confidence Interval):")
            print(f"STR Dollar Premium: ${mean_premium:.2f}/sqft (95% CI: ${lower_bound:.2f} to ${upper_bound:.2f})")
            print(f"STR Percent Premium: {mean_premium_pct:.1f}% (95% CI: {lower_bound_pct:.1f}% to {upper_bound_pct:.1f}%)")
            
            # Save results
            results = {
                'Mean_Premium': mean_premium,
                'Lower_Bound': lower_bound,
                'Upper_Bound': upper_bound,
                'Mean_Premium_Pct': mean_premium_pct,
                'Lower_Bound_Pct': lower_bound_pct,
                'Upper_Bound_Pct': upper_bound_pct,
                'Bootstrap_Iterations': n_bootstraps
            }
            
            # Also save the full distribution for potential visualization
            bootstrap_df = pd.DataFrame({
                'Premium': premiums,
                'Premium_Pct': premium_pcts
            })
            bootstrap_path = os.path.join(self.output_dir, 'str_bootstrap_distribution.csv')
            bootstrap_df.to_csv(bootstrap_path, index=False)
            
            result_df = pd.DataFrame([results])
            result_path = os.path.join(self.output_dir, 'str_bootstrap_results.csv')
            result_df.to_csv(result_path, index=False)
            print(f"Bootstrap results saved to '{result_path}'")
            
            # Create visualization of bootstrap distribution
            plt.figure(figsize=(12, 6))
            
            # Create price premium histogram
            plt.subplot(1, 2, 1)
            plt.hist(premiums, bins=30, alpha=0.7)
            plt.axvline(mean_premium, color='red', linestyle='dashed', linewidth=2)
            plt.axvline(lower_bound, color='green', linestyle='dotted', linewidth=2)
            plt.axvline(upper_bound, color='green', linestyle='dotted', linewidth=2)
            plt.title('Bootstrap Distribution: STR Premium ($/sqft)')
            plt.xlabel('Premium ($/sqft)')
            plt.ylabel('Frequency')
            
            # Create percentage premium histogram
            plt.subplot(1, 2, 2)
            plt.hist(premium_pcts, bins=30, alpha=0.7)
            plt.axvline(mean_premium_pct, color='red', linestyle='dashed', linewidth=2)
            plt.axvline(lower_bound_pct, color='green', linestyle='dotted', linewidth=2)
            plt.axvline(upper_bound_pct, color='green', linestyle='dotted', linewidth=2)
            plt.title('Bootstrap Distribution: STR Premium (%)')
            plt.xlabel('Premium (%)')
            plt.ylabel('Frequency')
            
            plt.tight_layout()
            viz_path = os.path.join(self.output_dir, 'str_bootstrap_distribution.png')
            plt.savefig(viz_path)
            plt.close()
            print(f"Bootstrap distribution visualization saved to '{viz_path}'")
            
            return results
            
        except Exception as e:
            print(f"Error in bootstrap analysis: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_all_analyses(self, model=None, feature_names=None, model_path=None, n_bootstraps=100):
        """
        Run all STR analyses in sequence
        
        Parameters:
        -----------
        model : trained model, optional
            Model for prediction-based analyses
        feature_names : list, optional
            Feature names used by the model
        model_path : str, optional
            Path to load model from
        n_bootstraps : int, default=100
            Number of bootstrap iterations (smaller default for quicker execution)
        """
        print("\n===== RUNNING ALL STR ANALYSES =====")
        
        if self.df is None:
            print("No data loaded. Run load_data() first.")
            return
            
        # Run basic STR comparison
        self.analyze_basic_str_impact()
        
        # Run linear regression analysis
        self.run_linear_regression_analysis()
        
        # Run model-based analyses if model is available
        if model is not None or model_path is not None:
            self.compute_model_based_str_premium(model, feature_names, model_path)
            self.bootstrap_str_premium(model, feature_names, n_bootstraps, model_path)
        
        print("\n===== STR ANALYSES COMPLETE =====")