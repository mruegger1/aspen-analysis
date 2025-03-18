import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from functools import lru_cache

class BedBathAnalysis:
    """
    Analysis of the impact of bedroom and bathroom configurations on property values
    """
    
    def __init__(self, df=None, target_column='Price per SqFt', output_dir='output'):
        """
        Initialize with the dataset and target price column
        
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
        self.results = []  # Store premium calculations
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        print("Initialized Bed-Bath Analysis module")
    
    def load_data(self, df=None, load_from_file=False):
        """
        Load data either from a DataFrame or from file
        
        Parameters:
        -----------
        df : DataFrame, optional
            DataFrame to use for analysis
        load_from_file : bool, default=False
            Whether to load from featured_data.csv or processed_data.csv
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
            
        Returns:
        --------
        str: Category label
        """
        try:
            # Convert to appropriate types
            beds = int(beds) if pd.notna(beds) else 0
            baths = float(baths) if pd.notna(baths) else 0
            
            # Cap at 7+ bedrooms to avoid sparse categories
            if beds > 7:
                beds = "7+"
            
            # Create bath categories using a more readable approach
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
    
    def compute_bed_bath_premium(self):
        """
        Calculate the premium of different bathroom configurations for each bedroom count
        
        Returns:
        --------
        DataFrame: DataFrame with premium calculations
        """
        if self.df is None:
            print("No data loaded. Run load_data() first.")
            return None
            
        print("\n===== BED-BATH PREMIUM ANALYSIS =====")

        # Exclude hotel condominiums if flagged
        analysis_df = self.df[self.df['Is_Hotel_Condo'] == 0] if 'Is_Hotel_Condo' in self.df.columns else self.df
        print(f"Analyzing {len(analysis_df)} properties")
        
        # Group by bedrooms and calculate stats for each bathroom count
        results = []
        
        # Process each bedroom count from 1 to 7
        for bed_count in range(1, 8):
            # Filter to this bedroom count
            bed_df = analysis_df[analysis_df['Bedrooms'] == bed_count]
            
            if len(bed_df) < 5:  # Skip if too few samples
                continue
                
            # Analyze bathroom counts
            bath_groups = bed_df.groupby(
                bed_df['Total Baths'].apply(lambda x: round(x * 2) / 2)
            )
            bath_stats = bath_groups[self.target].agg(['median', 'mean', 'count'])
            bath_stats = bath_stats.sort_index()
            
            if len(bath_stats) > 1:
                # Get the baseline (usually the minimum bathroom count)
                baseline_bath = bath_stats.index.min()
                baseline_price = bath_stats.loc[baseline_bath, 'median']
                
                # Calculate premium for each additional bathroom
                for bath_count in bath_stats.index:
                    if bath_count == baseline_bath:
                        continue
                        
                    current_price = bath_stats.loc[bath_count, 'median']
                    premium = current_price - baseline_price
                    premium_pct = (current_price / baseline_price - 1) * 100
                    
                    # Add to results
                    results.append({
                        'Bedrooms': bed_count,
                        'Bathrooms': bath_count,
                        'Sample_Size': bath_stats.loc[bath_count, 'count'],
                        'Median_Price': current_price,
                        'Baseline_Price': baseline_price,
                        'Premium': premium,
                        'Premium_Pct': premium_pct
                    })
            
            # Print summary for this bedroom count
            print(f"\n{bed_count} Bedroom Properties:")
            print(f"Total samples: {len(bed_df)}")
            print("Price by bathroom count:")
            
            for bath, row in bath_stats.iterrows():
                print(f"- {bath} bath: ${row['median']:.2f}/sqft (n={int(row['count'])})")
            
            # If we found premiums, show them
            bed_results = [r for r in results if r['Bedrooms'] == bed_count]
            if bed_results:
                print("Bath premiums (vs. minimum bath count):")
                for r in bed_results:
                    print(f"- {r['Bathrooms']} bath: +${r['Premium']:.2f}/sqft (+{r['Premium_Pct']:.1f}%)")
        
        # Convert to DataFrame and save
        if results:
            premium_df = pd.DataFrame(results)
            # Save premium data
            premium_path = os.path.join(self.output_dir, 'bathroom_premium_data.csv')
            premium_df.to_csv(premium_path, index=False)
            print(f"\nBathroom premium data saved to '{premium_path}'")
            self.premium_df = premium_df
            return premium_df
        else:
            print("No premium results found")
            return None
    
    def plot_bathroom_premium(self, premium_df=None):
        """
        Visualize bathroom price premium by bedroom count
        
        Parameters:
        -----------
        premium_df : DataFrame, optional
            DataFrame with premium calculations (if None, uses result from compute_bed_bath_premium)
        """
        if premium_df is None:
            if hasattr(self, 'premium_df'):
                premium_df = self.premium_df
            else:
                print("No premium data available. Run compute_bed_bath_premium() first.")
                return
        
        if len(premium_df) == 0:
            print("No premium data to plot")
            return
            
        plt.figure(figsize=(12, 8))
        for bed_count in sorted(premium_df['Bedrooms'].unique()):
            bed_data = premium_df[premium_df['Bedrooms'] == bed_count]
            plt.plot(bed_data['Bathrooms'], bed_data['Premium_Pct'], 
                    marker='o', label=f'{bed_count} Bedroom')
        
        plt.xlabel('Bathroom Count')
        plt.ylabel('Price Premium (%)')
        plt.title('Bathroom Premium by Property Size')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        viz_path = os.path.join(self.output_dir, 'bathroom_premium.png')
        plt.savefig(viz_path)
        plt.close()
        print(f"Bathroom premium visualization saved to '{viz_path}'")
    
    def analyze_bath_richness(self, create_features=True):
        """
        Analyze and categorize bathroom richness
        
        Parameters:
        -----------
        create_features : bool, default=True
            Whether to create bath richness features if they don't exist
        """
        if self.df is None:
            print("No data loaded. Run load_data() first.")
            return
            
        print("\n===== BATHROOM RICHNESS ANALYSIS =====")
        
        # Create bath per bedroom metrics if it doesn't exist
        if 'Baths_Per_Bedroom' not in self.df.columns:
            if create_features:
                print("Creating bath richness features")
                self.df['Baths_Per_Bedroom'] = self.df['Total Baths'] / self.df['Bedrooms'].apply(lambda x: max(x, 1))
                
                # Create bath richness categories
                conditions = [
                    (self.df['Baths_Per_Bedroom'] < 1),
                    (self.df['Baths_Per_Bedroom'] == 1),
                    (self.df['Baths_Per_Bedroom'] > 1) & (self.df['Baths_Per_Bedroom'] <= 1.5),
                    (self.df['Baths_Per_Bedroom'] > 1.5) & (self.df['Baths_Per_Bedroom'] <= 2),
                    (self.df['Baths_Per_Bedroom'] > 2)
                ]
                
                values = ['LowBath', 'EqualBath', 'ModerateExtraBath', 'HighExtraBath', 'VeryHighExtraBath']
                self.df['Bath_Richness'] = np.select(conditions, values, default='Unknown')
            else:
                print("Bath richness features not found and create_features=False")
                return
                
        # Filter non-hotel properties if flagged
        analysis_df = self.df[self.df['Is_Hotel_Condo'] == 0] if 'Is_Hotel_Condo' in self.df.columns else self.df
            
        # Analyze the impact of bath richness
        richness_stats = analysis_df.groupby('Bath_Richness')[self.target].agg(['median', 'mean', 'count'])
        
        print("\nImpact of Bathroom Richness:")
        for richness, row in richness_stats.iterrows():
            if row['count'] > 5:  # Only show if enough samples
                print(f"- {richness}: ${row['median']:.2f}/sqft (n={int(row['count'])})")
        
        # Save the richness stats
        richness_path = os.path.join(self.output_dir, 'bath_richness_stats.csv')
        richness_stats.to_csv(richness_path)
        print(f"Bath richness statistics saved to '{richness_path}'")
        
        # Create visualization
        try:
            # Define the order of categories for visualization
            order = ['LowBath', 'EqualBath', 'ModerateExtraBath', 'HighExtraBath', 'VeryHighExtraBath']
            
            # Only use categories with sufficient data
            valid_categories = [cat for cat in order if cat in richness_stats.index and richness_stats.loc[cat, 'count'] > 5]
            
            if valid_categories:
                plt.figure(figsize=(10, 6))
                
                # Filter dataframe to only those categories and add count to label
                plot_df = analysis_df[analysis_df['Bath_Richness'].isin(valid_categories)].copy()
                
                # Create the plot
                sns.barplot(x='Bath_Richness', y=self.target, data=plot_df, order=valid_categories)
                plt.title('Price Impact of Bathroom Richness')
                plt.xlabel('Bathroom Richness Category')
                plt.ylabel(f'{self.target}')
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                viz_path = os.path.join(self.output_dir, 'bath_richness_impact.png')
                plt.savefig(viz_path)
                plt.close()
                print(f"Bath richness impact chart saved to '{viz_path}'")
        except Exception as e:
            print(f"Error creating visualization: {e}")
    
    def run_all_analyses(self):
        """Run all bed-bath analyses in sequence"""
        if self.df is None:
            print("No data loaded. Run load_data() first.")
            return
            
        print("\n===== RUNNING ALL BED-BATH ANALYSES =====")
        
        # Run premium analysis
        premium_df = self.compute_bed_bath_premium()
        
        # Create visualization if we have results
        if premium_df is not None and len(premium_df) > 0:
            self.plot_bathroom_premium(premium_df)
        
        # Run bath richness analysis
        self.analyze_bath_richness()
        
        print("\n===== BED-BATH ANALYSES COMPLETE =====")