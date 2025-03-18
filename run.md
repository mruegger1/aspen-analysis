# Running the Analysis Pipeline

Execute the scripts in the following order:

## 1. Data Merging
```bash
python3 updated-merge-script.py
```

## 2. Data Cleaning
```bash
python3 updated-cleaning-script-2.py
python3 standalone-condition-script.py
```

## 3. Feature Engineering
```bash
python3 feature_engineering.py
```

## 4. Analysis Modules
```bash
# Premium location analysis
python3 premium-locations/premium_address_analysis.py

# Time adjustment analysis
python3 time-adjustments/time_adjusted_analysis.py

# STR impact analysis
python3 str-analysis/str_gondola_analysis.py
```

## Output Files
Each script generates CSV outputs that serve as inputs for subsequent steps:
- `merged_data.csv` → Initial merged dataset
- `cleaned_real_estate_dataset.csv` → Cleaned dataset
- `real_estate_with_condition.csv` → Dataset with property condition
- `featured_real_estate_data.csv` → Dataset with all engineered features
