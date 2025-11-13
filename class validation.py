import pandas as pd
import numpy as np
import os

def validate_dataframe(df):
    """
    Validate the dataframe before modeling - FIXED VERSION
    """
    errors = []
    
    # Check required columns
    required_columns = ['Time', 'Class'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
    
    # Check data types
    if 'Class' in df.columns and df['Class'].dtype not in [np.int64, np.int32, np.float64]:
        errors.append("Class column should be numeric type")
    
    # Check for NaN values - MORE DETAILED
    nan_count = df.isnull().sum().sum()
    if nan_count > 0:
        nan_details = df.isnull().sum()
        nan_cols = nan_details[nan_details > 0]
        errors.append(f"DataFrame contains {nan_count} NaN values in columns: {dict(nan_cols)}")
    
    # Check class distribution
    if 'Class' in df.columns:
        class_counts = df['Class'].value_counts()
        if len(class_counts) != 2:
            errors.append("Class column should have exactly 2 unique values (0 and 1)")
        # Check if there are any fraud cases
        if 1 not in class_counts.index:
            errors.append("No fraud cases (Class=1) found in the dataset")
    
    if errors:
        print("Data validation issues found:")
        for error in errors:
            print(f"  - {error}")
        return False, errors
    else:
        print("✓ Data validation passed!")
        return True, []

def enhanced_data_cleaning(df):
    """
    Enhanced cleaning that properly handles NaN values
    """
    print("Starting enhanced data cleaning...")
    df_clean = df.copy()
    
    # 1. Remove duplicates
    initial_size = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    removed_duplicates = initial_size - len(df_clean)
    print(f"✓ Removed {removed_duplicates} duplicate rows")
    
    # 2. Handle NaN values more robustly
    nan_before = df_clean.isnull().sum().sum()
    if nan_before > 0:
        print(f"Found {nan_before} NaN values, handling them...")
        
        # For numeric columns, fill with median
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
                print(f"  - Filled NaN in {col} with median: {median_val}")
        
        # For non-numeric columns, fill with mode or 'unknown'
        non_numeric_cols = df_clean.select_dtypes(include=['object']).columns
        for col in non_numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                mode_val = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'unknown'
                df_clean[col].fillna(mode_val, inplace=True)
                print(f"  - Filled NaN in {col} with mode: {mode_val}")
    
    # 3. Validate Class column
    if 'Class' in df_clean.columns:
        valid_classes = df_clean[df_clean['Class'].isin([0, 1])]
        if len(valid_classes) != len(df_clean):
            invalid_count = len(df_clean) - len(valid_classes)
            print(f"✓ Removed {invalid_count} rows with invalid Class values")
            df_clean = valid_classes.copy()
    
    # 4. Final NaN check
    nan_after = df_clean.isnull().sum().sum()
    if nan_after == 0:
        print("✓ All NaN values handled successfully")
    else:
        print(f"⚠ Warning: {nan_after} NaN values remain after cleaning")
    
    print(f"✓ Final dataset shape: {df_clean.shape}")
    return df_clean

def safe_feature_engineering(df):
    """
    Safe feature engineering that handles potential issues
    """
    df_eng = df.copy()
    
    try:
        # 1. Create time-based features
        df_eng['Hour'] = (df_eng['Time'] // 3600) % 24
        
        # 2. Create amount categories safely
        amount_bins = [0, 10, 50, 100, 500, 1000, float('inf')]
        amount_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High', 'Extreme']
        
        df_eng['Amount_Category'] = pd.cut(
            df_eng['Amount'], 
            bins=amount_bins, 
            labels=amount_labels,
            include_lowest=True
        )
        
        # 3. Safe interaction features
        if 'V1' in df_eng.columns and 'V2' in df_eng.columns:
            df_eng['V1_V2_Interaction'] = df_eng['V1'] * df_eng['V2']
        
        if 'V3' in df_eng.columns and 'V4' in df_eng.columns:
            df_eng['V3_V4_Interaction'] = df_eng['V3'] * df_eng['V4']
        
        print("✓ Feature engineering completed successfully")
        
    except Exception as e:
        print(f"⚠ Feature engineering warning: {e}")
        print("  Continuing with basic features...")
    
    return df_eng

def complete_data_preparation(csv_path, output_path=None):
    """
    Complete pipeline from raw data to modeling-ready data - FIXED VERSION
    """
    print("=== COMPLETE DATA PREPARATION PIPELINE ===\n")
    
    try:
        # 1. Load data
        print("Step 1: Loading data...")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        print(f"✓ Data loaded successfully! Shape: {df.shape}")
        
        # 2. Enhanced cleaning
        print("\nStep 2: Enhanced data cleaning...")
        df_clean = enhanced_data_cleaning(df)
        
        # 3. Feature engineering
        print("\nStep 3: Safe feature engineering...")
        df_engineered = safe_feature_engineering(df_clean)
        
        # 4. Validate data
        print("\nStep 4: Validating data...")
        is_valid, errors = validate_dataframe(df_engineered)
        
        if not is_valid:
            print("Trying to fix validation issues...")
            # Apply additional cleaning if validation fails
            df_engineered = enhanced_data_cleaning(df_engineered)
            is_valid, errors = validate_dataframe(df_engineered)
            
            if not is_valid:
                print("Critical validation errors remain. Cannot proceed.")
                return None, None, None, None, None, None
        
        # 5. Save cleaned data
        if output_path:
            df_engineered.to_csv(output_path, index=False)
            print(f"✓ Cleaned data saved to: {output_path}")
        
        # 6. Prepare for modeling
        print("\nStep 5: Preparing for modeling...")
        from sklearn.model_selection import train_test_split
        
        # Separate features and target
        X = df_engineered.drop('Class', axis=1)
        y = df_engineered['Class']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Calculate class weights for handling imbalance
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(y_train), 
            y=y_train
        )
        class_weights = dict(enumerate(class_weights))
        
        print("✓ Data preparation completed successfully!")
        print(f"  Training set shape: {X_train.shape}")
        print(f"  Test set shape: {X_test.shape}")
        print(f"  Class distribution: {y_train.value_counts().to_dict()}")
        print(f"  Class weights: {class_weights}")
        
        return X_train, X_test, y_train, y_test, class_weights, df_engineered
        
    except Exception as e:
        print(f"❌ Error in data preparation: {e}")
        return None, None, None, None, None, None

# Simple alternative if the above still fails
def simple_data_preparation(csv_path):
    """
    Ultra-simple data preparation as fallback
    """
    print("Using simple data preparation...")
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Basic cleaning
    df_clean = df.drop_duplicates().dropna()
    
    # Remove any infinite values
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Ensure Class column is valid
    df_clean = df_clean[df_clean['Class'].isin([0, 1])]
    
    # Simple preprocessing - scale only Time and Amount
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df_clean[['Time', 'Amount']] = scaler.fit_transform(df_clean[['Time', 'Amount']])
    
    # Split data
    from sklearn.model_selection import train_test_split
    X = df_clean.drop('Class', axis=1)
    y = df_clean['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print("✓ Simple preparation completed!")
    print(f"  Final shape: {X_train.shape}, {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, None, df_clean

# Main execution with error handling
if __name__ == "__main__":
    csv_file = "creditcard.csv"
    
    if os.path.exists(csv_file):
        print("File found! Starting data preparation...\n")
        
        # Try the complete pipeline first
        result = complete_data_preparation(csv_file, 'creditcard_cleaned.csv')
        
        # If complete pipeline fails, use simple version
        if result[0] is None:
            print("\nFalling back to simple data preparation...")
            X_train, X_test, y_train, y_test, class_weights, final_df = simple_data_preparation(csv_file)
        else:
            X_train, X_test, y_train, y_test, class_weights, final_df = result
        
        if X_train is not None:
            print("\n=== FINAL RESULT ===")
            print(f"✓ Data is ready for modeling!")
            print(f"  Training samples: {len(X_train)}")
            print(f"  Test samples: {len(X_test)}")
            print(f"  Features: {X_train.shape[1]}")
            print(f"  Fraud cases in training: {y_train.sum()} ({y_train.mean()*100:.4f}%)")
            
            # Save prepared data
            final_df.to_csv("creditcard_final_cleaned.csv", index=False)
            print(f"✓ Final dataset saved to: creditcard_final_cleaned.csv")
            
        else:
            print("❌ Data preparation failed completely.")
            
    else:
        print(f"Error: File '{csv_file}' not found!")
        print("Please make sure the CSV file is in the same directory.")