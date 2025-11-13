import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('creditcard.csv')

# Basic information
print("Dataset Shape:", df.shape)
print("\nColumn Names:")
print(df.columns.tolist())
print("\nData Types:")
print(df.dtypes)
print("\nFirst 5 rows:")
print(df.head())

# Check for missing values
print("Missing Values:")
print(df.isnull().sum())

# Check for duplicates
print(f"\nDuplicate rows: {df.duplicated().sum()}")

# Check class distribution
print("\nClass Distribution:")
print(df['Class'].value_counts())
print("\nClass Percentage:")
print(df['Class'].value_counts(normalize=True) * 100)

#data cleaning

def clean_creditcard_data(df):
    """
    Comprehensive cleaning function for credit card fraud dataset
    """
    # Create a copy to avoid modifying original data
    df_clean = df.copy()
    
    # 1. Handle duplicates
    initial_size = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    print(f"Removed {initial_size - len(df_clean)} duplicate rows")
    
    # 2. Check for infinite values in V1-V28 columns
    v_columns = [f'V{i}' for i in range(1, 29)]
    for col in v_columns:
        if np.any(np.isinf(df_clean[col])):
            print(f"Found infinite values in {col}, replacing with NaN")
            df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
    
    # 3. Handle missing values (though this dataset typically has none)
    missing_before = df_clean.isnull().sum().sum()
    if missing_before > 0:
        print(f"Missing values found: {missing_before}")
        # For PCA components (V1-V28), we can use median imputation
        for col in v_columns + ['Amount']:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    # 4. Validate Class column
    invalid_classes = df_clean[~df_clean['Class'].isin([0, 1])]
    if len(invalid_classes) > 0:
        print(f"Found {len(invalid_classes)} invalid class values, removing them")
        df_clean = df_clean[df_clean['Class'].isin([0, 1])]
    
    # 5. Check for outliers in Amount (though we'll handle this separately)
    amount_stats = df_clean['Amount'].describe()
    print(f"\nAmount statistics:\n{amount_stats}")
    
    # 6. Validate Time column (should be sequential)
    time_diff = df_clean['Time'].diff().dropna()
    negative_time_jumps = (time_diff < 0).sum()
    print(f"Negative time jumps: {negative_time_jumps}")
    
    return df_clean

# Apply cleaning
df_clean = clean_creditcard_data(df)

# feature Engineering 4

def engineer_features(df):
    """
    Create new features that might help with fraud detection
    """
    df_eng = df.copy()
    
    # 1. Create time-based features
    # Convert seconds to hours of day (assuming dataset starts at midnight)
    df_eng['Hour'] = (df_eng['Time'] // 3600) % 24
    df_eng['Time_of_Day'] = pd.cut(df_eng['Hour'], 
                                  bins=[0, 6, 12, 18, 24], 
                                  labels=['Night', 'Morning', 'Afternoon', 'Evening'])
    
    # 2. Create amount categories
    df_eng['Amount_Category'] = pd.cut(df_eng['Amount'],
                                      bins=[0, 10, 50, 100, 500, 1000, float('inf')],
                                      labels=['Very Low', 'Low', 'Medium', 'High', 'Very High', 'Extreme'])
    
    # 3. Create interaction features (combinations of V features that might be meaningful)
    df_eng['V1_V2_Interaction'] = df_eng['V1'] * df_eng['V2']
    df_eng['V3_V4_Interaction'] = df_eng['V3'] * df_eng['V4']
    
    # 4. Create statistical features
    v_columns = [f'V{i}' for i in range(1, 29)]
    df_eng['V_Mean'] = df_eng[v_columns].mean(axis=1)
    df_eng['V_Std'] = df_eng[v_columns].std(axis=1)
    df_eng['V_Sum_Abs'] = df_eng[v_columns].abs().sum(axis=1)
    
    return df_eng

df_engineered = engineer_features(df_clean)

# handling class imblance 5 

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import class_weight

def handle_imbalance(X, y, method='smote', random_state=42):
    """
    Handle class imbalance using various techniques
    """
    print(f"Before balancing - Class distribution: {np.bincount(y)}")
    
    if method == 'smote':
        # SMOTE: Synthetic Minority Over-sampling Technique
        smote = SMOTE(random_state=random_state)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
    elif method == 'undersample':
        # Random undersampling of majority class
        undersampler = RandomUnderSampler(random_state=random_state)
        X_balanced, y_balanced = undersampler.fit_resample(X, y)
        
    elif method == 'class_weight':
        # Use class weights in the model instead of resampling
        class_weights = class_weight.compute_class_weight(
            'balanced', classes=np.unique(y), y=y)
        return X, y, class_weights
    
    else:
        # No balancing
        return X, y, None
    
    print(f"After balancing - Class distribution: {np.bincount(y_balanced)}")
    return X_balanced, y_balanced, None

# data processing pipline 6
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def create_preprocessing_pipeline():
    """
    Create a preprocessing pipeline for the credit card data
    """
    # Features to scale
    numeric_features = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Hour']
    
    # Features to encode (categorical)
    categorical_features = ['Time_of_Day', 'Amount_Category']
    
    # Create preprocessing steps
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def prepare_data_for_modeling(df, test_size=0.2, balance_method='class_weight'):
    """
    Complete data preparation function
    """
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    
    # Handle class imbalance
    X_train_balanced, y_train_balanced, class_weights = handle_imbalance(
        X_train, y_train, method=balance_method
    )
    
    return X_train_balanced, X_test, y_train_balanced, y_test, class_weights

# error handling 7

def validate_dataframe(df):
    """
    Validate the dataframe before modeling
    """
    errors = []
    
    # Check required columns
    required_columns = ['Time', 'Class'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
    
    # Check data types
    if df['Class'].dtype not in [np.int64, np.int32]:
        errors.append("Class column should be integer type")
    
    # Check for NaN values
    if df.isnull().sum().sum() > 0:
        errors.append("DataFrame contains NaN values")
    
    # Check class distribution
    class_counts = df['Class'].value_counts()
    if len(class_counts) != 2:
        errors.append("Class column should have exactly 2 unique values (0 and 1)")
    
    if errors:
        raise ValueError(f"Data validation errors:\n" + "\n".join(errors))
    else:
        print("Data validation passed!")

# Usage
try:
    validate_dataframe(df_engineered)
except ValueError as e:
    print(e)
    # Handle the error appropriately

#complete data processing script 8
def complete_data_preparation(csv_path, output_path=None):
    """
    Complete pipeline from raw data to modeling-ready data
    """
    # 1. Load data
    print("Step 1: Loading data...")
    df = pd.read_csv(csv_path)
    
    # 2. Clean data
    print("Step 2: Cleaning data...")
    df_clean = clean_creditcard_data(df)
    
    # 3. Engineer features
    print("Step 3: Engineering features...")
    df_engineered = engineer_features(df_clean)
    
    # 4. Validate data
    print("Step 4: Validating data...")
    validate_dataframe(df_engineered)
    
    # 5. Save cleaned data
    if output_path:
        df_engineered.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
    
    # 6. Prepare for modeling
    print("Step 5: Preparing for modeling...")
    X_train, X_test, y_train, y_test, class_weights = prepare_data_for_modeling(df_engineered)
    
    print("Data preparation completed successfully!")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Class distribution in training: {np.bincount(y_train)}")
    
    return X_train, X_test, y_train, y_test, class_weights, df_engineered

# Run the complete pipeline
X_train, X_test, y_train, y_test, class_weights, final_df = complete_data_preparation(
    'creditcard.csv', 
    'creditcard_cleaned.csv'
)