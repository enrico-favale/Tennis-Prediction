import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder


def __drop_unusable_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that are not useful for model training.
    """
    if "Score" in df.columns:
        df = df.drop("Score", axis=1)
        
    return df


def __encode_categorical(df: pd.DataFrame):
    """
    Encode categorical variables with consistent encoding for player names.
    Player_1, Player_2, and Winner share the same encoder to ensure consistency.
    """
    # Separate player columns from other categorical columns
    player_columns = ["Player_1", "Player_2", "Winner"]
    other_categorical_columns = ["Tournament", "Series", "Court", "Surface", "Round"]
    
    label_encoders = {}
    
    # Create a unified encoder for all player names
    if all(col in df.columns for col in player_columns):
        print("Creating unified player encoder...")
        
        # Collect all unique player names from all three columns
        all_players = pd.concat([
            df["Player_1"],
            df["Player_2"],
            df["Winner"]
        ]).unique()
        
        # Create single encoder for all players
        player_encoder = LabelEncoder()
        player_encoder.fit(all_players.astype(str))
        
        # Apply the same encoder to all player columns
        for col in player_columns:
            df[col] = player_encoder.transform(df[col].astype(str))
        
        # Store the same encoder for all player columns
        label_encoders["Player_1"] = player_encoder
        label_encoders["Player_2"] = player_encoder
        label_encoders["Winner"] = player_encoder
        
        print(f"  Total unique players: {len(all_players)}")
        print(f"  Encoded range: 0 to {len(all_players) - 1}")
    
    # Encode other categorical columns separately
    for col in other_categorical_columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            
            unique_count = len(le.classes_)
            print(f"  Encoded '{col}': {unique_count} unique values")
    
    return df, label_encoders


def __define_target_features(df: pd.DataFrame):
    """
    Define features (X) and target (y) from the dataframe.
    Target is binary: 1 if Player_1 wins, 0 if Player_2 wins.
    """
    if "Winner" not in df.columns:
        raise ValueError("Dataset must contain 'Winner' column for training.")
    
    # Binary target: 1 if Player_1 wins, 0 otherwise
    y = (df["Winner"] == df["Player_1"]).astype(int)
    
    # Features: everything except Winner
    X = df.drop(["Winner"], axis=1)
    
    return X, y


def __split_by_date(df: pd.DataFrame, cutoff_date: str = '2024-01-01') -> tuple:
    """
    Split dataframe into train and test sets based on date.
    Maintains temporal order for time-series data.
    """
    print(f"\nSplitting data by date: {cutoff_date}")
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    train_df = df[df['Date'] < cutoff_date].copy()
    test_df = df[df['Date'] >= cutoff_date].copy()
    
    print(f"  Training set: {len(train_df)} matches ({df['Date'].min()} to {train_df['Date'].max()})")
    print(f"  Test set: {len(test_df)} matches ({test_df['Date'].min()} to {df['Date'].max()})")
    
    # Drop Date column after split
    train_df = train_df.drop("Date", axis=1)
    test_df = test_df.drop("Date", axis=1)
    
    return train_df, test_df


def __save_label_encoders(label_encoders: dict, filename: str = "../data/labels/label_encoders.pkl") -> None:
    """
    Save label encoders to disk for future use (e.g., prediction on new data).
    """
    import os
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    joblib.dump(label_encoders, filename)
    print(f"\nLabel encoders saved to: {filename}")


def __validate_encoding_consistency(df: pd.DataFrame) -> None:
    """
    Verify that player encoding is consistent across Player_1, Player_2, and Winner columns.
    """
    print("\nValidating encoding consistency...")
    
    # Check if Winner values are always from Player_1 or Player_2
    valid_winners = (df["Winner"] == df["Player_1"]) | (df["Winner"] == df["Player_2"])
    
    if not valid_winners.all():
        invalid_count = (~valid_winners).sum()
        print(f"  WARNING: {invalid_count} rows have Winner that doesn't match Player_1 or Player_2!")
    else:
        print(f"  ✓ All {len(df)} rows have consistent winner encoding")
    
    # Check encoding ranges
    p1_range = (df["Player_1"].min(), df["Player_1"].max())
    p2_range = (df["Player_2"].min(), df["Player_2"].max())
    w_range = (df["Winner"].min(), df["Winner"].max())
    
    print(f"  Player_1 encoding range: {p1_range}")
    print(f"  Player_2 encoding range: {p2_range}")
    print(f"  Winner encoding range: {w_range}")


def preprocess_data(
    df_processed: pd.DataFrame, 
    cutoff_date: str = '2024-01-01',
    validate: bool = True
) -> tuple:
    """
    Complete preprocessing pipeline for tennis match data.
    
    Parameters:
    -----------
    df_processed : pd.DataFrame
        Processed dataframe with features
    cutoff_date : str, default='2024-01-01'
        Date to split train/test sets
    validate : bool, default=True
        Whether to validate encoding consistency
        
    Returns:
    --------
    tuple: (df_processed, X_train, y_train, X_test, y_test, label_encoders)
    """
    print("=" * 70)
    print("PREPROCESSING DATA")
    print("=" * 70)
    
    # Step 1: Drop unusable columns
    print("\n1. Dropping unusable columns...")
    df_processed = __drop_unusable_columns(df_processed)
    print(f"   Remaining columns: {len(df_processed.columns)}")
    
    # Step 2: Encode categorical variables (with unified player encoding)
    print("\n2. Encoding categorical variables...")
    df_processed, label_encoders = __encode_categorical(df_processed)
    
    # Step 3: Validate encoding consistency
    if validate:
        __validate_encoding_consistency(df_processed)
    
    # Step 4: Split by date
    train_df, test_df = __split_by_date(df_processed, cutoff_date)
    
    # Step 5: Save label encoders
    __save_label_encoders(label_encoders, "../data/labels/label_encoders.pkl")
    
    # Step 6: Define features and target
    print("\n3. Defining features and target...")
    X_train, y_train = __define_target_features(train_df)
    X_test, y_test = __define_target_features(test_df)
    
    print(f"   X_train shape: {X_train.shape}")
    print(f"   X_test shape: {X_test.shape}")
    print(f"   y_train distribution: 0={np.sum(y_train==0)}, 1={np.sum(y_train==1)}")
    print(f"   y_test distribution: 0={np.sum(y_test==0)}, 1={np.sum(y_test==1)}")
    
    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETED")
    print("=" * 70)
    
    return df_processed, X_train, y_train, X_test, y_test, label_encoders


def load_label_encoders(filename: str = "../data/labels/label_encoders.pkl") -> dict:
    """
    Load saved label encoders from disk.
    
    Parameters:
    -----------
    filename : str
        Path to saved label encoders
        
    Returns:
    --------
    dict: Dictionary of label encoders
    """
    label_encoders = joblib.load(filename)
    print(f"Label encoders loaded from: {filename}")
    return label_encoders


def decode_predictions(encoded_values, encoder_name: str, label_encoders: dict):
    """
    Decode encoded values back to original labels.
    
    Parameters:
    -----------
    encoded_values : array-like
        Encoded values to decode
    encoder_name : str
        Name of the encoder to use (e.g., 'Player_1', 'Tournament')
    label_encoders : dict
        Dictionary of label encoders
        
    Returns:
    --------
    array: Decoded values
    """
    if encoder_name not in label_encoders:
        raise ValueError(f"Encoder '{encoder_name}' not found in label_encoders")
    
    encoder = label_encoders[encoder_name]
    return encoder.inverse_transform(encoded_values)


# ============================================
# Usage Example
# ============================================

if __name__ == "__main__":
    # Load processed data
    df = pd.read_csv("../data/processed/atp_tennis_processed.csv")
    
    # Preprocess with validation
    df_encoded, X_train, y_train, X_test, y_test, encoders = preprocess_data(
        df, 
        cutoff_date='2024-01-01',
        validate=True
    )
    
    # Example: Decode some player IDs back to names
    sample_player_ids = X_train["Player_1"].head(5).values
    player_names = decode_predictions(sample_player_ids, "Player_1", encoders)
    print("\nExample decoded player names:")
    for pid, name in zip(sample_player_ids, player_names):
        print(f"  ID {pid} -> {name}")
    
    # Load encoders later if needed
    # loaded_encoders = load_label_encoders("../data/labels/label_encoders.pkl")
