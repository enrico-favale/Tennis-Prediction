import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

def __drop_unusable_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "Score" in df.columns:
        df = df.drop("Score", axis=1)
        
    return df

def __encode_categorical(df: pd.DataFrame):
    categorical_columns = [
        "Tournament", "Series", "Court", "Surface", "Round", "Season"
    ]
    
    label_encoders = {}
    
    for col in categorical_columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    return df, label_encoders

def __define_target_features(df: pd.DataFrame):
    if "Winner" not in df.columns:
        raise ValueError("Dataset must contain 'Winner' column for training.")
    
    y = (df["Winner"] == df["Player_1"]).astype(int)
    X = df.drop(["Winner"], axis=1)
    
    return X, y

def __split_by_date(df: pd.DataFrame, cutoff_date: str = '2024-01-01') -> tuple[pd.DataFrame, pd.DataFrame]:
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    train_df = df[df['Date'] < cutoff_date].copy()
    test_df = df[df['Date'] >= cutoff_date].copy()
    
    # drop Date
    train_df = train_df.drop("Date", axis=1)
    test_df = test_df.drop("Date", axis=1)
    
    return train_df, test_df

def __save_label(label_encoders: dict, filename: str = "../data/labels/label_encoders.pkl") -> None:
    joblib.dump(label_encoders, filename)

def preprocess_data(df_processed: pd.DataFrame, cutoff_date: str = '2024-01-01'):
    df_processed = __drop_unusable_columns(df_processed)
    df_processed, label_encoders = __encode_categorical(df_processed)
    train_df, test_df = __split_by_date(df_processed, cutoff_date)
    
    __save_label(label_encoders, "../data/labels/label_encoders.pkl")
    
    X_train, y_train = __define_target_features(train_df)
    X_test, y_test = __define_target_features(test_df)
    
    return df_processed, X_train, y_train, X_test, y_test, label_encoders
