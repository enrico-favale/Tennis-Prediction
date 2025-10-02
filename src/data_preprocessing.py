import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, Dict

class TennisPreprocessor:
    def __init__(self, cutoff_date: str = "2024-01-01", labels_path: str = "../data/labels/preprocessor.pkl"):
        self.cutoff_date = cutoff_date
        self.labels_path = labels_path
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.fitted = False

    def __drop_unusable_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Score" in df.columns:
            df = df.drop("Score", axis=1)
        return df

    def __encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features (players share the same encoder).
        """
        player_columns = ["Player_1", "Player_2", "Winner"]
        other_categorical_columns = ["Tournament", "Series", "Court", "Surface", "Round", "Season"]

        # Player encoder (unique across all players)
        if all(col in df.columns for col in player_columns):
            all_players = pd.concat([df["Player_1"], df["Player_2"], df["Winner"]]).unique()
            player_encoder = LabelEncoder()
            player_encoder.fit(all_players.astype(str))

            for col in player_columns:
                df[col] = player_encoder.transform(df[col].astype(str))

            # Save same encoder for all player columns
            for col in player_columns:
                self.label_encoders[col] = player_encoder

        # Other categoricals
        for col in other_categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le

        return df

    def __define_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        if "Winner" not in df.columns:
            raise ValueError("Dataset must contain 'Winner' column for training.")
        y = (df["Winner"] == df["Player_1"]).astype(int)
        X = df.drop(["Winner"], axis=1)
        return X, y

    def __split_by_date(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        train_df = df[df["Date"] < self.cutoff_date].copy()
        test_df = df[df["Date"] >= self.cutoff_date].copy()

        train_df = train_df.drop("Date", axis=1)
        test_df = test_df.drop("Date", axis=1)

        return train_df, test_df

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Fit encoders & scaler on training set and transform the dataset.
        """
        print("\n=== FIT & TRANSFORM PREPROCESSOR ===")
        df = self.__drop_unusable_columns(df)
        df = self.__encode_categorical(df)

        train_df, test_df = self.__split_by_date(df)

        X_train, y_train = self.__define_target(train_df)
        X_test, y_test = self.__define_target(test_df)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Save encoders + scaler
        self.save()

        self.fitted = True
        return X_train_scaled, y_train, X_test_scaled, y_test

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Transform new data using fitted encoders & scaler (for prediction).
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit_transform first or load().")

        df = self.__drop_unusable_columns(df)

        # Encode with existing encoders
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[col].astype(str))

        X, y = self.__define_target(df)
        X_scaled = self.scaler.transform(X)
        return X_scaled, y

    def save(self):
        """
        Save encoders and scaler.
        """
        to_save = {"encoders": self.label_encoders, "scaler": self.scaler}
        joblib.dump(to_save, self.labels_path)
        print(f"✅ Preprocessor saved to {self.labels_path}")

    def load(self):
        """
        Load encoders and scaler.
        """
        loaded = joblib.load(self.labels_path)
        self.label_encoders = loaded["encoders"]
        self.scaler = loaded["scaler"]
        self.fitted = True
        print(f"✅ Preprocessor loaded from {self.labels_path}")

    def decode(self, encoded_values, encoder_name: str):
        """
        Decode encoded values back to original labels.
        """
        if encoder_name not in self.label_encoders:
            raise ValueError(f"Encoder '{encoder_name}' not found.")
        return self.label_encoders[encoder_name].inverse_transform(encoded_values)


# =============================
# Example usage
# =============================
if __name__ == "__main__":
    df = pd.read_csv("../data/processed/atp_tennis_processed.csv")

    preprocessor = TennisPreprocessor(cutoff_date="2024-01-01")

    # Fit & transform
    X_train, y_train, X_test, y_test = preprocessor.fit_transform(df)

    # Later for prediction
    preprocessor.load()
    sample_encoded = [0, 1, 2]
    print("Decoded:", preprocessor.decode(sample_encoded, "Tournament"))
