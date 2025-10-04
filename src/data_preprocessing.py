import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, Dict

class TennisPreprocessor:
    def __init__(self, cutoff_date: str = "2024-01-01", labels_path: str = "../data/labels/label_encoders.pkl"):
        """
        Initialize the tennis data preprocessor.

        Parameters
        ----------
        cutoff_date : str
            Date to split training and test sets (format YYYY-MM-DD)
        labels_path : str
            Path to save/load label encoders and scaler
        """
        self.cutoff_date = cutoff_date
        self.labels_path = labels_path
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.fitted = False

    def __drop_unusable_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop columns that are not useful for model training.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe

        Returns
        -------
        pd.DataFrame
            Dataframe with unused columns removed
        """
        if "Score" in df.columns:
            df = df.drop("Score", axis=1)
        return df

    def __encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features using LabelEncoder.
        Player columns (Player_1, Player_2, Winner) share the same encoder
        to maintain consistency across all player-related features.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with categorical columns

        Returns
        -------
        pd.DataFrame
            Dataframe with categorical features encoded as integers
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

        # Encode other categorical features separately
        for col in other_categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le

        return df

    def __define_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Define the target variable and separate features.
        Target is binary: 1 if Player_1 wins, 0 if Player_2 wins.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with 'Winner' column

        Returns
        -------
        X : pd.DataFrame
            Feature dataframe without 'Winner'
        y : pd.Series
            Binary target series indicating if Player_1 won
        """
        if "Winner" not in df.columns:
            raise ValueError("Dataset must contain 'Winner' column for training.")
        y = (df["Winner"] == df["Player_1"]).astype(int)
        X = df.drop(["Winner"], axis=1)
        return X, y

    def __split_by_date(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split dataframe into training and test sets based on the cutoff date.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with 'Date' column

        Returns
        -------
        train_df : pd.DataFrame
            Training set with Date < cutoff_date
        test_df : pd.DataFrame
            Test set with Date >= cutoff_date
        """
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        train_df = df[df["Date"] < self.cutoff_date].copy()
        test_df = df[df["Date"] >= self.cutoff_date].copy()

        train_df = train_df.drop("Date", axis=1)
        test_df = test_df.drop("Date", axis=1)

        return train_df, test_df

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, pd.Series, np.ndarray, pd.Series]:
        """
        Fit label encoders and scaler on the training set and transform the dataset.
        Returns the processed dataframe and scaled train/test features and targets.

        Parameters
        ----------
        df : pd.DataFrame
            Raw dataframe to preprocess

        Returns
        -------
        df_processed : pd.DataFrame
            Full dataframe after encoding categorical features
        X_train_scaled : np.ndarray
            Scaled training features
        y_train : pd.Series
            Training target
        X_test_scaled : np.ndarray
            Scaled test features
        y_test : pd.Series
            Test target
        """
        print("\n=== FIT & TRANSFORM PREPROCESSOR ===")
        df_processed = self.__drop_unusable_columns(df)
        df_processed = self.__encode_categorical(df_processed)

        train_df, test_df = self.__split_by_date(df_processed)

        X_train, y_train = self.__define_target(train_df)
        X_test, y_test = self.__define_target(test_df)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Save encoders + scaler
        self.save()

        self.fitted = True
    
        return df_processed, X_train_scaled, y_train, X_test_scaled, y_test

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Transform new/unseen data using fitted encoders and scaler.

        Parameters
        ----------
        df : pd.DataFrame
            New dataframe to transform

        Returns
        -------
        X_scaled : np.ndarray
            Scaled feature matrix
        y : pd.Series
            Target variable
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
        Save the fitted label encoders and scaler to disk.
        """
        to_save = {"encoders": self.label_encoders, "scaler": self.scaler}
        joblib.dump(to_save, self.labels_path)
        print(f"✅ Preprocessor saved to {self.labels_path}")

    def load(self):
        """
        Load previously saved label encoders and scaler from disk.
        """
        loaded = joblib.load(self.labels_path)
        self.label_encoders = loaded["encoders"]
        self.scaler = loaded["scaler"]
        self.fitted = True
        print(f"✅ Preprocessor loaded from {self.labels_path}")

    def decode(self, encoded_values, encoder_name: str):
        """
        Decode encoded integer values back to original categorical labels.

        Parameters
        ----------
        encoded_values : array-like
            Encoded integer values
        encoder_name : str
            Name of the encoder to use

        Returns
        -------
        array
            Original categorical labels
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
