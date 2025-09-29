import pandas as pd
import numpy as np

def __straight_sets_victory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate whether a match was won in straight sets (no set lost).
    Adds a new column 'Straight_sets_victory' with 1 for straight-sets wins and 0 otherwise.
    Input: DataFrame with columns 'Score' and 'Best of'
    Output: DataFrame with new column 'Straight_sets_victory'
    """
    
    n_sets = df["Score"].str.split().apply(len)
    
    df["Straight_sets_victory"] = np.where(
        ((df["Best of"] == 3) & (n_sets <= 2)) |
        ((df["Best of"] == 5) & (n_sets <= 3)), 1, 0
    )
    
    return df

def __season(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the season from the 'Date' column.
    Encode seasons as numerical IDs: Spring=0, Summer=1, Autumn=2, Winter=3.
    Input: DataFrame with 'Date' column in format YYYY-MM-DD
    Output: DataFrame with new column 'Season_id'
    """
    
    months = df["Date"].str.split("-").str[1].astype(int)

    conditions = [
        months.isin([3, 4, 5]),       # Spring
        months.isin([6, 7, 8]),       # Summer
        months.isin([9, 10, 11]),     # Autumn
        months.isin([12, 1, 2])       # Winter
    ]
    seasons = ["Spring", "Summer", "Autumn", "Winter"]

    df["Season"] = np.select(conditions, seasons, default="Unknown")
    df["Season_id"] = pd.Categorical(df["Season"], categories=seasons).codes
    
    return df

def __rank_difference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the difference in ranking and points between the two players.
    Adds 'Rank_diff' = Rank_1 - Rank_2 and 'Pts_diff' = Pts_1 - Pts_2.
    Input: DataFrame with 'Rank_1', 'Rank_2', 'Pts_1', 'Pts_2'
    Output: DataFrame with new columns 'Rank_diff', 'Pts_diff'
    """

def __odds_difference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the difference in betting odds between the two players.
    Adds 'Odds_diff' = Odd_1 - Odd_2
    Input: DataFrame with 'Odd_1' and 'Odd_2'
    Output: DataFrame with new column 'Odds_diff'
    """

def __h2h_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute head-to-head statistics between Player_1 and Player_2.
    Adds columns for previous wins and differences: 'H2H_wins_1', 'H2H_wins_2', 'H2H_diff'
    Input: DataFrame with historical match results
    Output: DataFrame with new H2H feature columns
    """

def __recent_form(df: pd.DataFrame, n_matches: int = 5) -> pd.DataFrame:
    """
    Compute recent form of each player based on last n_matches.
    Adds features like 'Win_pct_1_lastN', 'Win_pct_2_lastN', 'Win_pct_diff_lastN',
    'Sets_won_ratio', 'Games_won_ratio', etc.
    Input: DataFrame with historical match results
    Output: DataFrame with new recent form features
    """

def __surface_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute performance of each player on the match surface.
    Adds features like 'Surface_win_pct_1', 'Surface_win_pct_2', 'Surface_win_diff'
    Input: DataFrame with 'Surface' column and historical match results
    Output: DataFrame with new surface performance features
    """

def __tournament_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute performance of each player in this tournament or similar tournaments.
    Adds features like 'Tournament_win_pct_1', 'Tournament_win_pct_2'
    Input: DataFrame with 'Tournament' column and historical match results
    Output: DataFrame with new tournament performance features
    """

def __rank_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ranking or points change over recent period.
    Adds features like 'Rank_change_1', 'Rank_change_2', 'Pts_change_1', 'Pts_change_2'
    Input: DataFrame with 'Rank_1', 'Rank_2', 'Pts_1', 'Pts_2' and historical data
    Output: DataFrame with trend features
    """

def __match_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute context-based features such as favorite player, close match.
    Adds 'Is_Favorite', 'Close_match', etc.
    Input: DataFrame with odds, best-of format, and score
    Output: DataFrame with context features
    """

def __recovery_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute timing-based features like 'Days_since_last_match' for both players.
    Input: DataFrame with match dates and historical data
    Output: DataFrame with timing features
    """

    
def process_features(path_to_df: str) -> pd.DataFrame:
    df = pd.read_csv(path_to_df)
    df_processed = df.copy()
    
    df_processed = __straight_sets_victory(df_processed)
    df_processed = __season(df_processed)
    
    df_processed.to_csv("../data/processed/atp_tennis_processed.csv", index=False)
    
    return df_processed
    
if __name__ == "main":
    process_features()