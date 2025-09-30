import pandas as pd
import numpy as np

def __rank_difference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the difference in ranking and points between the two players.
    Adds 'Rank_diff' = Rank_1 - Rank_2 and 'Pts_diff' = Pts_1 - Pts_2.
    Input: DataFrame with 'Rank_1', 'Rank_2', 'Pts_1', 'Pts_2'
    Output: DataFrame with new columns 'Rank_diff', 'Pts_diff'
    """
    
    df["Rank_diff"]= df["Rank_1"] - df["Rank_2"]
    df["Pts_diff"]= df["Pts_1"] - df["Pts_2"]
    
    return df

def __odds_difference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the difference in betting odds between the two players.
    Adds 'Odds_diff' = Odd_1 - Odd_2
    Input: DataFrame with 'Odd_1' and 'Odd_2'
    Output: DataFrame with new column 'Odds_diff'
    """
    
    df["Odds_diff"]= round(df["Odd_1"] - df["Odd_2"], 2)
    
    return df

def __h2h_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute head-to-head statistics between Player_1 and Player_2.
    Adds columns for previous wins and differences: 'H2H_wins_1', 'H2H_wins_2', 'H2H_diff'
    Input: DataFrame with historical match results
    Output: DataFrame with new H2H feature columns
    """
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Initialize columns
    h2h_wins_1 = []
    h2h_wins_2 = []
    
    for idx in range(len(df)):
        player_1 = df.loc[idx, 'Player_1']
        player_2 = df.loc[idx, 'Player_2']
        
        # Filter all previous matches between these players
        previous_matches = df.iloc[:idx]
        h2h_matches = previous_matches[
            ((previous_matches['Player_1'] == player_1) & (previous_matches['Player_2'] == player_2)) |
            ((previous_matches['Player_1'] == player_2) & (previous_matches['Player_2'] == player_1))
        ]
        
        # Count wins for each player
        wins_1 = (
            ((h2h_matches['Player_1'] == player_1) & (h2h_matches['Winner'] == player_1)).sum() +
            ((h2h_matches['Player_2'] == player_1) & (h2h_matches['Winner'] == player_1)).sum()
        )
        
        wins_2 = (
            ((h2h_matches['Player_1'] == player_2) & (h2h_matches['Winner'] == player_2)).sum() +
            ((h2h_matches['Player_2'] == player_2) & (h2h_matches['Winner'] == player_2)).sum()
        )
        
        h2h_wins_1.append(wins_1)
        h2h_wins_2.append(wins_2)
    
    df['H2H_wins_1'] = h2h_wins_1
    df['H2H_wins_2'] = h2h_wins_2
    df['H2H_diff'] = df['H2H_wins_1'] - df['H2H_wins_2']
    
    return df

def __recent_form(df: pd.DataFrame, n_matches: int = 5) -> pd.DataFrame:
    """
    Compute recent form of each player based on the last n_matches.
    Adds features like 'Win_pct_1_lastN', 'Win_pct_2_lastN', 'Win_pct_diff_lastN',
    and number of matches played.
    
    Input: DataFrame with historical match results
    Output: DataFrame with new recent form features
    """
    
    # Sort matches chronologically
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Initialize feature columns
    df['Win_pct_1_lastN'] = 0.0
    df['Win_pct_2_lastN'] = 0.0
    df['Win_pct_diff_lastN'] = 0.0
    df['Matches_played_1'] = 0
    df['Matches_played_2'] = 0
    
    # Dictionary to store each player's recent match history
    player_history = {}
    
    for idx, row in df.iterrows():
        player_1 = row['Player_1']
        player_2 = row['Player_2']
        winner = row['Winner']
        
        # Initialize player history if not already present
        if player_1 not in player_history:
            player_history[player_1] = {'wins': []}
        if player_2 not in player_history:
            player_history[player_2] = {'wins': []}
        
        # Retrieve player histories BEFORE this match
        hist_1 = player_history[player_1]
        hist_2 = player_history[player_2]
        
        # Compute recent form stats for Player 1
        recent_wins_1 = hist_1['wins'][-n_matches:]
        win_pct_1 = sum(recent_wins_1) / len(recent_wins_1) if recent_wins_1 else 0.0
        matches_played_1 = len(recent_wins_1)
        
        # Compute recent form stats for Player 2
        recent_wins_2 = hist_2['wins'][-n_matches:]
        win_pct_2 = sum(recent_wins_2) / len(recent_wins_2) if recent_wins_2 else 0.0
        matches_played_2 = len(recent_wins_2)
        
        # Assign computed features to this match
        df.at[idx, 'Win_pct_1_lastN'] = round(win_pct_1, 2)
        df.at[idx, 'Win_pct_2_lastN'] = round(win_pct_2, 2)
        df.at[idx, 'Win_pct_diff_lastN'] = round(win_pct_1 - win_pct_2, 2)
        df.at[idx, 'Matches_played_1'] = matches_played_1
        df.at[idx, 'Matches_played_2'] = matches_played_2
        
        # Update player history AFTER this match
        hist_1['wins'].append(1 if winner == player_1 else 0)
        hist_2['wins'].append(1 if winner == player_2 else 0)
    
    return df

def __career_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute performance of each player on career.
    Adds features like 'Career_win_pct','Career_lose'
    Input: DataFrame with 'Winner' column and historical match results
    Output: DataFrame with new win/lose performance features
    """
    # Sort chronologically
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Initialize feature columns
    df['Career_wins_1'] = 0
    df['Career_losses_1'] = 0
    df['Career_matches_1'] = 0
    df['Career_win_pct_1'] = 0.0
    
    df['Career_wins_2'] = 0
    df['Career_losses_2'] = 0
    df['Career_matches_2'] = 0
    df['Career_win_pct_2'] = 0.0
    
    df['Career_win_pct_diff'] = 0.0
    df['Career_matches_diff'] = 0
    
    # Dictionary to track career statistics
    career_stats = {}
    
    for idx, row in df.iterrows():
        player_1 = row['Player_1']
        player_2 = row['Player_2']
        winner = row['Winner']
        
        # Initialize statistics if player doesn't exist
        if player_1 not in career_stats:
            career_stats[player_1] = {'wins': 0, 'losses': 0, 'matches': 0}
        if player_2 not in career_stats:
            career_stats[player_2] = {'wins': 0, 'losses': 0, 'matches': 0}
        
        # Get statistics BEFORE this match
        stats_1 = career_stats[player_1]
        stats_2 = career_stats[player_2]
        
        # Assign features for Player 1
        df.at[idx, 'Career_wins_1'] = stats_1['wins']
        df.at[idx, 'Career_losses_1'] = stats_1['losses']
        df.at[idx, 'Career_matches_1'] = stats_1['matches']
        df.at[idx, 'Career_win_pct_1'] = (
            round(stats_1['wins'] / stats_1['matches'] if stats_1['matches'] > 0 else 0.0, 2)
        )
        
        # Assign features for Player 2
        df.at[idx, 'Career_wins_2'] = stats_2['wins']
        df.at[idx, 'Career_losses_2'] = stats_2['losses']
        df.at[idx, 'Career_matches_2'] = stats_2['matches']
        df.at[idx, 'Career_win_pct_2'] = (
            round(stats_2['wins'] / stats_2['matches'] if stats_2['matches'] > 0 else 0.0, 2)
        )
        
        # Calculate differences
        df.at[idx, 'Career_win_pct_diff'] = (
            round(df.at[idx, 'Career_win_pct_1'] - df.at[idx, 'Career_win_pct_2'], 2)
        )
        df.at[idx, 'Career_matches_diff'] = (
            round(stats_1['matches'] - stats_2['matches'], 2)
        )
        
        # Update statistics AFTER this match
        career_stats[player_1]['matches'] += 1
        career_stats[player_2]['matches'] += 1
        
        if winner == player_1:
            career_stats[player_1]['wins'] += 1
            career_stats[player_2]['losses'] += 1
        elif winner == player_2:
            career_stats[player_2]['wins'] += 1
            career_stats[player_1]['losses'] += 1
    
    return df

def __surface_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute performance of each player on the match surface.
    Adds features like 'Surface_win_pct_1', 'Surface_win_pct_2', 'Surface_win_diff'
    Input: DataFrame with 'Surface' column and historical match results
    Output: DataFrame with new surface performance features
    """
    
    # Sort chronologically
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Initialize feature columns
    df['Surface_wins_1'] = 0
    df['Surface_losses_1'] = 0
    df['Surface_matches_1'] = 0
    df['Surface_win_pct_1'] = 0.0
    
    df['Surface_wins_2'] = 0
    df['Surface_losses_2'] = 0
    df['Surface_matches_2'] = 0
    df['Surface_win_pct_2'] = 0.0
    
    df['Surface_win_pct_diff'] = 0.0
    df['Surface_matches_diff'] = 0
    
    # Dictionary to track surface-specific statistics
    surface_stats = {}
    
    for idx, row in df.iterrows():
        player_1 = row['Player_1']
        player_2 = row['Player_2']
        surface = row['Surface']
        winner = row['Winner']
        
        # Initialize if not exists
        if player_1 not in surface_stats:
            surface_stats[player_1] = {}
        if player_2 not in surface_stats:
            surface_stats[player_2] = {}
        if surface not in surface_stats[player_1]:
            surface_stats[player_1][surface] = {'wins': 0, 'losses': 0, 'matches': 0}
        if surface not in surface_stats[player_2]:
            surface_stats[player_2][surface] = {'wins': 0, 'losses': 0, 'matches': 0}
        
        # Get stats BEFORE this match
        stats_1 = surface_stats[player_1][surface]
        stats_2 = surface_stats[player_2][surface]
        
        # Assign features for Player 1
        df.at[idx, 'Surface_wins_1'] = stats_1['wins']
        df.at[idx, 'Surface_losses_1'] = stats_1['losses']
        df.at[idx, 'Surface_matches_1'] = stats_1['matches']
        df.at[idx, 'Surface_win_pct_1'] = (
            round(stats_1['wins'] / stats_1['matches'] if stats_1['matches'] > 0 else 0.0, 2)
        )
        
        # Assign features for Player 2
        df.at[idx, 'Surface_wins_2'] = stats_2['wins']
        df.at[idx, 'Surface_losses_2'] = stats_2['losses']
        df.at[idx, 'Surface_matches_2'] = stats_2['matches']
        df.at[idx, 'Surface_win_pct_2'] = (
            round(stats_2['wins'] / stats_2['matches'] if stats_2['matches'] > 0 else 0.0, 2)
        )
        
        # Calculate differences
        df.at[idx, 'Surface_win_pct_diff'] = (
            round(df.at[idx, 'Surface_win_pct_1'] - df.at[idx, 'Surface_win_pct_2'], 2)
        )
        df.at[idx, 'Surface_matches_diff'] = (
            stats_1['matches'] - stats_2['matches']
        )
        
        # Update AFTER this match
        surface_stats[player_1][surface]['matches'] += 1
        surface_stats[player_2][surface]['matches'] += 1
        
        if winner == player_1:
            surface_stats[player_1][surface]['wins'] += 1
            surface_stats[player_2][surface]['losses'] += 1
        elif winner == player_2:
            surface_stats[player_2][surface]['wins'] += 1
            surface_stats[player_1][surface]['losses'] += 1
    
    return df

def __tournament_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute performance of each player in the specific tournament.
    Adds features like 'Tournament_win_pct_1', 'Tournament_win_pct_2', 'Tournament_win_diff'
    Input: DataFrame with 'Tournament' column and historical match results
    Output: DataFrame with new tournament performance features
    """
    
    # Sort chronologically
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Initialize feature columns
    df['Tournament_wins_1'] = 0
    df['Tournament_losses_1'] = 0
    df['Tournament_matches_1'] = 0
    df['Tournament_win_pct_1'] = 0.0
    
    df['Tournament_wins_2'] = 0
    df['Tournament_losses_2'] = 0
    df['Tournament_matches_2'] = 0
    df['Tournament_win_pct_2'] = 0.0
    
    df['Tournament_win_pct_diff'] = 0.0
    df['Tournament_matches_diff'] = 0
    
    # Dictionary to track tournament-specific statistics
    tournament_stats = {}
    
    for idx, row in df.iterrows():
        player_1 = row['Player_1']
        player_2 = row['Player_2']
        tournament = row['Tournament']
        winner = row['Winner']
        
        # Initialize if not exists
        if player_1 not in tournament_stats:
            tournament_stats[player_1] = {}
        if player_2 not in tournament_stats:
            tournament_stats[player_2] = {}
        if tournament not in tournament_stats[player_1]:
            tournament_stats[player_1][tournament] = {'wins': 0, 'losses': 0, 'matches': 0}
        if tournament not in tournament_stats[player_2]:
            tournament_stats[player_2][tournament] = {'wins': 0, 'losses': 0, 'matches': 0}
        
        # Get stats BEFORE this match
        stats_1 = tournament_stats[player_1][tournament]
        stats_2 = tournament_stats[player_2][tournament]
        
        # Assign features for Player 1
        df.at[idx, 'Tournament_wins_1'] = stats_1['wins']
        df.at[idx, 'Tournament_losses_1'] = stats_1['losses']
        df.at[idx, 'Tournament_matches_1'] = stats_1['matches']
        df.at[idx, 'Tournament_win_pct_1'] = (
            round(stats_1['wins'] / stats_1['matches'] if stats_1['matches'] > 0 else 0.0, 2)
        )
        
        # Assign features for Player 2
        df.at[idx, 'Tournament_wins_2'] = stats_2['wins']
        df.at[idx, 'Tournament_losses_2'] = stats_2['losses']
        df.at[idx, 'Tournament_matches_2'] = stats_2['matches']
        df.at[idx, 'Tournament_win_pct_2'] = (
            round(stats_2['wins'] / stats_2['matches'] if stats_2['matches'] > 0 else 0.0, 2)
        )
        
        # Calculate differences
        df.at[idx, 'Tournament_win_pct_diff'] = (
            round(df.at[idx, 'Tournament_win_pct_1'] - df.at[idx, 'Tournament_win_pct_2'], 2)
        )
        df.at[idx, 'Tournament_matches_diff'] = (
            stats_1['matches'] - stats_2['matches']
        )
        
        # Update AFTER this match
        tournament_stats[player_1][tournament]['matches'] += 1
        tournament_stats[player_2][tournament]['matches'] += 1
        
        if winner == player_1:
            tournament_stats[player_1][tournament]['wins'] += 1
            tournament_stats[player_2][tournament]['losses'] += 1
        elif winner == player_2:
            tournament_stats[player_2][tournament]['wins'] += 1
            tournament_stats[player_1][tournament]['losses'] += 1
    
    return df

def __rank_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ranking or points change over recent period.
    Adds features like 'Rank_change_1', 'Rank_change_2', 'Pts_change_1', 'Pts_change_2'
    Input: DataFrame with 'Rank_1', 'Rank_2', 'Pts_1', 'Pts_2' and historical data
    Output: DataFrame with trend features
    """
    
    return df

def __recovery_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute timing-based features like 'Days_since_last_match' for both players.
    Input: DataFrame with match dates and historical data
    Output: DataFrame with timing features
    """
    
    # Sort matches chronologically
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Ensure Date column is in datetime format
    if df['Date'].dtype != 'datetime64[ns]':
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Initialize feature columns
    df['Days_since_last_match_1'] = np.nan
    df['Days_since_last_match_2'] = np.nan
    df['Days_diff'] = 0.0
    
    # Dictionary to track each player's last match date
    last_match_date = {}
    
    for idx, row in df.iterrows():
        player_1 = row['Player_1']
        player_2 = row['Player_2']
        current_date = row['Date']
        
        # Calculate days since last match for Player 1
        if player_1 in last_match_date:
            days_since_1 = (current_date - last_match_date[player_1]).days
            df.at[idx, 'Days_since_last_match_1'] = days_since_1
        else:
            # First match for this player in the dataset
            df.at[idx, 'Days_since_last_match_1'] = np.nan
        
        # Calculate days since last match for Player 2
        if player_2 in last_match_date:
            days_since_2 = (current_date - last_match_date[player_2]).days
            df.at[idx, 'Days_since_last_match_2'] = days_since_2
        else:
            # First match for this player in the dataset
            df.at[idx, 'Days_since_last_match_2'] = np.nan
        
        # Calculate difference in rest days (positive means player 1 is more rested)
        if pd.notna(df.at[idx, 'Days_since_last_match_1']) and pd.notna(df.at[idx, 'Days_since_last_match_2']):
            df.at[idx, 'Days_diff'] = df.at[idx, 'Days_since_last_match_1'] - df.at[idx, 'Days_since_last_match_2']
        
        # Update last match date for both players
        last_match_date[player_1] = current_date
        last_match_date[player_2] = current_date
    
    # Fill NaN values with median - FIX: use assignment instead of inplace
    median_rest_1 = df['Days_since_last_match_1'].median()
    median_rest_2 = df['Days_since_last_match_2'].median()
    
    df['Days_since_last_match_1'] = df['Days_since_last_match_1'].fillna(median_rest_1)
    df['Days_since_last_match_2'] = df['Days_since_last_match_2'].fillna(median_rest_2)
    
    return df

def __match_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute context-based features such as favorite player, close match.
    Adds 'Is_Favorite', 'Close_match', etc.
    Input: DataFrame with odds, best-of format, and score
    Output: DataFrame with context features
    """
    
    return df

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
    Extract the season from the 'Date' column and encode it as a numeric ID.
    Spring=0, Summer=1, Autumn=2, Winter=3.
    Input: DataFrame with 'Date' column in datetime format
    Output: DataFrame with new column 'Season'
    """
    
    # Ensure Date is in datetime format
    if df['Date'].dtype != 'datetime64[ns]':
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Extract month using .dt accessor (not .str since Date is datetime)
    months = df['Date'].dt.month
    
    df['Season'] = np.select(
        [
            months.isin([3, 4, 5]),        # Spring
            months.isin([6, 7, 8]),        # Summer
            months.isin([9, 10, 11]),      # Autumn
            months.isin([12, 1, 2])        # Winter
        ],
        [0, 1, 2, 3],
        default=-1
    )
    
    return df

def process_features(path_to_df: str) -> pd.DataFrame:
    df = pd.read_csv(path_to_df)
    df_processed = df.copy()
    
    df_processed = __rank_difference(df_processed)
    df_processed = __odds_difference(df_processed)
    df_processed = __h2h_features(df_processed)
    df_processed = __recent_form(df_processed)
    df_processed = __career_performance(df_processed)
    df_processed = __surface_performance(df_processed)
    df_processed = __tournament_performance(df_processed)
    df_processed = __rank_trends(df_processed)
    df_processed = __recovery_time(df_processed)
    df_processed = __match_context(df_processed)
    df_processed = __straight_sets_victory(df_processed)
    df_processed = __season(df_processed)
    
    df_processed.to_csv("../data/processed/atp_tennis_processed.csv", index=False)
    
    return df_processed
    
if __name__ == "main":
    process_features()