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
    Extract the season from the 'Date' column and encode it as a numeric ID.
    Spring=0, Summer=1, Autumn=2, Winter=3.
    Input: DataFrame with 'Date' column in format YYYY-MM-DD
    Output: DataFrame with new column 'Season'
    """

    months = df["Date"].str.split("-").str[1].astype(int)

    df["Season"] = np.select(
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

def __rank_difference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the difference in ranking and points between the two players.
    Adds 'Rank_diff' = Rank_1 - Rank_2 and 'Pts_diff' = Pts_1 - Pts_2.
    Input: DataFrame with 'Rank_1', 'Rank_2', 'Pts_1', 'Pts_2'
    Output: DataFrame with new columns 'Rank_diff', 'Pts_diff'
    """
    
    df["Rank_diff"]= df["Rank_1"] - df["Rank_2"]
    df["Pts_diff"]= df["Pts_1"] - df["Pts_2"]
    df["Odds_diff"]= round(df["Odd_1"] - df["Odd_2"], 2)
    
    return df

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
    
    df['H2H_wins_1'] = round(h2h_wins_1, 2)
    df['H2H_wins_2'] = round(h2h_wins_2, 2)
    df['H2H_diff'] = round(df['H2H_wins_1'] - df['H2H_wins_2'], 2)
    
    return df

    

def __recent_form(df: pd.DataFrame, n_matches: int = 5) -> pd.DataFrame:
    """
    Compute recent form of each player based on last n_matches.
    Adds features like 'Win_pct_1_lastN', 'Win_pct_2_lastN', 'Win_pct_diff_lastN',
    'Sets_won_ratio', 'Games_won_ratio', etc.
    Input: DataFrame with historical match results
    Output: DataFrame with new recent form features
    """
    import re
    
    # Funzione helper per parsare lo score
    def parse_score(score_str):
        """Estrae set vinti da una stringa di score (es. '6-4 6-3' o '6-4 3-6 7-5')"""
        if pd.isna(score_str) or score_str == '':
            return None, None
        
        try:
            # Rimuovi eventuali caratteri speciali o annotazioni (tiebreak, ritiri, etc.)
            score_str = str(score_str).strip()
            score_str = re.sub(r'\([^)]*\)', '', score_str)  # Rimuove contenuto tra parentesi
            score_str = re.sub(r'[A-Za-z]+', '', score_str)  # Rimuove lettere (RET, W/O, etc.)
            
            # Dividi per set
            sets = score_str.split()
            sets_player1 = 0
            sets_player2 = 0
            games_player1 = 0
            games_player2 = 0
            
            for set_score in sets:
                if '-' in set_score:
                    games = set_score.split('-')
                    if len(games) == 2:
                        g1 = int(re.sub(r'\D', '', games[0])) if games[0] else 0
                        g2 = int(re.sub(r'\D', '', games[1])) if games[1] else 0
                        
                        games_player1 += g1
                        games_player2 += g2
                        
                        if g1 > g2:
                            sets_player1 += 1
                        else:
                            sets_player2 += 1
            
            return (sets_player1, sets_player2, games_player1, games_player2)
        except:
            return None, None, None, None
    
    # Ordina cronologicamente
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Inizializza colonne delle feature
    df['Win_pct_1_lastN'] = 0.0
    df['Win_pct_2_lastN'] = 0.0
    df['Win_pct_diff_lastN'] = 0.0
    df['Sets_won_ratio_1'] = 0.0
    df['Sets_won_ratio_2'] = 0.0
    df['Games_won_ratio_1'] = 0.0
    df['Games_won_ratio_2'] = 0.0
    df['Matches_played_1'] = 0
    df['Matches_played_2'] = 0
    
    # Dizionario per memorizzare la storia recente di ogni giocatore
    player_history = {}
    
    for idx, row in df.iterrows():
        player_1 = row['Player_1']
        player_2 = row['Player_2']
        winner = row['Winner']
        score = row.get('Score', '')
        
        # Inizializza storia giocatori se non esiste
        if player_1 not in player_history:
            player_history[player_1] = {
                'wins': [], 
                'sets_won': [], 
                'sets_total': [],
                'games_won': [],
                'games_total': []
            }
        if player_2 not in player_history:
            player_history[player_2] = {
                'wins': [], 
                'sets_won': [], 
                'sets_total': [],
                'games_won': [],
                'games_total': []
            }
        
        # Ottieni forma recente PRIMA di questo match
        hist_1 = player_history[player_1]
        hist_2 = player_history[player_2]
        
        # Calcola statistiche per Player 1
        recent_wins_1 = hist_1['wins'][-n_matches:]
        win_pct_1 = sum(recent_wins_1) / len(recent_wins_1) if recent_wins_1 else 0.0
        matches_played_1 = len(recent_wins_1)
        
        recent_sets_won_1 = sum(hist_1['sets_won'][-n_matches:])
        recent_sets_total_1 = sum(hist_1['sets_total'][-n_matches:])
        sets_ratio_1 = recent_sets_won_1 / recent_sets_total_1 if recent_sets_total_1 > 0 else 0.0
        
        recent_games_won_1 = sum(hist_1['games_won'][-n_matches:])
        recent_games_total_1 = sum(hist_1['games_total'][-n_matches:])
        games_ratio_1 = recent_games_won_1 / recent_games_total_1 if recent_games_total_1 > 0 else 0.0
        
        # Calcola statistiche per Player 2
        recent_wins_2 = hist_2['wins'][-n_matches:]
        win_pct_2 = sum(recent_wins_2) / len(recent_wins_2) if recent_wins_2 else 0.0
        matches_played_2 = len(recent_wins_2)
        
        recent_sets_won_2 = sum(hist_2['sets_won'][-n_matches:])
        recent_sets_total_2 = sum(hist_2['sets_total'][-n_matches:])
        sets_ratio_2 = recent_sets_won_2 / recent_sets_total_2 if recent_sets_total_2 > 0 else 0.0
        
        recent_games_won_2 = sum(hist_2['games_won'][-n_matches:])
        recent_games_total_2 = sum(hist_2['games_total'][-n_matches:])
        games_ratio_2 = recent_games_won_2 / recent_games_total_2 if recent_games_total_2 > 0 else 0.0
        
        # Assegna feature per questo match
        df.at[idx, 'Win_pct_1_lastN'] = win_pct_1
        df.at[idx, 'Win_pct_2_lastN'] = win_pct_2
        df.at[idx, 'Win_pct_diff_lastN'] = win_pct_1 - win_pct_2
        df.at[idx, 'Sets_won_ratio_1'] = sets_ratio_1
        df.at[idx, 'Sets_won_ratio_2'] = sets_ratio_2
        df.at[idx, 'Games_won_ratio_1'] = games_ratio_1
        df.at[idx, 'Games_won_ratio_2'] = games_ratio_2
        df.at[idx, 'Matches_played_1'] = matches_played_1
        df.at[idx, 'Matches_played_2'] = matches_played_2
        
        # Parsa lo score per ottenere dettagli su set e game
        sets_p1, sets_p2, games_p1, games_p2 = parse_score(score)
        
        # Aggiorna storia DOPO questo match
        # Player 1
        hist_1['wins'].append(1 if winner == player_1 else 0)
        if sets_p1 is not None:
            hist_1['sets_won'].append(sets_p1)
            hist_1['sets_total'].append(sets_p1 + sets_p2 if sets_p2 is not None else sets_p1)
            hist_1['games_won'].append(games_p1 if games_p1 is not None else 0)
            hist_1['games_total'].append((games_p1 + games_p2) if (games_p1 is not None and games_p2 is not None) else 0)
        else:
            hist_1['sets_won'].append(0)
            hist_1['sets_total'].append(0)
            hist_1['games_won'].append(0)
            hist_1['games_total'].append(0)
        
        # Player 2
        hist_2['wins'].append(1 if winner == player_2 else 0)
        if sets_p2 is not None:
            hist_2['sets_won'].append(sets_p2)
            hist_2['sets_total'].append(sets_p1 + sets_p2 if sets_p1 is not None else sets_p2)
            hist_2['games_won'].append(games_p2 if games_p2 is not None else 0)
            hist_2['games_total'].append((games_p1 + games_p2) if (games_p1 is not None and games_p2 is not None) else 0)
        else:
            hist_2['sets_won'].append(0)
            hist_2['sets_total'].append(0)
            hist_2['games_won'].append(0)
            hist_2['games_total'].append(0)
    
    return df

    

def __career_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute performance of each player on career.
    Adds features like 'Career_win_pct','Career_lose'
    Input: DataFrame with 'Winner' column and historical match results
    Output: DataFrame with new win/lose performance features
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
    df_processed = __rank_difference(df_processed)
    df_processed = __h2h_features(df_processed)
    df_processed = __recent_form(df_processed)
    
    df_processed.to_csv("../data/processed/atp_tennis_processed.csv", index=False)
    
    return df_processed
    
if __name__ == "main":
    process_features()