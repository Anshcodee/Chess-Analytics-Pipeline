import pandas as pd
import os

def audit_staged_data():
    # 1. Define paths to the very first chunk
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    games_file = os.path.join(BASE_DIR, '..', '..', 'data', '02_staged', 'games', 'games_0001.parquet')
    moves_file = os.path.join(BASE_DIR, '..', '..', 'data', '02_staged', 'moves', 'moves_0001.parquet')

    print("Auditing Staged Data...\n")

    # 2. Load the Parquet files
    df_games = pd.read_parquet(games_file)
    df_moves = pd.read_parquet(moves_file)

    # 3. Check the Games Schema & Missing Values
    print("--- GAMES TABLE: HEAD ---")
    print(df_games[['game_id', 'white', 'black', 'white_elo', 'opening']].head())
    print("\n--- GAMES TABLE: DATA TYPES ---")
    print(df_games.dtypes)
    
    # 4. Check the Moves Schema
    print("\n--- MOVES TABLE: HEAD ---")
    print(df_moves.head())

    # 5. The Ultimate Relational Test: Join Check
    # Let's pull Game #1744 and see if the moves line up correctly
    test_game_id = 1744
    print(f"\n RELATIONAL INTEGRITY CHECK: GAME #{test_game_id} ---")
    
    game_meta = df_games[df_games['game_id'] == test_game_id]
    game_moves = df_moves[df_moves['game_id'] == test_game_id]
    
    print(f"Match: {game_meta['white'].values[0]} vs {game_meta['black'].values[0]}")
    print(f"Total Moves Captured: {len(game_moves)}")
    print("First 5 moves:")
    print(game_moves[['move_number', 'color', 'san']].head())

if __name__ == "__main__":
    audit_staged_data()