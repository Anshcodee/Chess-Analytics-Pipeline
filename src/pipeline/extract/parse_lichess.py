import zstandard as zstd
import chess.pgn
import io
import pandas as pd
import os

def parse_lichess_dump(file_path, output_dir, chunk_size=50000):
    """
    Safely streams a Lichess .pgn.zst file, generates surrogate keys, 
    and exports strictly relational Parquet files with optimized move traversal.
    """
    games_dir = os.path.join(output_dir, "games")
    moves_dir = os.path.join(output_dir, "moves")
    os.makedirs(games_dir, exist_ok=True)
    os.makedirs(moves_dir, exist_ok=True)
    
    with open(file_path, 'rb') as compressed_file:
        dctx = zstd.ZstdDecompressor()
        
        with dctx.stream_reader(compressed_file) as reader:
            # Safe decoding: replaces weird characters 
            text_stream = io.TextIOWrapper(reader, encoding='utf-8', errors='replace')
            
            games_batch = []
            moves_batch = []
            
            chunk_index = 1
            game_id = 1 
            
            print(f"Streaming {file_path}")
            
            while True:
                game = chess.pgn.read_game(text_stream)
                
                if game is None:
                    break 
                
                headers = game.headers
                
                # 1. Extract Game Metadata
                # To catch standard opening (like London or Caro-Kann) games
                game_data = {
                    "game_id": game_id,
                    "event": headers.get("Event", "Unknown"),
                    "white": headers.get("White", "Unknown"),
                    "black": headers.get("Black", "Unknown"),
                    "white_elo": headers.get("WhiteElo", "?"),
                    "black_elo": headers.get("BlackElo", "?"),
                    "result": headers.get("Result", "*"),
                    "time_control": headers.get("TimeControl", "Unknown"),
                    "opening": headers.get("Opening", "Unknown"),
                    "eco": headers.get("ECO", "Unknown")
                }
                games_batch.append(game_data)
                
                # 2. Extract Move Flow (Optimized)
                node = game
                ply = 1 # A "ply" is one half-move (e.g., White's turn is ply 1, Black's is ply 2)
                
                while node.variations:
                    next_node = node.variation(0)
                    
                    # Math to figure out move number and color without recalculating board state
                    move_number = (ply + 1) // 2
                    color = "White" if ply % 2 != 0 else "Black"
                    
                    move_data = {
                        "game_id": game_id,
                        "move_number": move_number,
                        "color": color,
                        "san": next_node.san(),
                        "clock": next_node.clock()
                    }
                    moves_batch.append(move_data)
                    
                    node = next_node
                    ply += 1
                        
                # 3. Export and flush memory
                if len(games_batch) >= chunk_size:
                    export_chunks(games_batch, moves_batch, games_dir, moves_dir, chunk_index)
                    games_batch = [] 
                    moves_batch = [] 
                    chunk_index += 1
                
                game_id += 1
            
            # Catch the leftovers!
            if games_batch:
                export_chunks(games_batch, moves_batch, games_dir, moves_dir, chunk_index)
                
    print("Done! Data is strictly typed and ready for the cluster.")

def export_chunks(games_data, moves_data, games_dir, moves_dir, chunk_index):
    """Converts to DataFrames, enforces strict nullable integer typing, and exports."""
    
    # --- Games ---
    df_games = pd.DataFrame(games_data)
    
    # FIX: Use 'Int64' to allow integers AND nulls, preventing the float conversion issue
    df_games['white_elo'] = pd.to_numeric(df_games['white_elo'], errors='coerce').astype('Int64')
    df_games['black_elo'] = pd.to_numeric(df_games['black_elo'], errors='coerce').astype('Int64')
    
    games_file = os.path.join(games_dir, f"games_chunk_{chunk_index:04d}.parquet")
    df_games.to_parquet(games_file, index=False)
    
    # --- Moves ---
    df_moves = pd.DataFrame(moves_data)
    moves_file = os.path.join(moves_dir, f"moves_chunk_{chunk_index:04d}.parquet")
    df_moves.to_parquet(moves_file, index=False)
    
    print(f"Exported Chunk {chunk_index} -> {len(games_data)} games, {len(moves_data)} moves.")

# Execution!
if __name__ == "__main__":
    ZST_FILE_PATH = "data/01_raw/lichess_db_standard_rated_2014-01.pgn.zst" 
    OUTPUT_DIRECTORY = "./data/02_staged"
    
    # Bumped to 50k to avoid creating too many tiny files (PySpark prefers larger partitions)
    parse_lichess_dump(ZST_FILE_PATH, OUTPUT_DIRECTORY, chunk_size=50000)