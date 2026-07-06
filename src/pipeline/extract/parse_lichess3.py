# Most recent rendition of the parser, 
# aimed to solve the white-black move split being at 58-42 at parse_lichess2.
# Now, it's at 50.3 - 49.6
import zstandard as zstd
import io
import pandas as pd
import os
import time


def parse_lichess_dump_fast(file_path, output_dir, chunk_size=100000):
   """
   A hyper-optimized, pure-Python PGN parser. Bypasses board-state
   validation to achieve disk-speed parsing for Big Data pipelines.
   """
   games_dir = os.path.join(output_dir, "games")
   moves_dir = os.path.join(output_dir, "moves")
   os.makedirs(games_dir, exist_ok=True)
   os.makedirs(moves_dir, exist_ok=True)

   with open(file_path, 'rb') as compressed_file:
       dctx = zstd.ZstdDecompressor()

       with dctx.stream_reader(compressed_file) as reader:
           text_stream = io.TextIOWrapper(reader, encoding='utf-8', errors='replace')

           games_batch, moves_batch = [], []
           chunk_index, game_id = 1, 1

           current_headers = {}
           current_moves_text = []

           print(f"🚀 Spinning up FAST parser... Streaming {file_path}")
           start_time = time.time()

           for line in text_stream:
               line = line.strip()

               # Skip empty lines, but they signal transitions
               if not line:
                   continue

               # 1. Parse Game Metadata (Headers)
               if line.startswith("["):
                   try:
                       # Extract key and value cleanly from [Event "Rated Blitz"]
                       key_part, value_part = line[1:-1].split(' "', 1)
                       current_headers[key_part] = value_part.rstrip('"')
                   except ValueError:
                       pass # Ignore malformed headers safely

               # 2. Accumulate Move Text
               # If it doesn't start with [, it's part of the move sequence
               else:
                   current_moves_text.append(line)

                   # 3. Detect Game End (Lichess ends games with a result marker)
                   if line.endswith(("1-0", "0-1", "1/2-1/2", "*")):

                       # --- Process Game Metadata ---
                       games_batch.append({
                           "game_id": game_id,
                           "event": current_headers.get("Event", "Unknown"),
                           "white": current_headers.get("White", "Unknown"),
                           "black": current_headers.get("Black", "Unknown"),
                           "white_elo": current_headers.get("WhiteElo", "?"),
                           "black_elo": current_headers.get("BlackElo", "?"),
                           "result": current_headers.get("Result", "*"),
                           "time_control": current_headers.get("TimeControl", "Unknown"),
                           "opening": current_headers.get("Opening", "Unknown"),
                           "eco": current_headers.get("ECO", "Unknown")
                       })

                       # --- Process Moves via Token Iterator ---
                       full_move_string = " ".join(current_moves_text)
                       parse_moves_to_batch(full_move_string, game_id, moves_batch)

                       # --- Export Threshold Check ---
                       if len(games_batch) >= chunk_size:
                           export_chunks(games_batch, moves_batch, games_dir, moves_dir, chunk_index)
                           games_batch, moves_batch = [], []
                           chunk_index += 1
                           elapsed = round(time.time() - start_time, 2)
                           print(f"⏱️ Elapsed Time: {elapsed} seconds")

                       # Reset for the next game
                       game_id += 1
                       current_headers = {}
                       current_moves_text = []


           # Catch the final leftover batch
           if games_batch:
               export_chunks(games_batch, moves_batch, games_dir, moves_dir, chunk_index)

   print(f"✅ Complete! Total time: {round(time.time() - start_time, 2)} seconds")


def parse_moves_to_batch(move_string, game_id, moves_batch):
   """A highly efficient token iterator that extracts moves and clocks without regex."""
   # NOTE: the old move_string.replace(".", ". ") was removed. That replace turned the
   # Black-move marker "1..." into "1. . .", so the parser read it as a fresh "White to
   # move" and mislabelled every Black move in any game that carried comments (clocks/evals).
   # Lichess already space-separates move numbers ("1. e4", "1... c5"), so split() is enough.
   tokens = move_string.split()

   move_number = 1
   current_color = "White"

   i = 0
   while i < len(tokens):
       token = tokens[i]

       # Move Number Token: "1." -> White to move, "1..." -> Black to move.
       # The "1..." form shows up whenever a comment sits between the two half-moves,
       # so we read the colour straight from the dot count instead of assuming White.
       if token.endswith('.'):
           stripped = token.rstrip('.')
           if stripped.isdigit():
               move_number = int(stripped)
               dots = len(token) - len(stripped)
               current_color = "White" if dots == 1 else "Black"
           i += 1
           continue

       # End of Game Markers (Ignore them)
       if token in ["1-0", "0-1", "1/2-1/2", "*"]:
           break

       # Comments (Extracting the clock if it exists)
       if token.startswith("{"):
           clock_val = None
           # Scan forward until the closing bracket
           while i < len(tokens):
               if tokens[i] == "[%clk" and i + 1 < len(tokens):
                   clock_val = tokens[i+1].rstrip("]")
               if tokens[i].endswith("}"):
                   break
               i += 1

           # Backfill the clock onto the most recent move
           if clock_val and len(moves_batch) > 0 and moves_batch[-1]["game_id"] == game_id:
               moves_batch[-1]["clock"] = clock_val
           i += 1
           continue

       # Standard Chess Move Token (e.g., "e4", "Nf3")
       if token:
           moves_batch.append({
               "game_id": game_id,
               "move_number": move_number,
               "color": current_color,
               "san": token,
               "clock": None
           })
           current_color = "Black" if current_color == "White" else "White"

       i += 1


def export_chunks(games_data, moves_data, games_dir, moves_dir, chunk_index):
   """Safely converts to Parquet."""
   df_games = pd.DataFrame(games_data)
   df_games['white_elo'] = pd.to_numeric(df_games['white_elo'], errors='coerce').astype('Int64')
   df_games['black_elo'] = pd.to_numeric(df_games['black_elo'], errors='coerce').astype('Int64')
   df_games.to_parquet(os.path.join(games_dir, f"games_{chunk_index:04d}.parquet"), index=False)

   df_moves = pd.DataFrame(moves_data)
   df_moves.to_parquet(os.path.join(moves_dir, f"moves_{chunk_index:04d}.parquet"), index=False)

   print(f"Exported Chunk {chunk_index} -> {len(games_data)} games, {len(moves_data)} moves.")


if __name__ == "__main__":
   BASE_DIR = os.path.dirname(os.path.abspath(__file__))
   # point this at whichever month you're processing
   ZST_FILE_PATH = os.path.join(BASE_DIR, '..', '..', 'data', '01_raw', 'lichess_db_standard_rated_2015-05.pgn.zst')
   OUTPUT_DIRECTORY = os.path.join(BASE_DIR, '..', '..', 'data', '02_staged')

   # 100k chunk size is perfect for pure-Python parsing
   parse_lichess_dump_fast(ZST_FILE_PATH, OUTPUT_DIRECTORY, chunk_size=100000)
