import os
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, when, count, abs as spark_abs, xxhash64, concat_ws


def run_pipeline():
   print("starting spark etl job...")

   # allocate 4gb of ram to the spark driver to prevent memory throttling
   spark = SparkSession.builder \
       .appName("Chess Analytics - ETL") \
       .master("local[*]") \
       .config("spark.driver.memory", "4g") \
       .config("spark.executor.memory", "4g") \
       .getOrCreate()

   # dynamically getting paths so this runs without breaking if the folder moves
   BASE_DIR = os.path.dirname(os.path.abspath(__file__))
   STAGED_GAMES = os.path.join(BASE_DIR, '..', '..', 'data', '02_staged', 'games', '*.parquet')
   STAGED_MOVES = os.path.join(BASE_DIR, '..', '..', 'data', '02_staged', 'moves', '*.parquet')

   PROCESSED_DIR = os.path.join(BASE_DIR, '..', '..', 'data', '03_processed')
   ANALYTICS_DIR = os.path.join(BASE_DIR, '..', '..', 'data', '04_analytics')

   # generate a timestamp (YYYYMMDD_HHMM) to tag the ML archive files
   timestamp = datetime.now().strftime("%Y%m%d_%H%M")

   print("loading parquet data...")
   df_games_raw = spark.read.parquet(STAGED_GAMES)
   df_moves = spark.read.parquet(STAGED_MOVES)

   # --- data transformations ---
   print("transforming data...")

   # SIGNED rating gap (white - black). signed matters: it tells the model WHO is
   # favoured, not just how big the gap is. this is the strongest pre-game feature.
   df_games = df_games_raw.withColumn("elo_diff", col("white_elo") - col("black_elo"))

   # safe split: if time_control is "-", make it null BEFORE trying to split and cast
   df_games = df_games.withColumn("base_time",
       when(col("time_control") == "-", None)
       .otherwise(split(col("time_control"), r"\+").getItem(0).cast("int"))
   ).withColumn("increment",
       when(col("time_control") == "-", None)
       .otherwise(split(col("time_control"), r"\+").getItem(1).cast("int"))
   )

   # bucket the game speeds. added a check for nulls just in case there are untimed games
   df_games = df_games.withColumn("game_category",
       when(col("base_time").isNull(), "Untimed")
       .when(col("base_time") < 180, "Bullet")
       .when((col("base_time") >= 180) & (col("base_time") < 600), "Blitz")
       .when((col("base_time") >= 600) & (col("base_time") < 1800), "Rapid")
       .otherwise("Classical")
   )

   # get total moves and join back to the main table.
   # WARNING: total_moves is only known AFTER the game ends, so it is NOT a pre-game
   # feature. it's kept here for the star schema / EDA, but the ML model must NOT
   # train on it (it leaks the outcome).
   df_move_counts = df_moves.groupBy("game_id").agg(count("*").alias("total_moves"))
   df_games = df_games.join(df_move_counts, on="game_id", how="left")

   # caching this because we reuse it heavily for the dimension tables below
   df_games.cache()

   # --- dimensional modeling ---
   print("creating dimension tables...")

   df_white = df_games.select(col("white").alias("username"))
   df_black = df_games.select(col("black").alias("username"))

   # deterministic ID generation using hashes.
   # xxhash64 (64-bit) instead of hash (32-bit): with hundreds of thousands of
   # players a 32-bit space collides almost for sure, which breaks the PRIMARY KEY
   # load in mysql once you scale past this one month.
   df_dim_player = df_white.union(df_black).distinct() \
                       .withColumn("player_id", spark_abs(xxhash64(col("username"))).cast("bigint"))

   # the table grain is (eco, opening), so the key MUST be built from BOTH columns.
   # hashing "opening" alone gave the same id to different (eco, opening) rows ->
   # duplicate primary key -> dim_opening failed to load -> fact_game FK then failed.
   df_dim_opening = df_games.select("eco", "opening").distinct() \
                        .withColumn("opening_id", spark_abs(xxhash64(concat_ws("|", col("eco"), col("opening")))).cast("bigint"))

   # --- fact table creation ---
   print("joining dimensions to fact table...")

   df_fact_game = df_games \
       .join(df_dim_player.withColumnRenamed("username", "white").withColumnRenamed("player_id", "white_player_id"), on="white", how="left") \
       .join(df_dim_player.withColumnRenamed("username", "black").withColumnRenamed("player_id", "black_player_id"), on="black", how="left") \
       .join(df_dim_opening, on=["eco", "opening"], how="left")

   # keep only the keys and metrics we actually need for the final warehouse
   df_fact_game = df_fact_game.select(
       "game_id", "white_player_id", "black_player_id", "opening_id",
       "white_elo", "black_elo", "elo_diff", "result", "game_category", "total_moves"
   )

   # --- export block ---

   # 1. star schema -> transit folder for the mysql loader (overwrite each run)
   print("saving processed data for mysql loader...")
   df_dim_player.write.mode("overwrite").parquet(os.path.join(PROCESSED_DIR, "dim_player"))
   df_dim_opening.write.mode("overwrite").parquet(os.path.join(PROCESSED_DIR, "dim_opening"))
   df_fact_game.write.mode("overwrite").parquet(os.path.join(PROCESSED_DIR, "fact_game"))
   df_moves.write.mode("overwrite").parquet(os.path.join(PROCESSED_DIR, "fact_moves"))  # loader doesn't load this yet; here if you want moves in mysql later

   # 2. ONE flat feature table for the ml model (no hashes, no star schema).
   # df_games already carries the raw text (white/black/eco/opening) plus the
   # engineered pre-game features (elo_diff, base_time, increment, game_category),
   # so it is the perfect flat input for xgboost.
   print(f"archiving flat feature table to analytics (batch: {timestamp})...")
   df_games.write.mode("overwrite").parquet(os.path.join(ANALYTICS_DIR, f"features_{timestamp}"))

   print("spark job finished!")
   spark.stop()


if __name__ == "__main__":
   run_pipeline()
   