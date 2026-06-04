import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, split, when, count, abs as spark_abs, monotonically_increasing_id
)

def run_pipeline():
    print("Starting Spark ETL job...")
    
    # Just spinning up a standard local session
    spark = SparkSession.builder \
        .appName("Chess Analytics - ETL") \
        .master("local[*]") \
        .getOrCreate()

    # dynamically getting paths so this runs without breaking if the folder moves
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    STAGED_GAMES = os.path.join(BASE_DIR, '..', '..', 'data', '02_staged', 'games', '*.parquet')
    STAGED_MOVES = os.path.join(BASE_DIR, '..', '..', 'data', '02_staged', 'moves', '*.parquet')
    PROCESSED_DIR = os.path.join(BASE_DIR, '..', '..', 'data', '03_processed')

    print("Loading Parquet data...")
    df_games_raw = spark.read.parquet(STAGED_GAMES)
    df_moves = spark.read.parquet(STAGED_MOVES)

    # --- Data Transformations ---
    print("Transforming data...")
    
    # get the absolute diff between ratings
    df_games = df_games_raw.withColumn("elo_diff", spark_abs(col("white_elo") - col("black_elo")))

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

    # get total moves and join back to the main table
    df_move_counts = df_moves.groupBy("game_id").agg(count("*").alias("total_moves"))
    df_games = df_games.join(df_move_counts, on="game_id", how="left")

    # caching this because we reuse it heavily for the dimension tables below
    df_games.cache()

    # --- Dimensional Modeling ---
    print("Creating Dimension tables...")

    df_white = df_games.select(col("white").alias("username"))
    df_black = df_games.select(col("black").alias("username"))
    
    # note for the DB schema: monotonically_increasing_id creates massive 64-bit integers. 
    # casting to bigint here, just remember to use BIGINT in MySQL so the insert doesn't crash.
    df_dim_player = df_white.union(df_black).distinct() \
                            .withColumn("player_id", monotonically_increasing_id().cast("bigint"))
    
    df_dim_opening = df_games.select("eco", "opening").distinct() \
                             .withColumn("opening_id", monotonically_increasing_id().cast("bigint"))

    # --- Fact Table Creation ---
    print("Joining dimensions to Fact table...")
    
    df_fact_game = df_games \
        .join(df_dim_player.withColumnRenamed("username", "white").withColumnRenamed("player_id", "white_player_id"), on="white", how="left") \
        .join(df_dim_player.withColumnRenamed("username", "black").withColumnRenamed("player_id", "black_player_id"), on="black", how="left") \
        .join(df_dim_opening, on=["eco", "opening"], how="left")

    # keep only the keys and metrics we actually need for the final warehouse
    df_fact_game = df_fact_game.select(
        "game_id", "white_player_id", "black_player_id", "opening_id",
        "white_elo", "black_elo", "elo_diff", "result", "game_category", "total_moves"
    )

    # Export 
    print("Saving processed Parquet files...")
    
    df_dim_player.write.mode("overwrite").parquet(os.path.join(PROCESSED_DIR, "dim_player"))
    df_dim_opening.write.mode("overwrite").parquet(os.path.join(PROCESSED_DIR, "dim_opening"))
    df_fact_game.write.mode("overwrite").parquet(os.path.join(PROCESSED_DIR, "fact_game"))
    df_moves.write.mode("overwrite").parquet(os.path.join(PROCESSED_DIR, "fact_moves"))

    print("Spark job finished!")
    spark.stop()

if __name__ == "__main__":
    run_pipeline()