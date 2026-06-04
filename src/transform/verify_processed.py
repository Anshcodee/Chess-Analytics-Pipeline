import pandas as pd
import os
import glob

def run_quick_audit():
    # grab the directory paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fact_dir = os.path.join(base_dir, '..', '..', 'data', '03_processed', 'fact_game')

    print("Running quick sanity check on processed data...")

    # BIG DATA TRAP: Don't read the whole directory with pandas! 
    # Spark writes multiple part-0000X.parquet files. Let's just grab the first one to audit.
    try:
        # find all actual parquet files in the folder (ignores the _SUCCESS file)
        parquet_files = glob.glob(os.path.join(fact_dir, "*.parquet"))
        
        if not parquet_files:
            print(f"Error: No parquet files found in {fact_dir}. Did the Spark job finish?")
            return
            
        sample_file = parquet_files[0]
        df = pd.read_parquet(sample_file)

    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    print("\n--- Game Category Breakdown ---")
    # dropna=False is clutch here so we can see if our null handling actually worked
    print(df['game_category'].value_counts(dropna=False))
    
    print("\n--- Schema Check ---")
    print(df.dtypes)
    
    print("\n--- Sample Row ---")
    # Transposing (.T) makes it much easier to read a single row in the terminal
    print(df.head(1).T) 

if __name__ == "__main__":
    run_quick_audit()