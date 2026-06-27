import os
import glob
import pandas as pd
from sqlalchemy import create_engine

def load_to_mysql():
    print("connecting to mysql...")
    
    # swap these out with your local workbench creds
    db_user = "root"
    db_password = "your_actual_password"
    db_host = "localhost"
    db_name = "chess_schema" 
    
    # create the sqlalchemy engine to talk to mysql
    engine = create_engine(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")

    # dynamically find the processed data folder so this doesn't break if we move it
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(base_dir, '..', '..', 'data', '03_processed')

    # order matters here! dimensions need to go in before the fact table 
    # so the foreign keys don't freak out
    tables = [
        "dim_player",
        "dim_opening",
        "fact_game"
    ]

    for table in tables:
        folder_path = os.path.join(processed_dir, table)
        
        # grab all the parquet parts in the folder
        files = glob.glob(os.path.join(folder_path, "*.parquet"))
        
        if not files:
            print(f"warning: no files found for {table}, skipping...")
            continue

        for file in files:
            print(f"loading {os.path.basename(file)} into {table}...")
            df = pd.read_parquet(file)
            
            # if_exists='append' adds to the empty tables we just made in workbench.
            # chunksize keeps the memory usage safe so the db server doesn't crash on huge inserts.
            try:
                df.to_sql(name=table, con=engine, if_exists='append', index=False, chunksize=10000)
            except Exception as e:
                print(f"failed to load {table}: {e}")
                
    print("done loading all tables!")

if __name__ == "__main__":
    load_to_mysql()