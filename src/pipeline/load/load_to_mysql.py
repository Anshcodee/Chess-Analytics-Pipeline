import os
import glob
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from urllib.parse import quote_plus #Add this if the db password 

# load credentials from the .env file into the system environment
load_dotenv()

def load_to_mysql():
    print("connecting to mysql...")
    
    # pull the credentials securely from the environment
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_name = os.getenv("DB_NAME")
    
    # fail-safe: check if the .env loaded correctly
    if not db_password:
        raise ValueError("database password not found! check your .env file.")
    
    # URL-encode the password so special characters (like @ or #) don't break the connection string
    safe_password = quote_plus(db_password)
    
    # create the sqlalchemy engine using the safe password
    engine = create_engine(f"mysql+pymysql://{db_user}:{safe_password}@{db_host}/{db_name}")

    # dynamically find the processed data folder
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(base_dir, '..', '..', 'data', '03_processed')

    # order matters here, dimensions need to go in before the fact table 
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