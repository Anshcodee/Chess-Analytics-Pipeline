# ♟️ Predict the Match Before the First Move

## 🚀 Project Overview

This project is a full end-to-end data pipeline and machine learning setup designed to predict the outcome of a chess game (White Win, Black Win, or Draw) before the first move is ever made. It handles everything from processing raw game files to staging relational tables in a database and formatting clean, leak-free feature sets for predictive modeling.

The core objective is to process millions of historical Lichess matches, extract meaningful pre-game features (like player ratings, time formats, and openings), and organize the infrastructure so that it scales cleanly without running into data leakage or ID collisions.

## 🧠 How It Works

To handle the sheer volume of data without slowing down, the project is divided into separate, modular components:

- Extraction & Staging: Raw files are parsed locally to extract the core game parameters, player details, and move lists, saving them as lightweight intermediate files.

- Transformation (PySpark): PySpark handles the heavy processing. It runs calculations across the files, builds non-colliding surrogate primary keys using deterministic hashing, and outputs two distinct datasets: a normalized Star Schema for database analysis, and a flat table tailored specifically for training an XGBoost model.

- Database Hydration (MySQL): A relational layer loads the dimension and fact tables into MySQL for structured querying and verification.

- Machine Learning Preparation: The analytics export drops post-game metrics (like total moves) and high-cardinality text columns, structuring the target variables as clean integers so the data is directly compatible with an XGBoost and SHAP framework.

## 📂 Folder Structure

The repository organizes scripts by their explicit functional step (extract, transform, load). The large data binaries are kept locally and ignored by Git to keep the repository light.

```
chess-analytics-pipeline/ 
│ 
├── data/                       # Local storage (ignored by Git) 
│   ├── 01_raw/                 # Inbound raw Lichess files 
│   ├── 01.1_raw_not_needed/    # Backups or alternative sets 
│   ├── 02_staged/              # Cleaned intermediate game chunks 
│   ├── 03_processed/           # Star Schema Parquet files for the database 
│   └── 04_analytics/           # Flattened Parquet tables for ML training 
│ 
├── notebooks/                  # Local notebooks for data auditing and testing 
│   ├── audit_processed_data.ipynb 
│   └── audit_staged_data.ipynb 
│   └── model_testing.ipynb 
│ 
├── src/                        # Main source code split by pipeline step 
│   ├── pipeline/        
│     ├── extract/        
│     │   ├── parse_lichess.py    # Parsers to clean and split raw logs (1st iteration) 
│     │   ├── parse_lichess2.py   #(2nd iteration) 
│     │   └── parse_lichess3.py   #(Final iteration) 
│     ├── transform/                
│     │   └── spark_pipeline.py   # PySpark job for engineering & ID hashing 
│     └── load/                  
│         ├── load_to_mysql.py    # Loads processed Parquet files into MySQL 
│         └── schema.sql          # Target DDL schema definitions 
│   ├── modeling/        
│ 
├── .env                        # Local database credentials (ignored by Git) 
├── .gitignore                   
├── init.py 
├── LICENSE 
├── README.md                    
└── requirements.txt            # Project dependencies 
```

## 💻 Running the Pipeline
To run the pipeline locally, ensure your database environment parameters are configured in your local .env file, install the project dependencies, and execute the components in this sequence:

- Parse the Raw Input Data:
  python3 src/extract/parse_lichess.py
- Execute the PySpark Transformation Job:
  python3 src/transform/spark_pipeline.py
- Load the Relational Star Schema into MySQL:
  python3 src/load/load_to_mysql.py

## 🛠️ Tech Stack
- Data Engineering: Python, PySpark, Pandas, MySQL, SQLAlchemy
- Machine Learning Preparation: XGBoost, Scikit-Learn, Optuna, SHAP, Jupyter 
