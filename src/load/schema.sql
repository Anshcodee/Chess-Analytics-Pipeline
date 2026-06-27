-- 1. DIMENSION TABLES (The Lookup Context)

CREATE TABLE dim_player (
    player_id BIGINT PRIMARY KEY,
    username VARCHAR(255) NOT NULL,
    
    -- Adding an index here because searching by username will be your most common filter
    INDEX idx_username (username)
);

CREATE TABLE dim_opening (
    opening_id BIGINT PRIMARY KEY,
    eco VARCHAR(10) NOT NULL,
    opening VARCHAR(255) NOT NULL,
    
    -- Indexing the opening name for fast text lookups
    INDEX idx_opening_name (opening)
);

-- 2. FACT TABLE (The Core Event)

CREATE TABLE fact_game (
    game_id BIGINT PRIMARY KEY,
    white_player_id BIGINT,
    black_player_id BIGINT,
    opening_id BIGINT,
    
    white_elo INT,
    black_elo INT,
    elo_diff INT,
    
    result VARCHAR(20),
    game_category VARCHAR(50),
    total_moves INT,
    
    -- Establishing the relational web
    FOREIGN KEY (white_player_id) REFERENCES dim_player(player_id),
    FOREIGN KEY (black_player_id) REFERENCES dim_player(player_id),
    FOREIGN KEY (opening_id) REFERENCES dim_opening(opening_id),
    
    -- Indexing foreign keys is a best practice for fast JOIN performance
    INDEX idx_white_player (white_player_id),
    INDEX idx_black_player (black_player_id),
    INDEX idx_opening (opening_id)
);