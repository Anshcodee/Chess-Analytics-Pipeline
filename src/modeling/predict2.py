import joblib
import pandas as pd
import numpy as np
import os
import re

class ChessPredictor:
    def __init__(self, model_path: str):
        print("Initializing Defensive Inference Engine...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        self.model = joblib.load(model_path)
        
        self.expected_features = [
            'white_elo', 'black_elo', 'elo_diff', 'base_time', 'increment',
            'abs_elo_diff', 'avg_elo', 'white_is_higher', 'elo_exp_white',
            'has_increment', 'est_total_time', 'increment_reliance',
            'white_smoothed_winrate', 'black_smoothed_winrate',
            'white_recent_form', 'black_recent_form',
            'white_is_first_game', 'black_is_first_game',
            'game_category_Bullet', 'game_category_Classical', 
            'game_category_Rapid', 'game_category_Untimed'
        ]

    def _sanitize_elo(self, elo_input, default_elo=1500) -> int:
        """Sanitizes raw Elo inputs, handling Lichess provisional '?' marks and empty values."""
        if pd.isna(elo_input) or elo_input is None:
            return default_elo
        
        # Clean string inputs (e.g., "1500?")
        elo_str = str(elo_input).strip()
        cleaned_str = re.sub(r'[^\d-]', '', elo_str) # Strip out non-numeric characters
        
        try:
            elo_val = int(cleaned_str)
            # Clip to realistic human chess boundaries (100 to 3500)
            return int(np.clip(elo_val, 100, 3500))
        except ValueError:
            return default_elo

    def _sanitize_time(self, time_input) -> float:
        """Ensures clock values are non-negative floats."""
        if pd.isna(time_input) or time_input is None:
            return 0.0
        try:
            val = float(time_input)
            return max(0.0, val)
        except ValueError:
            return 0.0

    def predict_match(self, match_data: dict) -> dict:
        """
        Ingests, cleans, validates, and predicts a single match outcome.
        Safely catches schema and value errors.
        """
        # 1. Defensive Sanitization Layer
        w_elo = self._sanitize_elo(match_data.get('white_elo'))
        b_elo = self._sanitize_elo(match_data.get('black_elo'))
        base_time = self._sanitize_time(match_data.get('base_time'))
        inc = self._sanitize_time(match_data.get('increment'))
        
        elo_diff = w_elo - b_elo
        
        # 2. Base Feature Computations
        features = {
            'white_elo': w_elo,
            'black_elo': b_elo,
            'elo_diff': elo_diff,
            'base_time': base_time,
            'increment': inc,
            'abs_elo_diff': abs(elo_diff),
            'avg_elo': (w_elo + b_elo) / 2,
            'white_is_higher': int(w_elo > b_elo),
            'elo_exp_white': 1 / (1 + 10 ** ((b_elo - w_elo) / 400)),
            'has_increment': int(inc > 0),
            'est_total_time': base_time + (inc * 40),
            'increment_reliance': (inc * 40) / (base_time + 1)
        }
        
        # 3. Temporal History Check with Type-Casting Guardrails
        global_avg = 0.50
        for side in ['white', 'black']:
            hist_key = f"{side[0]}_history"
            form_key = f"{side[0]}_form"
            
            try:
                hist_val = float(match_data[hist_key]) if hist_key in match_data and match_data[hist_key] is not None else global_avg
                form_val = float(match_data[form_key]) if form_key in match_data and match_data[form_key] is not None else global_avg
                # Keep probabilities inside logical [0, 1] constraints
                hist_val = float(np.clip(hist_val, 0.0, 1.0))
                form_val = float(np.clip(form_val, 0.0, 1.0))
            except (ValueError, TypeError):
                hist_val, form_val = global_avg, global_avg

            features[f'{side}_smoothed_winrate'] = hist_val
            features[f'{side}_recent_form'] = form_val
            features[f'{side}_is_first_game'] = int(hist_key not in match_data or match_data[hist_key] is None)

        # 4. Standardizing String Categories
        raw_category = str(match_data.get('game_category', 'Blitz')).strip().capitalize()
        valid_categories = ['Bullet', 'Classical', 'Rapid', 'Untimed']
        
        for cat in valid_categories:
            features[f'game_category_{cat}'] = int(raw_category == cat)
            
        # 5. Pipeline Alignment & Matrix Shaping
        df = pd.DataFrame([features])
        for col in self.expected_features:
            if col not in df.columns:
                df[col] = 0
        df = df[self.expected_features]
        
        probs = self.model.predict_proba(df)[0]
        
        return {
            "White_Win_Prob": round(probs[0] * 100, 2),
            "Black_Win_Prob": round(probs[1] * 100, 2),
            "Draw_Prob": round(probs[2] * 100, 2),
            "Metadata": {
                "Sanitized_White_Elo": w_elo,
                "Sanitized_Black_Elo": b_elo,
                "Applied_Category": raw_category if raw_category in valid_categories else "Blitz (Fallback)"
            }
        }

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Processes batches safely by turning columns into validated dictionary records."""
        records = df.to_dict(orient='records')
        predictions = [self.predict_match(rec) for rec in records]
        
        # Flatten probabilities out of output structure
        flat_preds = [{k: v for k, v in p.items() if k != 'Metadata'} for p in predictions]
        return pd.concat([df.reset_index(drop=True), pd.DataFrame(flat_preds)], axis=1)

# --- Operational Safety & Edge Case Evaluation ---
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.abspath(os.path.join(script_dir, "..", "..", "saved_models", "xgb_checkpoint3_5.joblib"))
    
    predictor = ChessPredictor(model_path)
    
    print("\n--- Running Defensive Validation Evaluation ---")
    
    # Intentionally malformed production payload simulating dirty real-world input
    dirty_payload = {
        'white_elo': '1850?',       # Scenario 1: Lichess provisional string representation
        'black_elo': -500,          # Scenario 2: Corrupted physical database integer
        'base_time': None,          # Scenario 3: Missing schema value
        'increment': 5.0,
        'game_category': '  bullet ' # Scenario 4: Messy whitespace capitalization anomaly
    }
    
    output = predictor.predict_match(dirty_payload)
    print(f"Sanitized Engine Resolution: {output['Metadata']}")
    print(f"Model Output Trajectory -> White: {output['White_Win_Prob']}%, Black: {output['Black_Win_Prob']}%, Draw: {output['Draw_Prob']}%")

    # Testing own data:

## For single real match:
# Edit the 'single_match' dictionary in the execution block with at least - 
# (white_elo, black_elo, base_time, increment, game_category)


## Testing a Custom Dataset/DataFrame:
# load new new batch of games using Pandas and pass it directly to the 'predict_dataframe' method

# Load your raw real-world data
#my_new_data = pd.read_csv('new_tournament_games.csv')

# Ensure the column names match what the script expects (rename them if needed)
# my_new_data = my_new_data.rename(columns={'White Rating': 'white_elo', ...})

# Generate predictions
#predictions = predictor.predict_dataframe(my_new_data)

# Save the results
#predictions.to_csv('tournament_predictions.csv', index=False)