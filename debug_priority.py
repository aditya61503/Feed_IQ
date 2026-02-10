from data_manager_fixed import DataManager
from ml_engine import MLEngine
import pandas as pd
import numpy as np

try:
    print("Initializing...")
    manager = DataManager()
    df = manager.load()
    engine = MLEngine()
    texts = df['text'].values
    
    print(f"Data Loaded: {len(df)} rows")
    
    print("Vectorizing...")
    X = engine.vectorize(texts)
    sim_matrix = engine.similarity(X)
    
    print("Calculating Scores...")
    scores = [engine.priority_score(sim_matrix, i) for i in range(len(texts))]
    df['priority_score'] = scores
    df['priority_level'] = df['priority_score'].apply(engine.priority_level)
    
    with open("debug_priority_out.txt", "w", encoding="utf-8") as f:
        f.write("--- SCORES ANALYSIS ---\n")
        f.write(df[['text', 'priority_score', 'priority_level']].sort_values(by='priority_score', ascending=False).to_string())
        
        f.write("\n\n--- DISTRIBUTION ---\n")
        f.write(df['priority_level'].value_counts().to_string())
        
        f.write(f"\n\nMax Score: {max(scores)}\n")
        f.write(f"Min Score: {min(scores)}\n")
        f.write(f"Mean Score: {np.mean(scores)}\n")
    
    print("Output written to debug_priority_out.txt")

except Exception as e:
    print(f"Error: {e}")
