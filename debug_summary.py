from data_manager import DataManager
from ml_engine import MLEngine
import pandas as pd
import numpy as np

try:
    print("Initializing...")
    manager = DataManager()
    df = manager.load()
    engine = MLEngine()
    texts = df['text'].values
    
    print("Vectorizing & Clustering...")
    X = engine.vectorize(texts)
    clusters = engine.cluster(X)
    
    print("Naming clusters...")
    names = engine.name_clusters(X, texts, clusters)
    
    with open("debug_output.txt", "w") as f:
        f.write(f"Clusters type: {clusters.dtype}\n")
        f.write(f"Clusters unique: {np.unique(clusters)}\n")
        
        if names:
            f.write(f"Names keys type: {type(list(names.keys())[0])}\n")
        else:
            f.write("Names dict is empty!\n")
        f.write(f"Names: {names}\n")
        
        df['cluster'] = clusters
        f.write(f"df['cluster'] dtype: {df['cluster'].dtype}\n")
        
        df['cluster_name'] = df['cluster'].map(names)
        f.write(f"df['cluster_name'] unique: {df['cluster_name'].unique()}\n")
        f.write(f"df['cluster_name'] nulls: {df['cluster_name'].isnull().sum()}\n")
        
        summary = engine.generate_summary(df)
        f.write("--- SUMMARY START ---\n")
        f.write(summary)
        f.write("\n--- SUMMARY END ---\n")
        
    print("Debug output written to debug_output.txt")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
