from data_manager import DataManager
from ml_engine import MLEngine
import pandas as pd
import traceback
import numpy as np

try:
    print("Testing DataManager...")
    manager = DataManager()
    df = manager.load()
    print(f"Data Loaded: {len(df)} rows")

    print("Testing MLEngine initialization...")
    engine = MLEngine()
    texts = df['text'].values
    print(f"Texts extracted: {len(texts)}")

    print("Testing Vectorization...")
    X = engine.vectorize(texts)
    print(f"Vectorized shape: {X.shape}")

    print("Testing Clustering...")
    clusters = engine.cluster(X)
    print(f"Clusters generated: {len(clusters)}")

    print("Testing Similarity...")
    sim_matrix = engine.similarity(X)
    print(f"Similarity matrix shape: {sim_matrix.shape}")

    print("Testing Feature 1: Name Clusters...")
    names = engine.name_clusters(X, texts, clusters)
    print(f"Cluster names: {names}")

    print("Testing Feature 2: Priority Level...")
    score = engine.priority_score(sim_matrix, 0)
    level = engine.priority_level(score)
    print(f"Sample Priority: {score} -> {level}")

    print("Testing Feature 3: Tags...")
    tags = engine.generate_tags(texts[0])
    print(f"Sample Tags: {tags}")

    print("Testing Feature 4: Summary...")
    df['cluster'] = clusters
    df['cluster_name'] = df['cluster'].map(names)
    summary = engine.generate_summary(df)
    print(f"Summary len: {len(summary)}")

    print("Testing Feature 5: Similar Feedback...")
    similar = engine.find_similar(sim_matrix, texts, 0)
    print(f"Similar items: {len(similar)}")

    print("All tests passed successfully.")

except Exception:
    traceback.print_exc()
