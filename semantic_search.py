import sqlite3
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load the embedding model once
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def build_schema_index(sqlite_path):
    """
    Reads a SQLite file and builds FAISS index on 'table → column' format.
    Returns: schema_texts (raw), FAISS index, display_string (for UI)
    """
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()

    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]

    schema_texts = []     # For embedding & searching
    schema_display = []   # For UI display

    for table in tables:
        df_schema = pd.read_sql_query(f"PRAGMA table_info({table})", conn)
        for col in df_schema['name']:
            schema_texts.append(f"{table} → {col}")
            schema_display.append(f"{table}({col})")

    # Create embeddings
    schema_embeddings = embedding_model.encode(schema_texts)
    dimension = schema_embeddings.shape[1]
    # index = faiss.IndexFlatL2(dimension)
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(schema_embeddings)
    index.add(np.array(schema_embeddings))

    # Create full schema string for UI (optional)
    schema_str = "Database Schema:\n" + "\n".join(schema_display)

    return schema_texts, index, schema_str


def search_schema(query, schema_texts, index, top_k=5):
    """
    Given a search query, returns top_k matching schema entries with scores.
    """
    query_embedding = embedding_model.encode([query])
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(np.array(query_embedding), top_k)

    results = []
    for rank, (i, dist) in enumerate(zip(indices[0], distances[0]), 1):
        entry = schema_texts[i]
        confidence = 1 - dist
        confidence_level = "High" if confidence > 0.85 else "Medium" if confidence > 0.6 else "Low"
        results.append({
            "rank": rank,
            "entry": entry,
            "confidence": round(confidence, 2),
            "level": confidence_level
        })

    return results
