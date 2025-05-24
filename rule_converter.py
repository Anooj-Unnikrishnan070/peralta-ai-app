import sqlite3
import pandas as pd
import subprocess
import os
# … your existing imports …
import subprocess
import os
from typing import List, Dict

# bring in your DeepSeek client
from deepseek_client import deepseek_client

import re
from typing import List, Dict
from deepseek_client import deepseek_client  # adjust this import path as needed

def detect_pii_with_deepseek(column_name: str, samples: List[str]) -> Dict[str, str]:
    """
    Ask DeepSeek whether a given column holds PII, and if so what type.
    Returns a dict:
      {
        "is_pii": bool,
        "pii_type": str or None
      }
    """
    prompt = f"""
You are a data governance assistant.
Column name: {column_name}
Sample values: {samples}

Question:
1) Is this column storing personally identifiable information (PII)?
2) If yes, what type of PII is it? (e.g. Email, Phone, SSN, Passport Number, IP Address, Date of Birth, etc.)

Answer in JSON exactly like:
{{
  "is_pii": true or false,
  "pii_type": string or null
}}
"""

    try:
        result = deepseek_client.query(prompt).json()
    except Exception:
        result = {"is_pii": False, "pii_type": None}

    # --- FALLBACK HEURISTICS IF LLM SAYS NO ---
    if not result.get("is_pii", False):
        name = column_name.lower()
        # 1) Name‐based heuristics
        heuristics = {
            "Email":         r"\bemail\b",
            "Phone":         r"\b(phone|mobile|tel)\b",
            "SSN":           r"\bssn\b",
            "Date of Birth": r"\b(dob|birth)\b",
            "Address":       r"\b(address|street|city|zip|postalcode)\b",
            "Name":          r"\b(name|first|last)\b",
        }
        for pii_type, pattern in heuristics.items():
            if re.search(pattern, name):
                return {"is_pii": True, "pii_type": pii_type}

        # 2) Value‐based email regex
        email_re = re.compile(r"\S+@\S+\.\S+")
        if any(email_re.match(val) for val in samples):
            return {"is_pii": True, "pii_type": "Email"}

    return result


def extract_schema_from_sqlite(db_path: str) -> str:
    """
    Extracts and formats schema from an SQLite database for LLM prompt use.
    """
    if not os.path.exists(db_path):
        return ""

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]

    schema_lines = []
    for table in tables:
        df_schema = pd.read_sql_query(f"PRAGMA table_info({table})", conn)
        cols = ", ".join(df_schema['name'])
        schema_lines.append(f"{table}({cols})")

    conn.close()
    return "\n".join(schema_lines)

def convert_rule_with_deepseek(rule: str, db_path: str) -> str:
    """
    Uses DeepSeek via Ollama to convert a business rule into an enhanced SQL rule
    that returns violating rows, with schema awareness.
    """
    schema_str = extract_schema_from_sqlite(db_path)

    prompt = f"""
You are an AI assistant for data quality rule enforcement.

Your job is to convert business rules into SQL SELECT queries that return only the rows
that violate the rule. Always consider the database schema while doing this.

- Use JOINs if a foreign key exists in multiple tables.
- If checking for missing values, include both IS NULL and TRIM(column) = ''.
- Do not use ALTER TABLE or markdown.
- Output raw SQL only.

Database Schema:
{schema_str}

Business Rule: "{rule}"

Write a SQL SELECT query that:
- Returns rows from the primary table where the rule is violated
- Uses JOINs if needed (e.g., when checking if CustomerId exists in Customer table)
- Avoids assumptions like column lengths
- Works with standard SQL (e.g., SQLite-compatible)Business Rule: "{rule}"
- Do not use table aliases unless there are joins.
- Use full table and column names for clarity.
- Avoid markdown formatting.


SQL Query:
"""

    try:
        return deepseek_client.query(prompt)
    except Exception as e:
        return f"❌ Error during DeepSeek conversion: {str(e)}"


import pandas as pd
from rapidfuzz import process, fuzz

import subprocess
import pandas as pd
from rapidfuzz import process, fuzz

def recommend_term_definition_with_deepseek(term: str,
                                           glossary_path: str = "Business Terms.xlsx"
                                          ) -> str:
    # 1. Load your pre-built glossary
    df = pd.read_excel(glossary_path)
    terms       = df["Term"].astype(str).tolist()
    definitions = df["Definition"].astype(str).tolist()

    # 2. Find top 5 closest matches
    matches = process.extract(term, terms, scorer=fuzz.token_sort_ratio, limit=5)

    # 3. Build prompt context from those matches
    context_lines = []
    for match_term, score, idx in matches:
        context_lines.append(f"{match_term}: {definitions[idx]}")
    context = "\n".join(context_lines)

    prompt = f"""
You are a business-glossary expert.  Based on these existing definitions:
{context}

Suggest a concise, precise definition for the term: '{term}'
"""

    # 4. Call DeepSeek via DeepSeekClient
    try:
        return deepseek_client.query(prompt)
    except Exception as e:
        return f"❌ Error during DeepSeek conversion: {str(e)}"
