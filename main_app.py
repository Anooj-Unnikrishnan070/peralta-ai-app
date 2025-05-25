import streamlit as st
import sqlite3
import pandas as pd

# Data Quality modules
from missing_checker import run_missing_value_check
from rapidfuzz import fuzz
from itertools import combinations
from anomaly_checker import detect_anomalies
from rule_converter import detect_pii_with_deepseek

# Rule Converter modules
from lineage_helper import (
    extract_lineage_graph,
    extract_column_level_lineage,
    collapse_to_business_lineage,
    get_graphviz_layout_with_ranksep,
)
from rule_converter import convert_rule_with_deepseek, recommend_term_definition_with_deepseek
from semantic_search import build_schema_index, search_schema
import networkx as nx
import matplotlib.pyplot as plt
  
# ----------------------------------------------------------------------------
# Sidebar - File upload and navigation
st.set_page_config(layout="wide")
st.sidebar.markdown(
    '<h1 style="font-size:3.775rem; margin:0;">Peralta AI</h1>',
    unsafe_allow_html=True
)
st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader("Upload SQLite DB", type=["sqlite", "db"])
mode = st.sidebar.radio("Select Tool:", ["Data Quality Dashboard", "Lineage Generation and Semantic Search"])

if not uploaded_file:
    st.sidebar.warning("Please upload a SQLite database first.")
    st.stop()

# Save uploaded db and connect
temp_path = "temp_db.sqlite"
with open(temp_path, "wb") as f:
    f.write(uploaded_file.read())
conn = sqlite3.connect(temp_path)

# Common Table List
tables_df = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
table_list = tables_df["name"].tolist()

# =============================================================================
# Data Quality function
def run_data_quality():
    st.title("AI Agent - Data Quality Overview")
    st.subheader("Select Table for Analysis")
    selected_table = st.selectbox("Table", table_list)
    df = pd.read_sql_query(f"SELECT * FROM {selected_table}", conn)

    # Raw data
    st.subheader("Raw Table Data")
    st.dataframe(df)

    # Profiling
    st.subheader("Table Profiling")
    st.write(f"Total Rows: {df.shape[0]}")
    st.write(f"Total Columns: {df.shape[1]}")
    st.write("Data Types:")
    st.write(df.dtypes)


    # Analysis selection
    analysis_option = st.selectbox(
        "Choose an analysis",
        ["None", "Missing Value Detection", "Duplicate check using Machine Learning", "Anomaly Detection", "PII Detection", "Business Rule Conversion"],
    )

    if analysis_option == "Missing Value Detection":
        missing_pct = run_missing_value_check(df)
        score = max(0, 100 - missing_pct.mean())
        st.metric("Data Quality Score", f"{score:.1f}/100")

    elif analysis_option == "Duplicate check using Machine Learning":
        st.subheader("Duplicate Detection (Fuzzy-Based)")
        st.markdown("Only non-primary key columns are considered for fuzzy matching.")
        threshold = 0.90
        columns_to_compare = df.columns[1:]
        st.write(f"Columns used for comparison: {list(columns_to_compare)}")
        st.write(f"Threshold for average similarity: {threshold}")

        duplicates_found = False
        timestamp_cols = [col for col in columns_to_compare if 'date' in col.lower() or 'time' in col.lower()]
        foreign_key_cols = [col for col in columns_to_compare if col.lower().endswith("id") and col != df.columns[0]]
        compare_cols = [col for col in columns_to_compare if col not in timestamp_cols + foreign_key_cols]

        for i, j in combinations(range(len(df)), 2):
            row1, row2 = df.iloc[i], df.iloc[j]
            if any(str(row1[col]) != str(row2[col]) for col in timestamp_cols + foreign_key_cols if col in row1 and col in row2):
                continue
            scores = []
            for col in compare_cols:
                val1 = str(row1.get(col, ""))
                val2 = str(row2.get(col, ""))
                scores.append(fuzz.token_sort_ratio(val1, val2) / 100)
            avg_score = sum(scores) / len(scores) if scores else 0
            if avg_score >= threshold:
                st.warning(f"🔁 Duplicate Detected: Row {i+1} vs Row {j+1}")
                for col in compare_cols:
                    val1, val2 = str(row1.get(col, "")), str(row2.get(col, ""))
                    score = fuzz.token_sort_ratio(val1, val2)
                    status = (
                        "✅ Exact" if score == 100
                        else "🟠 Partial" if score >= threshold * 100
                        else "❌ Different"
                    )
                    cols = st.columns(4)
                    with cols[0]: st.markdown(f"**{col}**")
                    with cols[1]: st.write(val1)
                    with cols[2]: st.write(val2)
                    with cols[3]: st.markdown(f"**{status}**")
                st.divider()
                duplicates_found = True
        if not duplicates_found:
            st.success("No potential duplicates found.")

    elif analysis_option == "Anomaly Detection":
        st.subheader("Anomaly Detection")
        anomalies = detect_anomalies(df)
        if anomalies.empty:
            st.success("✅ No anomalies detected in numeric columns.")
        else:
            st.warning("Anomalies Detected")
            st.dataframe(anomalies)

    elif analysis_option == "PII Detection":
        st.subheader("Column Details with Tagging")

        # build a list of dicts for each column
        records = []
        for col in df.columns:
            # 1) grab dtype
            dtype = str(df[col].dtype)
            # 2) sample for the LLM
            samples = df[col].dropna().astype(str).head(10).tolist()
            pii_info = detect_pii_with_deepseek(col, samples)
            is_pii = pii_info.get("is_pii", False)

            records.append({
                "Column": col,
                "Data Type": dtype,
                "PII": "Yes" if is_pii else "No"
            })

        # 3) render as a table
        tag_df = pd.DataFrame(records)
        st.table(tag_df)

    elif analysis_option == "Business Rule Conversion":
        st.markdown("---\n### Enter your Business Rule")
        rule = st.text_area("Describe your business rule in natural language:", height=150)
        output_format = st.selectbox("Output Format", ["SQL", "Python", "Regex"])
        
        if st.button("Convert to Technical Rule"):
            if not rule.strip():
                st.warning("Please enter a business rule.")
            else:
                with st.spinner("Generating technical rule using DeepSeek..."):
                    output = convert_rule_with_deepseek(rule=rule, db_path=temp_path)
                st.success(f"Generated {output_format} Rule:")
                st.code(output, language=output_format.lower())


# =============================================================================
# Rule Converter function
def run_rule_converter():
    st.title("AI Agent - Business Rule Converter")

    # Build human-readable schema string
    all_schemas = []
    for table in table_list:
        df_schema = pd.read_sql_query(f"PRAGMA table_info({table})", conn)
        cols = ", ".join(df_schema["name"])
        all_schemas.append(f"{table}({cols})")
    human_schema = "Database Schema:\n" + "\n".join(all_schemas)

    # Build semantic index (ignore its schema_str)
    schema_texts, index, _ = build_schema_index(temp_path)

    # Display grouped schema
    st.markdown("### Database Schema")
    st.code(human_schema, language="")

    # Lineage visualization
    st.markdown("---\n### Lineage Visualization")
    lineage_option = st.selectbox("Select Lineage Type", ["None", "Technical Lineage", "Business Lineage"])
    if lineage_option != "None":
        with st.spinner("Generating lineage graph..."):
            if lineage_option == "Technical Lineage":
                G = extract_lineage_graph(temp_path)
            else:
                col_graph = extract_column_level_lineage(temp_path)
                G = collapse_to_business_lineage(col_graph)
            try:
                pos = get_graphviz_layout_with_ranksep(G, ranksep=2.0)
            except Exception as e:
                st.warning(f"Could not use pydot layout, using spring layout. Error: {e}")
                pos = nx.spring_layout(G, seed=42)
            fig, ax = plt.subplots(figsize=(12, 8))
            nx.draw(
                G,
                pos,
                with_labels=True,
                arrows=True,
                node_size=2000,
                node_color="#ADE1AF" if lineage_option == "Business Lineage" else "#D6EAF8",
                font_size=9,
                ax=ax,
            )
            if lineage_option == "Technical Lineage":
                edge_labels = nx.get_edge_attributes(G, 'label')
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, ax=ax)
            st.pyplot(fig)

    # Semantic search on schema
    st.markdown("---\n### Semantic Search on Database Schema")
    search_query = st.text_input("Enter your search (e.g., 'customer name'):")
    if search_query:
        if index:
            with st.spinner("Searching..."):
                results = search_schema(search_query, schema_texts, index)
            st.markdown("#### Top Matching Schema Entries:")
            for res in results:
                st.write(f"**Match {res['rank']}**: {res['entry']} ({res['level']} - confidence {res['confidence']})")
        else:
            st.warning("Please upload a valid SQLite file first to enable semantic search.")

    # Business term recommendation
    st.markdown("---\n### 💡 Recommend a Business-Glossary Definition")
    term = st.text_input("Enter a business term to define (e.g., 'customer'):")
    if st.button("Get Recommended Definition"):
        if not term.strip():
            st.warning("Please enter a business term first.")
        else:
            with st.spinner("Generating definition..."):
                definition = recommend_term_definition_with_deepseek(term=term)
            st.success("Here’s your recommended definition:")
            st.write(definition)

# =============================================================================
# Main dispatch
if mode == "Data Quality Dashboard":
    run_data_quality()
else:
    run_rule_converter()
