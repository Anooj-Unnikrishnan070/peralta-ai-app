import pandas as pd
import streamlit as st

def run_missing_value_check(df):
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_pct
    })

    st.subheader("Missing Value Detection")
    st.dataframe(missing_df)

    return missing_pct
