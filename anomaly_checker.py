import pandas as pd

def detect_anomalies(df, threshold=3.0):
    """
    Detect anomalies in numerical columns based on z-score.
    Excludes columns likely to be IDs or timestamps.
    """
    # Select numeric columns excluding IDs or timestamps
    numeric_columns = [
        col for col in df.select_dtypes(include=['number']).columns
        if 'id' not in col.lower() and 'date' not in col.lower() and 'time' not in col.lower()
    ]

    anomalies = []

    for col in numeric_columns:
        z_scores = (df[col] - df[col].mean()) / df[col].std()
        outliers = df[z_scores.abs() > threshold]

        for idx, val in outliers[col].items():
            anomalies.append({
                "RowIndex": idx,
                "Anomalous_Column": col,
                "Anomalous_Value": val
            })

    anomaly_df = pd.DataFrame(anomalies)
    return anomaly_df.drop_duplicates()