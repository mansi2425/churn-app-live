"""
db_setup.py
-----------
Simulates an enterprise data ingestion pipeline.
Reads the Telco Customer Churn CSV and loads it into a
local SQLite database (churn_data.db).

Usage:
    python src/db_setup.py
"""

import sqlite3
import pandas as pd
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH   = os.path.join(BASE_DIR, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
DB_PATH    = os.path.join(BASE_DIR, "churn_data.db")

# ── Step 1: Load raw CSV ───────────────────────────────────────────────────────
def load_csv(path: str) -> pd.DataFrame:
    print(f"[INFO] Reading CSV from: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] Raw dataset shape: {df.shape}")
    return df

# ── Step 2: Basic cleaning before storage ─────────────────────────────────────
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # TotalCharges is sometimes read as object due to whitespace strings
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop duplicate customerIDs if any
    before = len(df)
    df = df.drop_duplicates(subset=["customerID"])
    after  = len(df)
    if before != after:
        print(f"[WARN] Removed {before - after} duplicate rows.")

    print(f"[INFO] Cleaned dataset shape: {df.shape}")
    return df

# ── Step 3: Write to SQLite ────────────────────────────────────────────────────
def write_to_sqlite(df: pd.DataFrame, db_path: str, table_name: str = "customers") -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Drop table if it already exists (idempotent re-runs)
    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.commit()

    # Write dataframe
    df.to_sql(table_name, conn, index=False, if_exists="replace")
    conn.commit()

    # Verify
    count = cursor.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    print(f"[INFO] Successfully written {count} records to table '{table_name}' in {db_path}")

    conn.close()

# ── Step 4: Create an index for faster querying ────────────────────────────────
def create_index(db_path: str, table_name: str = "customers") -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_customerID ON {table_name}(customerID)"
    )
    conn.commit()
    conn.close()
    print("[INFO] Index created on customerID.")

# ── Step 5: Print a quick schema summary ──────────────────────────────────────
def print_schema(db_path: str, table_name: str = "customers") -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    cols = cursor.fetchall()
    print(f"\n[SCHEMA] Table '{table_name}' columns:")
    for col in cols:
        print(f"    {col[1]:30s}  {col[2]}")
    conn.close()

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"CSV not found at {CSV_PATH}.\n"
            "Please download 'WA_Fn-UseC_-Telco-Customer-Churn.csv' from Kaggle "
            "and place it inside the /data folder."
        )

    df = load_csv(CSV_PATH)
    df = clean_dataframe(df)
    write_to_sqlite(df, DB_PATH)
    create_index(DB_PATH)
    print_schema(DB_PATH)

    print("\n[SUCCESS] Database setup complete. File saved at:", DB_PATH)