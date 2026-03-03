"""
preprocess.py - Cleans the raw Netflix dataset for the recommendation system.

What this script does:
1. Reads the raw netflix_titles.csv file
2. Fills in missing values (some shows have blank descriptions)
3. Combines relevant text fields into one "tags" column
4. Saves the cleaned data for the training script to use
"""

import pandas as pd
import yaml
import os
import sys


def load_params():
    """Read configuration parameters from params.yaml."""
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params


def preprocess_data():
    """
    Main preprocessing function.
    
    Steps:
    1. Load raw data
    2. Fill missing values
    3. Create combined 'tags' column
    4. Filter out very short entries
    5. Save cleaned data
    """
    params = load_params()
    preprocess_params = params["preprocess"]

    # Step 1: Load raw CSV
    raw_data_path = os.path.join("data", "netflix_titles.csv")

    if not os.path.exists(raw_data_path):
        print(f"ERROR: Could not find {raw_data_path}")
        print("Make sure you have run: dvc pull")
        sys.exit(1)

    df = pd.read_csv(raw_data_path)
    print(f"Loaded {len(df)} rows from {raw_data_path}")
    print(f"Columns: {list(df.columns)}")

    # Step 2: Fill missing values
    fill_value = preprocess_params["fill_missing_with"]

    df["description"] = df["description"].fillna(fill_value)
    df["listed_in"] = df["listed_in"].fillna("")
    df["director"] = df["director"].fillna("")
    df["cast"] = df["cast"].fillna("")
    df["country"] = df["country"].fillna("")

    # Step 3: Create combined 'tags' column
    df["tags"] = (
        df["listed_in"]
        + " " + df["description"]
        + " " + df["director"]
        + " " + df["cast"]
        + " " + df["country"]
    )
    df["tags"] = df["tags"].str.lower()

    # Step 4: Filter short descriptions
    min_len = preprocess_params["min_description_length"]
    before_filter = len(df)
    df = df[df["description"].str.len() >= min_len]
    after_filter = len(df)
    print(f"Filtered: {before_filter} -> {after_filter} rows "
          f"(removed {before_filter - after_filter} short entries)")

    # Step 5: Reset index and save
    df = df.reset_index(drop=True)

    cleaned_data_path = os.path.join("data", "netflix_cleaned.csv")
    df.to_csv(cleaned_data_path, index=False)

    print(f"Saved cleaned data to {cleaned_data_path}")
    print(f"Final dataset: {len(df)} shows/movies with {len(df.columns)} columns")

    return df


if __name__ == "__main__":
    preprocess_data()