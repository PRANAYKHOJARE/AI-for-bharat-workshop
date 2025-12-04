import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    print("Loaded data:", df.shape)
    return df

def clean_data(df):
    df = df.dropna()
    df = df.drop_duplicates()
    print("Cleaned data:", df.shape)
    return df

def save_data(df, path):
    df.to_csv(path, index=False)
    print("Saved cleaned data to:", path)

if __name__ == "__main__":
    raw_path = "data/raw/sample_data.csv"
    processed_path = "data/processed/clean_data.csv"

    df = load_data(raw_path)
    df_clean = clean_data(df)
    save_data(df_clean, processed_path)
