import pandas as pd

# Load your local parquet file
df = pd.read_parquet("BigEarthNet-VQA.parquet")

print(f"Total dataset records available: {len(df)}")

# Sample 5 different unique questions from the dataset
sample_test_set = df.sample(n=5, random_state=42)

for idx, row in sample_test_set.iterrows():
    print(f"--- Sample Record {idx} ---")
    print(f"Question: {row['input']}")
    print(f"Ground Truth Answer: {row['output']}")
    print(f"Category: {row.get('category', 'N/A')}")
    print(f"Patch ID: {row['patch_id']}\n")