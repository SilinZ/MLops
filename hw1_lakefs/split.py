# split.py
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

def split(input_csv, train_csv, test_csv, test_size=0.2, random_state=42):
    df = pd.read_csv(input_csv)
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    train.to_csv(train_csv, index=False)
    test.to_csv(test_csv, index=False)
    print(f"split → train:{train.shape}, test:{test.shape}")

if __name__=='__main__':
    in_csv, train_csv, test_csv = sys.argv[1], sys.argv[2], sys.argv[3]
    split(in_csv, train_csv, test_csv)