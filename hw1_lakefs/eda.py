# eda.py
import pandas as pd
import matplotlib.pyplot as plt
import sys

def eda(input_csv, prefix):
    df = pd.read_csv(input_csv)
    print(f"--- EDA {prefix} ---")
    print(df.describe(include='all').T)
    # 直方图
    df.hist(bins=30, figsize=(12,10))
    plt.tight_layout()
    plt.savefig(f"{prefix}_hist.png")
    print(f"hist saved → {prefix}_hist.png")

if __name__=='__main__':
    in_csv, prefix = sys.argv[1], sys.argv[2]
    eda(in_csv, prefix)