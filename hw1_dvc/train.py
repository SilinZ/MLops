# train.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib, json, sys

def train_and_eval(train_csv, test_csv, model_path, metrics_path):
    # 1) 读数据
    df_tr = pd.read_csv(train_csv)
    df_te = pd.read_csv(test_csv)

    # 2) 计算 total_lift（若不存在）
    for df in (df_tr, df_te):
        if 'total_lift' not in df.columns:
            df['total_lift'] = df['deadlift'] + df['candj'] + df['snatch'] + df['backsq']

    # 3) 只保留数值列，并剔除标签
    num_cols = df_tr.select_dtypes(include=[np.number]).columns.tolist()
    num_cols.remove('total_lift')

    # 4) 丢弃 NaN
    df_tr = df_tr.dropna(subset=num_cols + ['total_lift'])
    df_te = df_te.dropna(subset=num_cols + ['total_lift'])

    # 5) 拆分特征/标签
    X_tr, y_tr = df_tr[num_cols], df_tr['total_lift']
    X_te, y_te = df_te[num_cols], df_te['total_lift']

    # 6) 训练
    model = RandomForestRegressor(random_state=42)
    model.fit(X_tr, y_tr)

    # 7) 预测 & 指标
    preds = model.predict(X_te)
    mae = mean_absolute_error(y_te, preds)
    mse = mean_squared_error(y_te, preds)
    rmse = np.sqrt(mse)
    r2  = r2_score(y_te, preds)
    metrics = {'MAE': mae, 'RMSE': rmse, 'R2': r2}

    # 8) 保存
    joblib.dump(model, model_path)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print("✅ Metrics:", metrics)

if __name__=='__main__':
    train_and_eval(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])