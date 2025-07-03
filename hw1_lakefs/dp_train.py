# dp_train.py  ---------------------------------------------------------------
"""
Differential-Private training with PyTorch + Opacus.

CLI usage
---------
python dp_train.py \
    lakefs://mlops-athletes/v2_clean/clean/train.csv \
    lakefs://mlops-athletes/v2_clean/clean/test.csv  \
    metrics/dp_metrics_v2.json metrics/epsilon_v2.txt
"""
import sys, json, math, pathlib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from opacus import PrivacyEngine

# --------------------------------------------------------------------------- #
# optional: read from lakeFS                                                   #
try:
    from lakefs_client import LakeFSClient
    from lakefs_client.models import ObjectStats
except ImportError:
    LakeFSClient = None  # if user runs without lakeFS

def read_csv(path: str) -> pd.DataFrame:
    """Load CSV from local disk **or** lakeFS URL."""
    if path.startswith("lakefs://"):
        if LakeFSClient is None:
            raise ImportError("pip install lakefs-client first")
        _, repo, rest = path.split("://", 1)[1].split("/", 2)
        client = LakeFSClient()
        obj: ObjectStats = client.objects_api.get_object(repo, rest, branch="")  # default
        return pd.read_csv(obj.raw)
    else:
        return pd.read_csv(path)

# --------------------------------------------------------------------------- #
def dp_train(train_csv, test_csv, metrics_path, eps_path,
             noise=1.1, clip=1.0, batch=128, epochs=10, lr=0.05, delta=1e-5):
    # 1) ------------------------------------------------------------------- #
    df_tr, df_te = read_csv(train_csv), read_csv(test_csv)
    for df in (df_tr, df_te):
        if "total_lift" not in df:
            df["total_lift"] = df[["deadlift","candj","snatch","backsq"]].sum(axis=1)

    num_cols = df_tr.select_dtypes(np.number).columns.drop("total_lift")
    scaler = StandardScaler().fit(df_tr[num_cols])

    X_tr = torch.tensor(scaler.transform(df_tr[num_cols]), dtype=torch.float32)
    y_tr = torch.tensor(df_tr["total_lift"].values,        dtype=torch.float32).view(-1,1)
    X_te = torch.tensor(scaler.transform(df_te[num_cols]), dtype=torch.float32)
    y_te = torch.tensor(df_te["total_lift"].values,        dtype=torch.float32).view(-1,1)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_te, y_te), batch_size=batch)

    # 2) ------------------------------------------------------------------- #
    model = nn.Sequential(
        nn.Linear(X_tr.shape[1], 64), nn.ReLU(),
        nn.Linear(64, 32),           nn.ReLU(),
        nn.Linear(32,1)
    )
    criterion, optimizer = nn.MSELoss(reduction="mean"), optim.Adam(model.parameters(), lr=lr)

    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module          = model,
        optimizer       = optimizer,
        data_loader     = train_loader,
        target_delta    = delta,
        target_epsilon  = math.inf,    # we don’t fix ε beforehand
        max_grad_norm   = clip,
        noise_multiplier= noise,
    )
    print(f" Model is now DP.  Noise={noise}, Clip={clip}")

    # 3) ------------------------------------------------------------------- #
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    for ep in range(1, epochs+1):
        model.train()
        running_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(xb)
        print(f"Epoch {ep}/{epochs}  |  train-MSE = {running_loss/len(X_tr):.2f}")

    # 4) ------------------------------------------------------------------- #
    model.eval()
    with torch.no_grad():
        preds = torch.cat([model(xb.to(device)) for xb,_ in test_loader]).cpu().numpy()
    mae = mean_absolute_error(y_te.numpy(), preds)
    r2  = r2_score(y_te.numpy(), preds)

    # 5) ------------------------------------------------------------------- #
    eps = privacy_engine.get_epsilon(delta)
    print(f"\n🔎  Test MAE = {mae:.2f} | R² = {r2:.3f}")
    print(f"\n  DP guarantee: ε = {eps:.3f}  (δ = {delta:.2e})")

    out = {"MAE": mae, "R2": r2, "epsilon": eps, "delta": delta,
           "noise_multiplier": noise, "clip": clip, "epochs": epochs}

    pathlib.Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f: json.dump(out, f, indent=2)
    with open(eps_path, "w") as f: f.write(str(eps))

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    dp_train(*sys.argv[1:])