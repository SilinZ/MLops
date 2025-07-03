"""
Run the full ML-Ops HW1 pipeline *on top of LakeFS*.

Assumes:
  • the five helper scripts (clean.py, split.py, train.py, eda.py, dp_train.py)
    sit next to this file;
  • env-vars LAKEFS_* already exported.

High-level flow
---------------
main ➜ add raw ➜ tag v1-raw
▶ branch exp-clean → run clean & split → commit → tag v2-clean
▶ branch exp-models – train v1 & v2 → commit
▶ branch exp-dp     – run dp_train   → commit
"""
import os, subprocess, pathlib, shutil, json, tempfile, sys, time
from lakefs_client import LakeFSClient
from lakefs_client.models import Commit, TagRef, BranchCreation

LK_ENDPOINT = os.getenv("LAKEFS_ENDPOINT")
LK_ACCESS   = os.getenv("LAKEFS_ACCESS_KEY")
LK_SECRET   = os.getenv("LAKEFS_SECRET_KEY")
REPO        = "mlops"
LOCAL_TMP   = pathlib.Path(tempfile.mkdtemp())          # workspace dir
SCRIPTS_DIR = pathlib.Path(__file__).parent

cli = LakeFSClient.configure_with_access_key(LK_ENDPOINT, LK_ACCESS, LK_SECRET)
api = cli.repositories

# ------------------------------------------------------------------ helpers
def sh(cmd, **kw):
    print(f"💻  $ {cmd}")
    subprocess.check_call(cmd, shell=True, **kw)

def lake_cp(local_path, lakefs_uri):
    """Upload (or download) via lakectl – keeps sample short & easy."""
    sh(f"lakectl fs upload {lakefs_uri} {local_path}")

def commit(branch, msg):
    res = cli.commits.commit(REPO, branch, Commit(message=msg))
    print(f"✅ committed on {branch[:12]}… {res.id[:8]}")
    return res.id

# ------------------------------------------------------------------ STEP 1 – raw ingest
RAW_CSV = "athletes.csv"
if not cli.objects.stat_object(repositories=REPO, ref="main",
                               path=f"data/raw/{RAW_CSV}", _preload_content=False):
    lake_cp(RAW_CSV, f"lakefs://{REPO}/main/data/raw/{RAW_CSV}")
    raw_commit = commit("main", "Add raw athletes.csv")
    cli.refs.create_tag(REPO, "v1-raw", TagRef(id=raw_commit))

# ------------------------------------------------------------------ STEP 2 – clean & split
if "exp-clean" not in [b.id for b in cli.branches.list_branches(REPO).results]:
    cli.branches.create_branch(
        REPO, BranchCreation(name="exp-clean", source="main"))
WORK = LOCAL_TMP / "workspace"
WORK.mkdir(exist_ok=True)

# download raw → local
sh(f"lakectl fs download lakefs://{REPO}/exp-clean/data/raw/{RAW_CSV} {WORK/RAW_CSV}")

# run clean.py + split.py
sh(f"python {SCRIPTS_DIR/'clean.py'} {WORK/RAW_CSV} {WORK/'athletes_clean.csv'}")
sh(f"python {SCRIPTS_DIR/'split.py'} {WORK/RAW_CSV}  {WORK/'train_v1.csv'} {WORK/'test_v1.csv'}")
sh(f"python {SCRIPTS_DIR/'split.py'} {WORK/'athletes_clean.csv'} "
   f"{WORK/'train_v2.csv'} {WORK/'test_v2.csv'}")

# upload结果
for file in WORK.glob("*.csv"):
    dst = f"lakefs://{REPO}/exp-clean/data/{file.name}"
    lake_cp(file, dst)

clean_commit = commit("exp-clean", "v2 cleaned & splits")
cli.refs.create_tag(REPO, "v2-clean", TagRef(id=clean_commit))

# ------------------------------------------------------------------ STEP 3 – modelling
cli.branches.create_branch(REPO, BranchCreation(
    name="exp-models", source="exp-clean"))

# EDA & baseline v1
sh(f"lakectl fs download lakefs://{REPO}/exp-models/data/train_v1.csv {WORK/'train_v1.csv'}")
sh(f"lakectl fs download lakefs://{REPO}/exp-models/data/test_v1.csv  {WORK/'test_v1.csv'}")
sh(f"python {SCRIPTS_DIR/'eda.py'}   {WORK/'train_v1.csv'} v1")
sh(f"python {SCRIPTS_DIR/'train.py'} {WORK/'train_v1.csv'} {WORK/'test_v1.csv'} "
   f"{WORK/'rf_v1.pkl'} {WORK/'metrics_v1.json'}")

# baseline v2
sh(f"lakectl fs download lakefs://{REPO}/exp-models/data/train_v2.csv {WORK/'train_v2.csv'}")
sh(f"lakectl fs download lakefs://{REPO}/exp-models/data/test_v2.csv  {WORK/'test_v2.csv'}")
sh(f"python {SCRIPTS_DIR/'eda.py'}   {WORK/'train_v2.csv'} v2")
sh(f"python {SCRIPTS_DIR/'train.py'} {WORK/'train_v2.csv'} {WORK/'test_v2.csv'} "
   f"{WORK/'rf_v2.pkl'} {WORK/'metrics_v2.json'}")

# upload models + metrics
for file in WORK.glob("*.pkl"): lake_cp(file, f"lakefs://{REPO}/exp-models/models/{file.name}")
for file in WORK.glob("metrics_*.json"): lake_cp(file, f"lakefs://{REPO}/exp-models/metrics/{file.name}")
model_commit = commit("exp-models", "baseline models v1+v2")

# ------------------------------------------------------------------ STEP 4 – DP model with Opacus
cli.branches.create_branch(REPO, BranchCreation(
    name="exp-dp", source="exp-models"))

sh(f"python {SCRIPTS_DIR/'dp_train.py'} "
   f"{WORK/'train_v2.csv'} {WORK/'test_v2.csv'} "
   f"{WORK/'dp_metrics.json'} {WORK/'epsilon.txt'}")

lake_cp(WORK/'dp_metrics.json', f"lakefs://{REPO}/exp-dp/metrics/dp_metrics.json")
lake_cp(WORK/'epsilon.txt',     f"lakefs://{REPO}/exp-dp/metrics/epsilon.txt")
dp_commit = commit("exp-dp", "DP model metrics")

print("\n🎉  Pipeline finished!  Check your commits & tags on the LakeFS UI.")