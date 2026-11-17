#!/usr/bin/env python3
"""Build the data files behind the paper's four charts from the archived run results.

Usage: python make_fig_data.py <cgfl.py dir> <results dir> [out dir, default fig_data]

Every value written here is computed from results/*.json; the figures (pgfplots) only read these CSVs.
Archived Top-k runs stored 32-bit indices; their data volume and uplink time are rescaled to
ceil(log2 d)-bit indices, as for the tables (see audit_tables.py).

Output
  curves_{nodp,dp}_<method>.csv  validation AUC vs cumulative uplink time, mean/std over 5 seeds (Fig. "AUC vs uplink")
  totals_{nodp,dp}.csv      total uplink time and test AUC per method, mean/std over 5 seeds (Fig. "AUC vs total uplink")
  eps_T.csv                 test AUC per (eps, T, policy), 3 seeds; total uplink time per (T, policy)
  robust.csv                test AUC and uplink-time reduction per setting, 3 seeds (Fig. "robustness")
"""
import sys, os, json, math, argparse
import numpy as np

CODE, RES = sys.argv[1:3]
OUT = sys.argv[3] if len(sys.argv) > 3 else "fig_data"
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, CODE)
import cgfl  # noqa: E402

BASE = vars(cgfl.defaults().parse_args(["--lr_dp", "0.5"]))
METHODS = ["fedavg", "fixed", "random", "inverse", "logistic", "capacity"]


def load(**o):
    rid, _ = cgfl.run_id(argparse.Namespace(**{**BASE, **o}))
    r = json.load(open(os.path.join(RES, rid + ".json")))
    c = r["cfg"]
    fac = 1.0
    if c.get("mask") == "topk" and c.get("method") not in ("fedavg", "qsgd", "qsgd_cap"):
        fac = (32 + math.ceil(math.log2(r["d"]))) / 64.0
    r["_fac"] = fac
    return r


def run_cfg(m, dp, **o):
    if m == "fedavg":
        return dict(method="fedavg", dp=dp, **o)
    return dict(method=m, dp=dp, mask="topk" if dp == "none" else "public", **o)


def total_uplink(r):
    return r["_fac"] * r["wallclock_uplink_s"]


def write(name, header, rows, fmt="{:.6g}"):
    with open(os.path.join(OUT, name), "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(x if isinstance(x, str) else fmt.format(x) for x in row) + "\n")


# ------------------------------------------------ AUC vs cumulative uplink time, and totals (main stage)
S5 = [1, 2, 3, 4, 5]
for dp, tag in [("none", "nodp"), ("record", "dp")]:
    rows, tot = [], []
    for m in METHODS:
        runs = [load(**run_cfg(m, dp, seed=s)) for s in S5]
        x, y = [], []
        for r in runs:
            cum = np.cumsum([h["lat_s"] for h in r["hist"]]) * r["_fac"]
            ev = [(c, h["val_auc"]) for c, h in zip(cum, r["hist"]) if "val_auc" in h]
            x.append([e[0] for e in ev]); y.append([e[1] for e in ev])
        x, y = np.array(x), np.array(y)
        ts = [h["t"] for h in runs[0]["hist"] if "val_auc" in h]
        for i, t in enumerate(ts):
            rows.append([m, t, x[:, i].mean(), y[:, i].mean(), y[:, i].std(ddof=1)])
        up = np.array([total_uplink(r) for r in runs]); auc = np.array([r["test"]["auc"] for r in runs])
        tot.append([m, up.mean(), up.std(ddof=1), auc.mean(), auc.std(ddof=1)])
    for m in METHODS:   # one file per method (simpler to plot)
        write(f"curves_{tag}_{m}.csv", ["round", "uplink_s", "auc", "auc_std"], [r[1:] for r in rows if r[0] == m])
    write(f"totals_{tag}.csv", ["method", "uplink_s", "uplink_std", "auc", "auc_std"], tot)

# ------------------------------------------------ privacy budget x rounds (eps_T stage)
S3 = [1, 2, 3]
rows = []
for e in [1, 3, 8]:
    for T in [100, 200, 400]:
        for m in ["fixed", "capacity"]:
            runs = [load(**run_cfg(m, "record", eps=e, T=T, seed=s)) for s in S3]
            auc = np.array([r["test"]["auc"] for r in runs]); up = np.array([total_uplink(r) for r in runs])
            rows.append([m, str(e), T, auc.mean(), auc.std(ddof=1), up.mean()])
write("eps_T.csv", ["method", "eps", "T", "auc", "auc_std", "uplink_s"], rows)

# ------------------------------------------------ robustness (scale, heterogeneity, channel), eps = 3
settings = [("default", {}), ("N50", dict(N=50)), ("N100", dict(N=100)),
            ("a01", dict(alpha_dir=0.1)), ("a10", dict(alpha_dir=1.0)),
            ("blockage", dict(scenario="blockage")), ("interf", dict(scenario="interf")),
            ("blockinterf", dict(scenario="blockage_interf"))]
rows = []
for i, (name, o) in enumerate(settings):
    st = {}
    for m in ["fixed", "capacity"]:
        runs = [load(**run_cfg(m, "record", seed=s, **o)) for s in S3]
        st[m] = (np.array([r["test"]["auc"] for r in runs]), np.array([total_uplink(r) for r in runs]))
    red = 100 * (1 - st["capacity"][1].mean() / st["fixed"][1].mean())
    rows.append([name, i, st["fixed"][0].mean(), st["fixed"][0].std(ddof=1),
                 st["capacity"][0].mean(), st["capacity"][0].std(ddof=1), red])
write("robust.csv", ["setting", "x", "auc_fixed", "std_fixed", "auc_cap", "std_cap", "reduction_pct"], rows)
print("wrote", sorted(os.listdir(OUT)), "to", OUT)
