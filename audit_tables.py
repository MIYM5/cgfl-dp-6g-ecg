#!/usr/bin/env python3
"""Consistency audit: recompute every value of the paper's result tables from the raw run files.

Usage: python audit_tables.py <cgfl.py dir> <results dir> <paper dir> <out.csv>
For each table row the expected configuration is listed explicitly; each table cell is parsed from the
LaTeX source and compared with the value recomputed from the JSON results at the printed precision.
"""
import sys, os, re, json, argparse, csv
import numpy as np
from scipy import stats

CODE, RES, PAPER, OUT = sys.argv[1:5]
sys.path.insert(0, CODE)
import cgfl  # noqa: E402

BASE = vars(cgfl.defaults().parse_args(["--lr_dp", "0.5"]))   # every reported run used --lr_dp 0.5
S5, S3 = [1, 2, 3, 4, 5], [1, 2, 3]
_cache = {}
INDEX_FIX = os.environ.get("INDEX_FIX", "1") == "1"


def runs(cfg, seeds):
    out = []
    for s in seeds:
        o = {**cfg, "seed": s}
        rid, _ = cgfl.run_id(argparse.Namespace(**{**BASE, **o}))
        f = os.path.join(RES, f"{rid}.json")
        if not os.path.exists(f):
            raise FileNotFoundError(f"missing run {rid} for {o}")
        if f not in _cache:
            _cache[f] = json.load(open(f))
        out.append(_cache[f])
    return out


def t2t(hist, target):
    cum = 0.0
    for h in hist:
        cum += h["lat_s"]
        if h.get("val_auc", -1) >= target:
            return cum
    return np.nan


def S(cfg, seeds=S5, target=None, ref=None, rare=None):
    """Statistics of one table row (mean over seeds)."""
    R = runs(cfg, seeds)
    # The stored runs count 32-bit indices for Top-k payloads; the paper uses ceil(log2 d)-bit indices
    # (b_idx = 18 for d < 2^18). Index-carrying runs are rescaled accordingly: payload (32+b_idx)/(32+32).
    fac = 1.0
    if INDEX_FIX and cfg.get("mask") == "topk" and cfg.get("method") not in ("fedavg", "qsgd", "qsgd_cap"):
        fac = (32 + int(np.ceil(np.log2(R[0]["d"])))) / 64.0
    a = np.array([r["test"]["auc"] for r in R])
    d = dict(auc=a.mean(), sd=a.std(ddof=1), f1=np.mean([r["test"]["f1"] for r in R]),
             bacc=np.mean([r["test"]["bacc"] for r in R]), mb=fac * np.mean([r["comm_mb"] for r in R]),
             up=fac * np.mean([r["wallclock_uplink_s"] for r in R]), sigma=np.mean([r["sigma_mean"] for r in R]),
             _a=a, _R=R)
    if target is not None:
        tt = [fac * t2t(r["hist"], target) for r in R]
        d["t2t"] = np.nanmean(tt) if not all(np.isnan(tt)) else np.nan
    if ref is not None:
        b = runs(ref, seeds)
        bb = np.array([r["test"]["auc"] for r in b])
        d["p"] = 1.0 if np.allclose(a, bb) else stats.wilcoxon(a, bb).pvalue
    if "site_auc" in R[0]["test"]:
        for k in R[0]["test"]["site_auc"]:
            d["site_" + k] = np.mean([r["test"]["site_auc"][k] for r in R])
    if rare is not None:   # mean AUC over the labels in `rare` (indices), per seed then averaged
        d["rare"] = np.mean([np.mean([r["test"]["label_auc"][str(c)] for c in rare if str(c) in r["test"]["label_auc"]])
                             for r in R])
    return d


def num(x):
    """Parse a LaTeX table cell into (value, decimals, sd, sd_decimals)."""
    x = x.replace("{,}", "").replace("\\textbf{", "").replace("}", "").replace(",", "").replace("$", "")
    x = x.replace("\\%", "").replace("%", "").strip()
    m = re.match(r"^(-?\d+(?:\.\d+)?)(?:\\pm(-?\d+(?:\.\d+)?))?", x)
    if not m:
        return None
    v, sd = m.group(1), m.group(2)
    dec = len(v.split(".")[1]) if "." in v else 0
    out = (float(v), dec, float(sd) if sd else None, (len(sd.split(".")[1]) if sd and "." in sd else 0))
    return out


ROWS = []   # (table, row label, column, paper value, recomputed, status)


def check(table, label, col, cell, val, sdval=None, tol_extra=0.0):
    p = num(cell)
    if p is None:
        ROWS.append((table, label, col, cell, "", "UNPARSED")); return
    v, dec, sd, sdd = p
    tol = 0.5 * 10 ** (-dec) + 1e-9 + tol_extra
    ok = abs(v - val) <= tol
    rec = f"{val:.{dec}f}"
    if sd is not None and sdval is not None:
        ok_sd = abs(sd - sdval) <= 0.5 * 10 ** (-sdd) + 1e-9
        rec += f"±{sdval:.{sdd}f}"
        ok = ok and ok_sd
    ROWS.append((table, label, col, cell.strip(), rec, "OK" if ok else "MISMATCH"))


# table files: <paper dir>/tables/<name> (current layout) or older flat file names
OLD_NAMES = {'main_tabs.tex': 'tables.tex', 'baselines.tex': 'tables_rev2_external.tex', 'tuned.tex': 'tables_rev2_best.tex', 'multilabel.tex': 'tables_rev2_multilabel.tex', 'diag44.tex': 'tables_diag.tex', 'cinc2021.tex': 'tables_rev2_cinc.tex', 'resnet.tex': 'tables_resnet.tex', 'norm_agg.tex': 'tables_normagg.tex', 'wireless.tex': 'tables_wireless.tex'}


def paper_file(fn):
    for c in (os.path.join(PAPER, "tables", fn), os.path.join(PAPER, fn), os.path.join(PAPER, OLD_NAMES.get(fn, fn))):
        if os.path.exists(c):
            return c
    raise FileNotFoundError(fn)


def table_rows(fn, label):
    """Return the data rows (list of cell lists) of the table with \\label{label} in file fn."""
    s = open(paper_file(fn)).read()
    i = s.index("\\label{%s}" % label)
    j = s.index("\\end{tabular}", i)
    k = s.index("\\midrule", i)
    body = s[k:j]
    rows = []
    for line in body.split("\\\\"):
        line = re.sub(r"(?<!\\)%.*", "", line)
        line = line.replace("\\midrule", "").replace("\\bottomrule", "").strip()
        if not line or "multicolumn" in line and "&" not in line:
            continue
        if "\\multicolumn" in line:
            line = line.split("}", 2)[-1] if line.count("&") == 0 else line
        cells = [c.strip() for c in line.split("&")]
        if len(cells) > 1:
            rows.append(cells)
    return rows


def audit(fn, label, spec):
    """spec: list of (row-label regex, stats dict, {col index: key or (key, 'pm')})."""
    rows = table_rows(fn, label)
    for rx, st, cols in spec:
        hit = [r for r in rows if re.search(rx, r[0] + " " + (r[1] if len(r) > 1 else ""))]
        if len(hit) != 1:
            ROWS.append((label, rx, "-", f"{len(hit)} rows matched", "", "ROW-NOT-FOUND")); continue
        r = hit[0]
        for ci, key in cols.items():
            if key == "pm":
                check(label, rx, ci, r[ci], st["auc"], st["sd"])
            else:
                if key not in st or st[key] is None or (isinstance(st[key], float) and np.isnan(st[key])):
                    ROWS.append((label, rx, ci, r[ci], "n/a", "NOT-COMPUTABLE")); continue
                check(label, rx, ci, r[ci], st[key])


# ---------------------------------------------------------------- table definitions
DP = dict(dp="record", mask="public")
ND = dict(dp="none", mask="topk")
cap = dict(method="capacity", **DP)

# Table: main results (tables.tex)
spec = []
for blk, extra, tgt in [("np", ND, 0.825), ("dp", DP, 0.68)]:
    fa = dict(method="fedavg", dp=extra["dp"])
    rows_np = [("FedAvg", fa)] + [(n, dict(method=m, **extra)) for n, m in
                                   [("Fixed ratio", "fixed"), ("Random ratio", "random"), ("Inverse map", "inverse"),
                                    ("Logistic map", "logistic"), ("Capacity map", "capacity")]]
    for n, c in rows_np:
        spec.append((n, c, tgt, blk))
rows = table_rows("main_tabs.tex", "tab:main")
blocks = {"np": rows[0:6], "dp": rows[6:12]}
for n, c, tgt, blk in spec:
    r = [x for x in blocks[blk] if x[0].replace("\\textbf{", "").startswith(n)]
    st = S(c, S5, target=tgt)
    lab = f"{n} [{blk}]"
    if len(r) != 1:
        ROWS.append(("tab:main", lab, "-", "row", "", "ROW-NOT-FOUND")); continue
    r = r[0]
    check("tab:main", lab, "AUC", r[1], st["auc"], st["sd"])
    check("tab:main", lab, "F1", r[2], st["f1"]); check("tab:main", lab, "BAcc", r[3], st["bacc"])
    check("tab:main", lab, "MB", r[4], st["mb"]); check("tab:main", lab, "Uplink", r[5], st["up"])
    if num(r[6]):
        check("tab:main", lab, "T2T", r[6], st["t2t"])
abl = rows[12:14]
st = S(dict(method="capacity", dp="record", mask="topk"))
check("tab:main", "Top-k after noise", "AUC", abl[0][1], st["auc"], st["sd"]); check("tab:main", "Top-k after noise", "MB", abl[0][4], st["mb"])
check("tab:main", "Top-k after noise", "Uplink", abl[0][5], st["up"])

# Table: eps x T
for r in table_rows("main_tabs.tex", "tab:epsT"):
    e, T = int(r[0]), int(r[1])
    f = S(dict(method="fixed", eps=e, T=T, **DP), S3); c = S(dict(method="capacity", eps=e, T=T, **DP), S3)
    lab = f"eps={e:g},T={T}"
    check("tab:epsT", lab, "sigma", r[2], c["sigma"])
    check("tab:epsT", lab, "AUC fixed", r[3], f["auc"], f["sd"]); check("tab:epsT", lab, "AUC cap", r[4], c["auc"], c["sd"])
    uf, uc = [x.strip() for x in r[5].split("/")]
    check("tab:epsT", lab, "Uplink fixed", uf, f["up"]); check("tab:epsT", lab, "Uplink cap", uc, c["up"])

# Table: one-at-a-time ablation
abl_map = {"default": {}, "s_{\\min}$$=0.05": dict(smin=0.05), "s_{\\min}$$=0.2": dict(smin=0.2),
           "s_{\\max}$$=0.6": dict(smax=0.6), "s_{\\max}$$=0.95": dict(smax=0.95), "$C$$=0.5": dict(clip=0.5),
           "$C$$=2.0": dict(clip=2.0), "\\tau$$=5": dict(tau=5), "\\tau$$=20": dict(tau=20), "$p$$=0.3": dict(part=0.3),
           "$p$$=1.0": dict(part=1.0), "\\alpha$$=0.15": dict(alpha=0.15), "\\alpha$$=1.0": dict(alpha=1.0),
           "\\gamma_{\\mathrm{ref}}$$=12.0": dict(gamma_ref=12.0), "\\gamma_{\\mathrm{ref}}$$=21.0": dict(gamma_ref=21.0)}
for r in table_rows("main_tabs.tex", "tab:ablation"):
    key = [k for k in abl_map if k in r[1]]
    if len(key) != 1:
        ROWS.append(("tab:ablation", r[0] + " " + r[1], "-", "row", "", "ROW-NOT-FOUND")); continue
    m = "logistic" if r[0].startswith("Logistic") else "capacity"
    st = S(dict(method=m, **DP, **abl_map[key[0]]), S3)
    lab = f"{r[0]} {r[1]}"
    check("tab:ablation", lab, "AUC", r[2], st["auc"], st["sd"]); check("tab:ablation", lab, "MB", r[3], st["mb"])
    check("tab:ablation", lab, "Uplink", r[4], st["up"]); check("tab:ablation", lab, "sigma", r[5], st["sigma"])

# Table: robustness
rob = [("Default", {}), ("N{=}50", dict(N=50)), ("N{=}100", dict(N=100)), ("alpha{=}0.1", dict(alpha_dir=0.1)),
       ("alpha{=}1.0", dict(alpha_dir=1.0)), ("blockage &", dict(scenario="blockage")),
       ("Inter-cell", dict(scenario="interf")), ("Blockage + interference", dict(scenario="blockage_interf"))]
rr = table_rows("main_tabs.tex", "tab:robust")
for key, o in rob:
    k2 = key.replace(" &", "")
    hit = [r for r in rr if k2 in r[0]]
    if key == "blockage &":
        hit = [r for r in rr if r[0].startswith("LoS/NLoS")]
    if len(hit) != 1:
        ROWS.append(("tab:robust", key, "-", "row", "", "ROW-NOT-FOUND")); continue
    r = hit[0]
    f = S(dict(method="fixed", **DP, **o), S3); c = S(dict(method="capacity", **DP, **o), S3)
    check("tab:robust", k2, "AUC fixed", r[1], f["auc"], f["sd"]); check("tab:robust", k2, "AUC cap", r[2], c["auc"], c["sd"])
    check("tab:robust", k2, "Uplink fixed", r[3], f["up"]); check("tab:robust", k2, "Uplink cap", r[4], c["up"])
    check("tab:robust", k2, "Reduction %", r[5], 100 * (1 - c["up"] / f["up"]))

# Table: external baselines (reference: CGFL, 5 seeds)
ext = [("DP-FedAvg \\(dense", dict(method="fedavg", dp="record")),
       ("DP-FedAvg \\+ adaptive", dict(method="fedavg", dp="record", adaclip=0.5, adaclip_norm=1)),
       ("Fed-SMP", dict(method="fixed", **DP, explore=1.0)),
       ("QSGD, matched", dict(method="qsgd", dp="record")),
       ("QSGD, capacity", dict(method="qsgd_cap", dp="record")),
       ("Fixed ratio, public", dict(method="fixed", **DP)),
       ("Capacity map, random mask", dict(method="capacity", **DP, explore=1.0)),
       ("CGFL \\+ adaptive", dict(method="capacity", **DP, adaclip=0.5, adaclip_norm=1)),
       ("CGFL \\(capacity map", cap)]
for rx, c in ext:
    st = S(c, S5, ref=cap)
    audit("baselines.tex", "tab:external", [(rx, st, {1: "pm", 2: "f1", 3: "mb", 4: "up", 5: "p"} if "CGFL \\(" not in rx else {1: "pm", 2: "f1", 3: "mb", 4: "up"})])
st = S(dict(method="capacity", **DP, explore=0.3), S3)
audit("baselines.tex", "tab:external", [("exploration", st, {1: "pm", 2: "f1", 3: "mb", 4: "up"})])

# Table: default vs selected configuration
for cfgname, extra in [("Default", {}), ("Selected", dict(tau=5))]:
    for n, c in [("DP-FedAvg", dict(method="fedavg", dp="record")), ("Fixed ratio", dict(method="fixed", **DP)), ("CGFL", cap)]:
        st = S({**c, **extra}, S5)
        audit("tuned.tex", "tab:bestcfg", [(f"^{cfgname} {re.escape(n)}", st, {2: "pm", 3: "f1", 4: "up", 5: "sigma"})])

# Table: superdiagnostic multi-label
def ml_rows(fn, label, extra):
    capnd, capdp = dict(method="capacity", **ND, **extra), dict(method="capacity", **DP, **extra)
    rows = table_rows(fn, label)
    nd_rows, dp_rows = rows[:3], rows[3:6]
    defs = [(nd_rows, [("FedAvg", dict(method="fedavg", dp="none", **extra)), ("Fixed", dict(method="fixed", **ND, **extra)),
                       ("Capacity", capnd)], capnd),
            (dp_rows, [("DP-FedAvg", dict(method="fedavg", dp="record", **extra)), ("Fixed", dict(method="fixed", **DP, **extra)),
                       ("CGFL", capdp)], capdp)]
    return defs


for (rws, items, ref) in ml_rows("multilabel.tex", "tab:multilabel", dict(task="multi")):
    for (n, c), r in zip(items, rws):
        st = S(c, S5, ref=ref)
        lab = f"{n} [{c['dp']}]"
        check("tab:multilabel", lab, "AUC", r[1], st["auc"], st["sd"]); check("tab:multilabel", lab, "F1", r[2], st["f1"])
        check("tab:multilabel", lab, "MB", r[3], st["mb"]); check("tab:multilabel", lab, "Uplink", r[4], st["up"])
        if num(r[5]): check("tab:multilabel", lab, "p", r[5], st["p"])

# Table: 44-label task (rare = labels with < 100 training records)
L = np.load(os.environ["PTBXL_ML"]); tr = L["fold"] <= 8; RARE = np.where(L["Yd"][tr].sum(0) < 100)[0]
for (rws, items, ref) in ml_rows("diag44.tex", "tab:diag", dict(task="multi", labelset="diag")):
    for (n, c), r in zip(items, rws):
        st = S(c, S5, ref=ref, rare=RARE)
        lab = f"{n} [{c['dp']}]"
        check("tab:diag", lab, "AUC", r[1], st["auc"], st["sd"]); check("tab:diag", lab, "Rare AUC", r[2], st["rare"])
        check("tab:diag", lab, "F1", r[3], st["f1"]); check("tab:diag", lab, "MB", r[4], st["mb"])
        check("tab:diag", lab, "Uplink", r[5], st["up"])
        if num(r[6]): check("tab:diag", lab, "p", r[6], st["p"])

# Table: CinC 2021
C = dict(dataset="cinc", task="multi", N=6, part=1.0)
ccap_nd, ccap = dict(method="capacity", **ND, **C), dict(method="capacity", **DP, **C)
rows = table_rows("cinc2021.tex", "tab:cinc")
items = [("FedAvg", dict(method="fedavg", dp="none", **C), ccap_nd), ("Fixed ratio", dict(method="fixed", **ND, **C), ccap_nd),
         ("Capacity map", ccap_nd, None),
         ("DP-FedAvg", dict(method="fedavg", dp="record", **C), ccap), ("Fixed ratio", dict(method="fixed", **DP, **C), ccap),
         ("Random ratio", dict(method="random", **DP, **C), ccap), ("QSGD", dict(method="qsgd_cap", dp="record", **C), ccap),
         ("CGFL", ccap, None)]
sites = ["CPSC", "CPSC-Extra", "Chapman", "Georgia", "Ningbo", "PTB-XL"]
for (n, c, ref), r in zip(items, rows):
    st = S(c, S5, ref=ref)
    lab = f"{n} [{c['dp']}]"
    check("tab:cinc", lab, "AUC", r[1], st["auc"], st["sd"]); check("tab:cinc", lab, "MB", r[2], st["mb"])
    check("tab:cinc", lab, "Uplink", r[3], st["up"])
    if num(r[4]): check("tab:cinc", lab, "p", r[4], st["p"])
    for k, sname in enumerate(sites):
        check("tab:cinc", lab, sname, r[5 + k], st["site_" + sname])

# Table: ResNet scaling
R = dict(arch="resnet", tau=5, lr=0.02, lr_dp=0.5)
rows = table_rows("resnet.tex", "tab:resnet")
cnn_cap, res_cap = dict(method="capacity", **DP, tau=5), dict(method="capacity", **DP, **R)
items = [("CNN DP-FedAvg", dict(method="fedavg", dp="record", tau=5), cnn_cap), ("CNN Fixed", dict(method="fixed", **DP, tau=5), cnn_cap),
         ("CNN CGFL", cnn_cap, None), ("ResNet DP-FedAvg", dict(method="fedavg", dp="record", **R), res_cap),
         ("ResNet Fixed", dict(method="fixed", **DP, **R), res_cap), ("ResNet CGFL", res_cap, None),
         ("ResNet CGFL norm. agg.", dict(method="capacity", **DP, agg="coord", **R), res_cap)]
for (n, c, ref), r in zip(items, rows):
    st = S(c, S5, ref=ref)
    check("tab:resnet", n, "AUC", r[2], st["auc"], st["sd"]); check("tab:resnet", n, "F1", r[3], st["f1"])
    check("tab:resnet", n, "MB", r[4], st["mb"]); check("tab:resnet", n, "Uplink", r[5], st["up"])
    if num(r[6]): check("tab:resnet", n, "p", r[6], st["p"])

# Table: normalised aggregation
rows = table_rows("norm_agg.tex", "tab:normagg")
for r, extra in zip(rows[:3], [{}, dict(tau=5), C]):
    for k, m in enumerate(["fixed", "capacity"]):
        w = S(dict(method=m, **DP, **extra)); c = S(dict(method=m, **DP, agg="coord", **extra))
        check("tab:normagg", r[0], f"{m} weighted", r[1 + 2 * k], w["auc"]); check("tab:normagg", r[0], f"{m} normalised", r[2 + 2 * k], c["auc"])
w = S(dict(method="capacity", **ND)); c = S(dict(method="capacity", **ND, agg="coord"))
check("tab:normagg", rows[3][0], "capacity weighted", rows[3][3], w["auc"]); check("tab:normagg", rows[3][0], "capacity normalised", rows[3][4], c["auc"])

# Table: end-to-end (arithmetic consistency: t_c*T + uplink)
rows = table_rows("wireless.tex", "tab:e2e")
for r in rows:
    tc = float(re.match(r"[\d.]+", r[0]).group()) if r[0][0].isdigit() and not r[0].startswith("0 (") else 0.0
    vals = [num(x)[0] for x in r[2:6]]
    up_scale = 10.0 if "1 MHz" in r[1] else 1.0
    for name, upl, cell in [("DP-FedAvg", 372.7, r[2]), ("Fixed", 103.9, r[3]), ("CGFL", 37.8, r[5])]:
        check("tab:e2e", f"{r[0]} {r[1]}", name, cell, tc * 200 + upl * up_scale, tol_extra=1.5)
    fixed, bwa, cg = vals[1], vals[2], vals[3]
    check("tab:e2e", f"{r[0]} {r[1]}", "vs Fixed %", r[6], -100 * (1 - cg / fixed), tol_extra=0.6)
    check("tab:e2e", f"{r[0]} {r[1]}", "vs BWA %", r[7], -100 * (1 - cg / bwa), tol_extra=0.6)

with open(OUT, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["table", "row", "column", "paper", "recomputed", "status"]); w.writerows(ROWS)
from collections import Counter
print(Counter(r[5] for r in ROWS))
for r in ROWS:
    if r[5] != "OK":
        print(" | ".join(map(str, r)))
