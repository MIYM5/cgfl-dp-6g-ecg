#!/usr/bin/env python3
"""
CGFL: Channel-aware gradient sparsification for DP federated learning.
Built to run on a Kaggle weekly GPU quota:
  * whole PTB-XL set kept on the GPU as fp16, with no DataLoader and no file I/O per round
  * compact 1-D CNN with GroupNorm (DP-compatible, ~0.15 M params, edge/TinyML-sized)
  * per-sample gradients via torch.func (vmap) instead of Opacus hooks, which is faster
  * record-level DP-SGD at each hospital with Poisson sampling and an RDP accountant (Opacus)
  * checkpoint/resume + skip-completed runs, so a 12 h session cutoff loses at most a few rounds
  * --shard i/n so the two T4s of a Kaggle "GPU T4 x2" session each run half the grid

Usage
  python cgfl.py prep  --out /kaggle/working/ptbxl100.npz          # one-off, CPU session
  python cgfl.py bench --data ptbxl100.npz                          # time 5 rounds -> projects grid cost
  python cgfl.py grid  --data ptbxl100.npz --stage main --shard 0/2 --device cuda:0
  python cgfl.py run   --data ptbxl100.npz --method capacity --dp record --mask public --seed 1
  python cgfl.py summarize --results results                        # mean +/- std, 95% CI, Wilcoxon
  python cgfl.py grid --synthetic ...                               # CPU smoke test, no data needed
"""
import warnings; warnings.filterwarnings("ignore", message="Optimal order")
import argparse, glob, hashlib, itertools, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

# ----------------------------------------------------------------------------------------------
# 1. Data preparation (run ONCE in a CPU session, which does not use GPU quota; save the npz as a Kaggle dataset)
# ----------------------------------------------------------------------------------------------
def prep_ptbxl(out, root=None):
    import ast, pandas as pd
    if root is None:
        hits = glob.glob("/kaggle/input/**/ptbxl_database.csv", recursive=True)
        if not hits:
            sys.exit("ptbxl_database.csv not found under /kaggle/input. Attach a PTB-XL dataset.")
        root = os.path.dirname(hits[0])
    db = pd.read_csv(os.path.join(root, "ptbxl_database.csv"), index_col="ecg_id")
    db.scp_codes = db.scp_codes.apply(ast.literal_eval)
    scp = pd.read_csv(os.path.join(root, "scp_statements.csv"), index_col=0)
    scp = scp[scp.diagnostic == 1]

    def primary(codes):  # superclass of the highest-likelihood diagnostic code (single-label task)
        best, lik = None, -1
        for c, v in codes.items():
            if c in scp.index and v > lik:
                best, lik = scp.loc[c, "diagnostic_class"], v
        return best
    db["y"] = db.scp_codes.apply(primary)
    db = db[db.y.isin(CLASSES)]
    X = np.zeros((len(db), 12, 1000), np.float16)
    for i, f in enumerate(db.filename_lr):
        X[i] = read_wfdb16(os.path.join(root, f))[:, :1000].astype(np.float16)
        if i % 2000 == 0:
            print(f"  {i}/{len(db)}", flush=True)
    y = db.y.map({c: i for i, c in enumerate(CLASSES)}).values.astype(np.int64)
    fold = db.strat_fold.values
    # per-lead standardisation with TRAIN statistics only (folds 1-8)
    tr = fold <= 8
    mu = X[tr].astype(np.float32).mean((0, 2), keepdims=True)
    sd = X[tr].astype(np.float32).std((0, 2), keepdims=True) + 1e-6
    X = ((X.astype(np.float32) - mu) / sd).astype(np.float16)
    np.savez_compressed(out, X=X, y=y, fold=fold)
    print("saved", out, X.shape, np.bincount(y))


def find_ptbxl_csv_root():
    """Locate ptbxl_database.csv + scp_statements.csv under /kaggle/input with ONE pruned walk
    (skips the raw-ECG folders, so attaching the raw dataset no longer costs minutes)."""
    for r, dirs, files in os.walk("/kaggle/input", followlinks=True):
        dirs[:] = [d for d in dirs if d not in ("records100", "records500", "records")]
        if "ptbxl_database.csv" in files and "scp_statements.csv" in files:
            return r
    sys.exit("ptbxl_database.csv + scp_statements.csv not found under /kaggle/input")


def prep_ml(out, root=None):
    """Labels for the STANDARD PTB-XL superdiagnostic multi-label task (Strodthoff et al. 2021),
    aligned row-by-row with ptbxl100.npz (same records, same order). Needs only the two CSV files."""
    import ast, pandas as pd
    root = root or find_ptbxl_csv_root()
    db = pd.read_csv(os.path.join(root, "ptbxl_database.csv"), index_col="ecg_id")
    db.scp_codes = db.scp_codes.apply(ast.literal_eval)
    scp = pd.read_csv(os.path.join(root, "scp_statements.csv"), index_col=0)
    scp = scp[scp.diagnostic == 1]

    def primary(codes):  # identical to prep_ptbxl -> identical row filter and order
        best, lik = None, -1
        for c, v in codes.items():
            if c in scp.index and v > lik:
                best, lik = scp.loc[c, "diagnostic_class"], v
        return best

    def multi(codes):    # every diagnostic superclass present (standard benchmark aggregation)
        v = np.zeros(5, np.uint8)
        for c in codes:
            if c in scp.index and scp.loc[c, "diagnostic_class"] in CLASSES:
                v[CLASSES.index(scp.loc[c, "diagnostic_class"])] = 1
        return v
    diag = sorted(scp.index)   # the 44 diagnostic statements (benchmark task "diag")

    def multi_diag(codes):   # every diagnostic statement present, any likelihood (benchmark aggregation)
        v = np.zeros(len(diag), np.uint8)
        for c in codes:
            if c in scp.index:
                v[diag.index(c)] = 1
        return v
    db["y"] = db.scp_codes.apply(primary)
    db = db[db.y.isin(CLASSES)]
    Y = np.stack(db.scp_codes.apply(multi).values)
    Yd = np.stack(db.scp_codes.apply(multi_diag).values)
    y = db.y.map({c: i for i, c in enumerate(CLASSES)}).values.astype(np.int64)
    np.savez_compressed(out, Y=Y, y=y, fold=db.strat_fold.values, Yd=Yd, diag_classes=np.array(diag))
    print("saved", out, Y.shape, "positives per class", Y.sum(0), "mean labels/record", Y.sum(1).mean().round(3))
    print("      diag task:", Yd.shape, "labels; positives min/median/max",
          int(Yd.sum(0).min()), int(np.median(Yd.sum(0))), int(Yd.sum(0).max()))


# ---- PhysioNet/CinC Challenge 2021: real multi-institution sites ------------------------------------
CINC_URL = "https://physionet.org/files/challenge-2021/1.0.3/"
# the 26 scored diagnoses of the Challenge (evaluation-2021/dx_mapping_scored.csv), equivalent codes merged
CINC_SCORED = {"164889003": "AF", "164890007": "AFL", "6374002": "BBB", "426627000": "Brady",
               "164909002": "LBBB", "59118001": "RBBB", "270492004": "IAVB", "713426002": "IRBBB",
               "39732003": "LAD", "445118002": "LAnFB", "164947007": "LPR", "251146004": "LQRSV",
               "111975006": "LQT", "698252002": "NSIVCB", "426783006": "NSR", "63593006": "PAC",
               "10370003": "PR", "365413008": "PRWP", "17338001": "PVC", "164917005": "QAb",
               "47665007": "RAD", "427393009": "SA", "426177001": "SB", "427084000": "STach",
               "164934002": "TAb", "59931005": "TInv"}
CINC_EQUIV = {"733534002": "164909002", "713427006": "59118001", "284470004": "63593006", "427172004": "17338001"}
CINC_SITES = {"cpsc_2018": "CPSC", "cpsc_2018_extra": "CPSC-Extra", "ptb": "PTB", "ptb-xl": "PTB-XL",
              "georgia": "Georgia", "chapman_shaoxing": "Chapman", "chapman-shaoxing": "Chapman", "ningbo": "Ningbo"}
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def cinc_record(hea_txt, mat_bytes, fs_out=100, n_out=1000):
    """One CinC-2021 record -> (12, 1000) float32 in mV at 100 Hz (first 10 s, zero-padded) + Dx codes.
    The .mat files are MATLAB v4 with a 24-byte header followed by interleaved int16 ('16x1+24')."""
    from fractions import Fraction
    from scipy.signal import resample_poly
    lines = [l for l in hea_txt.splitlines() if l.strip()]
    h = lines[0].split(); n_sig, fs, n_samp = int(h[1]), float(h[2]), int(h[3])
    gains, bases, names, off = [], [], [], 24
    for l in lines[1:1 + n_sig]:
        tok = l.split()
        if "+" in tok[1]:
            off = int(tok[1].split("+")[1])
        g = tok[2].split("/")[0]
        bases.append(float(g[g.index("(") + 1:g.index(")")]) if "(" in g else 0.0)
        gains.append(float(g.split("(")[0]) or 1000.0)
        names.append(tok[-1])
    raw = np.frombuffer(mat_bytes, dtype="<i2", offset=off)[: n_samp * n_sig].reshape(n_samp, n_sig)
    x = ((raw - np.array(bases)) / np.array(gains)).T
    x = x[[names.index(L) for L in LEADS]] if all(L in names for L in LEADS) else x[:12]
    fr = Fraction(fs_out, int(round(fs))).limit_denominator(1000)
    x = resample_poly(x, fr.numerator, fr.denominator, axis=1)[:, :n_out]
    if x.shape[1] < n_out:
        x = np.pad(x, ((0, 0), (0, n_out - x.shape[1])))
    dx = []
    for l in lines:
        if l.lstrip("#").strip().lower().startswith("dx:"):
            dx = [c.strip() for c in l.split(":", 1)[1].split(",") if c.strip()]
    return x.astype(np.float32), [CINC_EQUIV.get(c, c) for c in dx]


def cinc_site_of(rec):
    """Institution from the record name (works for any folder layout, e.g. Kaggle mirrors).
    A: CPSC 2018, Q: CPSC-Extra, S: PTB, HR: PTB-XL, E: Georgia, JS00001-JS10646: Chapman-Shaoxing,
    JS10647+: Ningbo, I: INCART (excluded)."""
    if rec.startswith("HR"): return "ptb-xl"
    if rec.startswith("JS"): return "chapman_shaoxing" if int(rec[2:]) <= 10646 else "ningbo"
    return {"A": "cpsc_2018", "Q": "cpsc_2018_extra", "S": "ptb", "E": "georgia"}.get(rec[:1])


def _cinc_iter_local(root):
    """(site, hea_path) for a local copy / Kaggle mirror; INCART excluded (74 Holter-length records)."""
    for r, dirs, files in os.walk(root, followlinks=True):
        for f in files:
            if f.endswith(".hea") and os.path.exists(os.path.join(r, f[:-4] + ".mat")):
                site = cinc_site_of(f[:-4])
                if site:
                    yield site, os.path.join(r, f[:-4])


def prep_cinc(out, root=None, threads=8, min_pos=20, min_sites=4, min_site=200, max_groups=6, budget_h=11.0):
    """Download (or read) CinC-2021 training data, keep the scored diagnoses that are labelled in at least
    `min_sites` institutions with >= `min_pos` records each, drop records with none of them, split every
    institution 80/10/10 (fixed seed), standardise with training statistics and save one npz."""
    import concurrent.futures as cf, urllib.request, shutil
    get = lambda u: urllib.request.urlopen(u, timeout=120).read()
    T0 = time.time()
    if root:   # local mirror: group = directory
        groups = {}
        for site, base in _cinc_iter_local(root):
            groups.setdefault(os.path.dirname(base), []).append((site, base, True))
    else:      # direct download from PhysioNet (CPU session, Internet ON). ~90 groups of <=1000 records
        gl = [g for g in get(CINC_URL + "RECORDS").decode().split() if "incart" not in g]
        # cap the groups (<=1000 records each) taken per institution: PhysioNet throttles bulk downloads, and
        # ~6k records per institution are plenty for 6 clients (Ningbo alone has 35 groups)
        per = {}
        for g in sorted(gl, key=lambda g: (g.split("/")[1], int(g.rstrip("/").split("/g")[-1]))):
            per.setdefault(g.split("/")[1], []).append(g)
        gl = [g for v in per.values() for g in v[:max_groups]]
        def recs(g):
            return g, [(g.split("/")[1], CINC_URL + g + r, False) for r in get(CINC_URL + g + "RECORDS").decode().split()]
        with cf.ThreadPoolExecutor(threads) as ex:
            groups = dict(ex.map(recs, gl))
    print(f"{sum(map(len, groups.values()))} records in {len(groups)} groups", flush=True)

    def work(job):
        site, base, local = job
        err = None
        for attempt in range(8):
            try:
                if local:
                    hea, mat = open(base + ".hea").read(), open(base + ".mat", "rb").read()
                else:
                    hea, mat = get(base + ".hea").decode(), get(base + ".mat")
                x, dx = cinc_record(hea, mat)
                return site, os.path.basename(base), x.astype(np.float16), dx
            except Exception as e:
                err = e
                if "404" in str(e):          # record listed in RECORDS but withdrawn: permanent, skip
                    return "404"
                time.sleep(5 * (attempt + 1))
        print("  failed", base, err, flush=True)
        return None
    # per-group cache -> a crashed or timed-out session resumes where it stopped
    cache = os.path.join(os.path.dirname(os.path.abspath(out)), "cinc_cache"); os.makedirs(cache, exist_ok=True)
    res, t0 = [], time.time()
    for gi, (gname, jobs) in enumerate(sorted(groups.items())):
        cf_ = os.path.join(cache, hashlib.md5(gname.encode()).hexdigest()[:12] + ".npz")
        if not os.path.exists(cf_):
            if time.time() - T0 > budget_h * 3600:
                print(f"!! time budget reached after {gi} groups: commit again with this output attached "
                      "(finished groups are cached)", flush=True)
                return
            with cf.ThreadPoolExecutor(threads) as ex:
                got = list(ex.map(work, jobs))
            n_fail = sum(r is None for r in got)
            part = [r for r in got if r is not None and r != "404"]
            if n_fail > 0.02 * len(jobs):      # too many transient failures: do NOT cache, retry next commit
                print(f"  group {gname}: {n_fail} transient failures -> not cached, will retry", flush=True)
                continue
            np.savez(cf_, X=np.stack([r[2] for r in part]), site=[r[0] for r in part], rec=[r[1] for r in part],
                     dx=np.array([",".join(r[3]) for r in part]))
        z = np.load(cf_)
        res += [(st, rc, x, [c for c in d.split(",") if c]) for st, rc, x, d in zip(z["site"], z["rec"], z["X"], z["dx"])]
        print(f"  group {gi+1}/{len(groups)} {gname}: {len(res)} records, {time.time()-t0:.0f}s", flush=True)
    missing = [g for g in groups if not os.path.exists(os.path.join(cache, hashlib.md5(g.encode()).hexdigest()[:12] + ".npz"))]
    if missing:
        print(f"!! {len(missing)} groups still missing (network): commit again with this output attached", flush=True)
        return
    res = [(str(a), str(b), c, d) for a, b, c, d in res]
    sites = sorted({CINC_SITES[r[0]] for r in res})
    sid = np.array([sites.index(CINC_SITES[r[0]]) for r in res])
    codes = sorted(CINC_SCORED)
    Yall = np.array([[c in set(r[3]) for c in codes] for r in res], np.uint8)
    per_site = np.array([Yall[sid == k].sum(0) for k in range(len(sites))])          # (sites, codes)
    keep_c = (per_site >= min_pos).sum(0) >= min_sites
    Y = Yall[:, keep_c]
    keep_r = Y.sum(1) > 0
    big = [k for k in range(len(sites)) if (keep_r & (sid == k)).sum() >= min_site]
    keep_r &= np.isin(sid, big)
    names = [sites[k] for k in big]
    remap = {k: i for i, k in enumerate(big)}
    X = np.stack([r[2] for r, k in zip(res, keep_r) if k])
    Y, sid = Y[keep_r], np.array([remap[k] for k in sid[keep_r]])
    rec = np.array([r[1] for r, k in zip(res, keep_r) if k])
    g = np.random.default_rng(0)
    fold = np.ones(len(Y), np.int64)
    for k in range(len(names)):     # per-institution 80/10/10 split (fold 1 = train, 9 = val, 10 = test)
        ids = g.permutation(np.where(sid == k)[0]); n = len(ids)
        fold[ids[int(.8 * n):int(.9 * n)]] = 9; fold[ids[int(.9 * n):]] = 10
    tr = fold <= 8
    mu = X[tr].astype(np.float32).mean((0, 2), keepdims=True)
    sd = X[tr].astype(np.float32).std((0, 2), keepdims=True) + 1e-6
    X = ((X.astype(np.float32) - mu) / sd).astype(np.float16)
    cls = [CINC_SCORED[c] for c, k in zip(codes, keep_c) if k]
    np.savez(out, X=X, Y=Y, y=Y.argmax(1), fold=fold, site=sid, site_names=np.array(names),
             classes=np.array(cls), rec=rec)
    print("saved", out, X.shape, "| sites:", dict(zip(names, np.bincount(sid))), "| classes:", cls)
    print("positives per site x class:")
    for k, nm in enumerate(names):
        print(f"  {nm:11s}", Y[sid == k].sum(0).tolist())
    shutil.rmtree(cache, ignore_errors=True)   # keep the notebook output small


def read_wfdb16(base):
    """Minimal reader for PTB-XL WFDB records (format 16, interleaved int16) -> (12, L) in mV.
    Avoids needing `pip install wfdb` (Kaggle sessions without internet)."""
    lines = [l for l in open(base + ".hea").read().splitlines() if l and not l.startswith("#")]
    n_sig, n_samp = int(lines[0].split()[1]), int(lines[0].split()[3])
    gains, bases = [], []
    for l in lines[1:1 + n_sig]:
        tok = l.split()
        assert tok[1].startswith("16"), "only WFDB format 16 supported"
        g = tok[2].split("/")[0]
        bases.append(float(g[g.index("(") + 1:g.index(")")]) if "(" in g else 0.0)
        gains.append(float(g.split("(")[0]) or 200.0)
    raw = np.fromfile(base + ".dat", dtype="<i2")[: n_samp * n_sig].reshape(n_samp, n_sig)
    return ((raw - np.array(bases)) / np.array(gains)).T


def load_data(path, synthetic, device, labels=None):
    if synthetic:  # tiny learnable fake ECG set for smoke tests
        g = np.random.default_rng(0)
        n = 3000
        y = g.choice(5, n, p=[.44, .25, .15, .11, .05])
        t = np.linspace(0, 1, 1000)
        X = g.normal(0, 1, (n, 12, 1000)) * .5
        X += np.sin(2 * np.pi * (y[:, None, None] + 1) * 3 * t)[:, :, :]  # class-dependent rhythm
        fold = g.integers(1, 11, n)
        X = X.astype(np.float16)
    else:
        d = np.load(path)
        X, y, fold = d["X"], d["y"], d["fold"]
        if "site" in d.files:   # CinC-2021: multi-label targets and natural sites ship inside the npz
            tr, va, te = fold <= 8, fold == 9, fold == 10
            T = lambda a: torch.as_tensor(a, device=device)
            Y = d["Y"].astype(np.float32)
            S = d["site"]
            site_cls = np.stack([Y[tr & (S == k)].sum(0) > 0 for k in range(len(d["site_names"]))])  # (sites, C)
            return dict(site_cls=site_cls, ann_va=site_cls[S[va]], ann_te=site_cls[S[te]],Xtr=T(X[tr]), ytr=T(y[tr]), Xva=T(X[va]), yva=T(y[va]), Xte=T(X[te]), yte=T(y[te]),
                        Ytr=T(Y[tr]), Yva=T(Y[va]), Yte=T(Y[te]), site_tr=d["site"][tr], site_te=d["site"][te],
                        site_names=[str(s) for s in d["site_names"]], classes=[str(c) for c in d["classes"]],
                        kind="cinc")
    tr, va, te = fold <= 8, fold == 9, fold == 10
    T = lambda a: torch.as_tensor(a, device=device)
    D = dict(Xtr=T(X[tr]), ytr=T(y[tr]), Xva=T(X[va]), yva=T(y[va]), Xte=T(X[te]), yte=T(y[te]))
    if synthetic:   # fake multi-label targets: true class + a random co-occurring one
        g = np.random.default_rng(1)
        Y = np.eye(5, dtype=np.uint8)[y]; Y[np.arange(len(y)), g.integers(0, 5, len(y))] |= (g.random(len(y)) < .3)
    elif labels:
        L = np.load(labels)
        assert len(L["y"]) == len(y) and (L["y"] == y).all() and (L["fold"] == fold).all(), \
            "multi-label file is not aligned with the npz: rebuild it with `cgfl.py prep_ml`"
        Y = L["Y"]
        Yd = L["Yd"] if "Yd" in L.files else None   # 44 diagnostic statements (task "diag"), if built
    else:
        Y = None
    if Y is not None:
        D.update(Ytr=T(Y[tr].astype(np.float32)), Yva=T(Y[va].astype(np.float32)), Yte=T(Y[te].astype(np.float32)))
    if not synthetic and labels and Yd is not None:
        D.update(Ydtr=T(Yd[tr].astype(np.float32)), Ydva=T(Yd[va].astype(np.float32)), Ydte=T(Yd[te].astype(np.float32)))
    D["kind"] = "ptbxl"
    return D


def dirichlet_split(y, n_clients, alpha, seed, min_size=100):
    """Label-Dirichlet non-IID split; tiny clients are topped up from the largest ones so that every
    hospital can run DP-SGD with a sensible sampling rate (report the min/max sizes in the paper)."""
    g = np.random.default_rng(seed)
    y = np.asarray(y)
    min_size = int(min(min_size, 0.3 * len(y) / n_clients))
    idx = [[] for _ in range(n_clients)]
    for c in np.unique(y):
        ids = g.permutation(np.where(y == c)[0])
        p = g.dirichlet([alpha] * n_clients)
        for k, part in enumerate(np.split(ids, (np.cumsum(p)[:-1] * len(ids)).astype(int))):
            idx[k] += part.tolist()
    for k in range(n_clients):
        while len(idx[k]) < min_size:
            big = max(range(n_clients), key=lambda j: len(idx[j]))
            take = min(min_size - len(idx[k]), len(idx[big]) - min_size)
            g.shuffle(idx[big])
            idx[k] += idx[big][:take]; idx[big] = idx[big][take:]
    return [np.array(i) for i in idx]


# ----------------------------------------------------------------------------------------------
# 2. Model: compact 1-D CNN, GroupNorm (BatchNorm breaks per-sample DP)
# ----------------------------------------------------------------------------------------------
class ECGNet(nn.Module):
    def __init__(self, width=32, n_cls=5):
        super().__init__()
        w = width
        def blk(i, o, k, s):
            return nn.Sequential(nn.Conv1d(i, o, k, s, k // 2, bias=False), nn.GroupNorm(8, o), nn.ReLU())
        self.f = nn.Sequential(blk(12, w, 15, 2), blk(w, w, 9, 2), blk(w, 2 * w, 9, 2),
                               blk(2 * w, 2 * w, 7, 2), blk(2 * w, 4 * w, 5, 2), blk(4 * w, 4 * w, 3, 1))
        self.head = nn.Linear(4 * w, n_cls)

    def forward(self, x):
        return self.head(self.f(x).mean(-1))

class ResBlock1D(nn.Module):
    def __init__(self, i, o, k, s):
        super().__init__()
        self.c1 = nn.Conv1d(i, o, k, s, k // 2, bias=False); self.n1 = nn.GroupNorm(8, o)
        self.c2 = nn.Conv1d(o, o, k, 1, k // 2, bias=False); self.n2 = nn.GroupNorm(8, o)
        self.sc = None if (i == o and s == 1) else nn.Sequential(nn.Conv1d(i, o, 1, s, bias=False), nn.GroupNorm(8, o))

    def forward(self, x):
        h = F.relu(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return F.relu(h + (x if self.sc is None else self.sc(x)))


class ResNet1D(nn.Module):
    """Larger 1-D ResNet (GroupNorm, DP-compatible) used to test how the methods scale with d.
    width=32 -> stage widths 64/128/192/256, two basic blocks per stage, kernel 5 (~2 M parameters)."""
    def __init__(self, width=32, n_cls=5, k=5):
        super().__init__()
        ws = [2 * width, 4 * width, 6 * width, 8 * width]
        self.stem = nn.Sequential(nn.Conv1d(12, ws[0], 15, 2, 7, bias=False), nn.GroupNorm(8, ws[0]), nn.ReLU())
        layers, c = [], ws[0]
        for j, w in enumerate(ws):
            layers += [ResBlock1D(c, w, k, 1 if j == 0 else 2), ResBlock1D(w, w, k, 1)]
            c = w
        self.f = nn.Sequential(*layers)
        self.head = nn.Linear(c, n_cls)

    def forward(self, x):
        return self.head(self.f(self.stem(x)).mean(-1))


def make_model(a, n_cls):
    return ResNet1D(a.width, n_cls) if getattr(a, "arch", "cnn") == "resnet" else ECGNet(a.width, n_cls)

# ----------------------------------------------------------------------------------------------
# 3. 6G uplink channel (3GPP TR 38.901 UMa, optional blockage + interference)
# ----------------------------------------------------------------------------------------------
class Channel:
    def __init__(self, n, seed, scenario="los", fc=28.0, ptx_dbm=23.0, bw=100e6, nf_db=0.0,
                 dmin=50, dmax=250, inr_db=6.0):
        g = np.random.default_rng(seed + 991)
        self.g, self.n, self.scenario, self.bw = g, n, scenario, bw
        self.d = g.uniform(dmin, dmax, n)
        pl_los = 28.0 + 22 * np.log10(self.d) + 20 * np.log10(fc)
        pl_nlos = 13.54 + 39.08 * np.log10(self.d) + 20 * np.log10(fc)  # simplified UMa NLoS
        self.pl_los, self.pl_nlos = pl_los, np.maximum(pl_los, pl_nlos)
        self.p_los = np.minimum(18 / self.d, 1) * (1 - np.exp(-self.d / 63)) + np.exp(-self.d / 63)
        self.noise_dbm = -174 + 10 * np.log10(bw) + nf_db
        self.ptx, self.inr = ptx_dbm, inr_db

    def snr_db(self):
        """Per-client instantaneous SNR (SINR) in dB for one round (block fading)."""
        pl = self.pl_los.copy()
        if self.scenario in ("blockage", "blockage_interf"):
            los = self.g.random(self.n) < self.p_los
            pl = np.where(los, self.pl_los, self.pl_nlos)
        fad = self.g.exponential(1.0, self.n)  # |h|^2 ~ Exp(1), Rayleigh
        snr = self.ptx - pl - self.noise_dbm + 10 * np.log10(fad)
        if self.scenario in ("interf", "blockage_interf"):  # random-activity inter-cell interference
            i_lin = 10 ** (self.inr / 10) * self.g.exponential(1.0, self.n) * (self.g.random(self.n) < .5)
            snr = snr - 10 * np.log10(1 + i_lin)
        return snr

# ----------------------------------------------------------------------------------------------
# 4. SNR -> retention-ratio policies  (s = RETAINED fraction)
# ----------------------------------------------------------------------------------------------
def capacity_bits(snr_db, bw_per_user):
    return bw_per_user * np.log2(1 + 10 ** (np.asarray(snr_db) / 10))  # bit/s


def retention(method, snr_db, a, rng, s_bar):
    smin, smax = a.smin, a.smax
    if method == "fedavg":
        return np.ones_like(snr_db)
    if method in ("fixed", "qsgd"):     # qsgd: s only sets the matched bit width (qsgd_bits)
        return np.full_like(snr_db, a.s_fixed if a.s_fixed > 0 else s_bar)
    if method == "logistic":       # conference CGFL map, eq. (6)
        return smin + (smax - smin) / (1 + np.exp(-a.alpha * (snr_db - a.gamma_ref)))
    if method == "inverse":        # mirror image: dense on BAD links (tests the monotonicity claim)
        return smin + (smax - smin) / (1 + np.exp(a.alpha * (snr_db - a._inv_gref)))
    if method == "random":         # channel-blind, same mean
        half = min(s_bar - smin, smax - s_bar)           # symmetric around s_bar, inside [smin, smax]
        return rng.uniform(s_bar - half, s_bar + half, len(snr_db))
    if method in ("capacity", "qsgd_cap"):       # closed form from the latency-constrained problem
        r = capacity_bits(snr_db, a.bw_user)
        return np.clip(a._tmax_coef * r, smin, smax)
    raise ValueError(method)


def calibrate(a, n_clients, seed):
    """Choose T_max (capacity map) and s_bar so every channel-aware policy has the SAME expected
    retention as the logistic map -> matched communication budget."""
    ch = Channel(n_clients, seed, a.scenario)          # SAME client distances as the run ...
    ch.g = np.random.default_rng(seed + 12345)          # ... but independent fading draws
    snr = np.concatenate([ch.snr_db() for _ in range(400)])
    s_bar = (a.smin + (a.smax - a.smin) / (1 + np.exp(-a.alpha * (snr - a.gamma_ref)))).mean()
    if a.s_budget > 0:
        s_bar = a.s_budget
    r = capacity_bits(snr, a.bw_user)
    lo, hi = 1e-15, 1e-3
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        m = np.clip(mid * r, a.smin, a.smax).mean()
        lo, hi = (mid, hi) if m < s_bar else (lo, mid)
    a._tmax_coef = math.sqrt(lo * hi)
    lo, hi = -60.0, 120.0          # inverse map: shift its midpoint until its mean retention is also s_bar
    for _ in range(100):
        mid = (lo + hi) / 2
        m = (a.smin + (a.smax - a.smin) / (1 + np.exp(a.alpha * (snr - mid)))).mean()
        lo, hi = (lo, mid) if m > s_bar else (mid, hi)
    a._inv_gref = (lo + hi) / 2
    return float(s_bar)

# ----------------------------------------------------------------------------------------------
# 5. Training utilities
# ----------------------------------------------------------------------------------------------
def flat(params):
    return torch.cat([p.reshape(-1) for p in params.values()])


def unflat(vec, like):
    out, i = {}, 0
    for k, p in like.items():
        out[k] = vec[i:i + p.numel()].view_as(p)
        i += p.numel()
    return out


@torch.no_grad()
def evaluate(model, X, y, bs=1024, multi=False, ann=None):
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
    model.eval()
    if multi:   # standard superdiagnostic task: macro-AUC over the 5 labels, F1 at threshold 0.5
        P = torch.cat([torch.sigmoid(model(X[i:i + bs].float())) for i in range(0, len(X), bs)]).cpu().numpy()
        Yt = y.cpu().numpy().astype(int); pred = (P > 0.5).astype(int)
        # ann[i, c] = 1 if record i comes from an institution that annotates diagnosis c (CinC-2021: label
        # vocabularies differ by institution, so an absent code there is "unknown", not "negative")
        A = np.ones_like(Yt, dtype=bool) if ann is None else np.asarray(ann, dtype=bool)
        ok = [c for c in range(Yt.shape[1]) if 0 < Yt[A[:, c], c].sum() < A[:, c].sum()]
        auc = float(np.mean([roc_auc_score(Yt[A[:, c], c], P[A[:, c], c]) for c in ok])) if ok else float("nan")
        bacc = np.mean([balanced_accuracy_score(Yt[:, c], pred[:, c]) for c in range(Yt.shape[1])])
        lab = {int(c): float(roc_auc_score(Yt[A[:, c], c], P[A[:, c], c])) for c in ok}
        return dict(auc=float(auc), acc=float((pred == Yt).all(1).mean()), bacc=float(bacc),
                    f1=float(f1_score(Yt, pred, average="macro", zero_division=0)),
                    pred_hist=pred.sum(0).tolist(), label_auc=lab, label_pos=Yt.sum(0).tolist())
    P = torch.cat([F.softmax(model(X[i:i + bs].float()), -1) for i in range(0, len(X), bs)]).cpu().numpy()
    yt = y.cpu().numpy()
    pred = P.argmax(1)
    try:
        auc = roc_auc_score(yt, P, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")
    return dict(auc=float(auc), acc=float((pred == yt).mean()),
                bacc=float(balanced_accuracy_score(yt, pred)),
                f1=float(f1_score(yt, pred, average="macro", zero_division=0)),
                pred_hist=np.bincount(pred, minlength=5).tolist())


def local_train(model, gparams, X, y, w_cls, a, mask, sigma, q, gen, clip=None, cls_mask=None):
    """tau local steps. Non-DP: plain SGD. DP: per-sample clip + Gaussian noise, Poisson batches.
    If `mask` (public, flat bool) is given, gradients are restricted to it BEFORE clipping/noise,
    so the noise lives in s*d dims instead of d (the DP<->sparsification synergy)."""
    names = list(gparams.keys())
    p = {k: v.clone().requires_grad_(a.dp != "record") for k, v in gparams.items()}
    mvec = None if mask is None else unflat(mask.float(), gparams)
    n = len(X)

    multi = a.task == "multi"
    lossf = (lambda o, t: F.binary_cross_entropy_with_logits(o, t, weight=cls_mask, pos_weight=w_cls)) if multi else \
            (lambda o, t: F.cross_entropy(o, t, weight=w_cls))

    def loss_one(params, xb, yb):
        out = functional_call(model, params, (xb.unsqueeze(0).float(),))
        return lossf(out, yb.unsqueeze(0))
    ps_grad = vmap(grad(loss_one), in_dims=(None, 0, 0))
    C = a.clip if clip is None else clip
    # Adaptive clipping (Andrew et al., NeurIPS 2021) adapted to record-level DP-SGD: the clip norm tracks
    # the a.adaclip-quantile of per-example norms via a noisy, centred count. Releasing the noisy sum
    # (multiplier s_D) and the noisy count (s_b, sensitivity 1/2) is equivalent to ONE Gaussian mechanism
    # with 1/sigma^2 = 1/s_D^2 + 1/(2 s_b)^2, so the accountant's sigma is unchanged; we pay a small
    # (1 + adaclip_infl) inflation of the gradient noise to fund the quantile estimate.
    ada = a.dp == "record" and a.adaclip > 0
    if ada:
        s_D = sigma * (1 + a.adaclip_infl)
        s_b = 0.5 / math.sqrt(sigma ** -2 - s_D ** -2)

    for _ in range(a.tau):
        if a.dp == "record":
            sel = torch.nonzero(torch.rand(n, generator=gen, device=X.device) < q).squeeze(1)
            if len(sel) == 0:
                continue
            sel = sel[: a.max_phys_batch]  # memory guard (rarely binds: expected batch = a.batch)
            gs = ps_grad(p, X[sel], y[sel])
            if mvec is not None:
                gs = {k: g * mvec[k] for k, g in gs.items()}
            norms = torch.sqrt(sum(g.reshape(len(sel), -1).pow(2).sum(1) for g in gs.values()))
            scale = (C / (norms + 1e-12)).clamp(max=1.0)
            exp_b = q * n
            for k in names:
                gk = (gs[k] * scale.view(-1, *[1] * (gs[k].dim() - 1))).sum(0)
                noise = torch.randn(gk.shape, generator=gen, device=gk.device) * (s_D if ada else sigma) * C
                if mvec is not None:
                    noise = noise * mvec[k]
                # adaclip_norm: step normalised by C/C0, so the adapted clip changes only the bias-noise
                # trade-off, not the effective learning rate (without it, a growing C enlarges every step,
                # the model diverges, per-example norms grow and C runs away: the v1 failure mode)
                step = a.lr_dp * (a.clip / C if (ada and a.adaclip_norm) else 1.0)
                p[k] = p[k] - step * (gk + noise) / exp_b
            if ada:   # geometric quantile update from the noisy centred count (uses the PRE-update C)
                cnt = ((norms <= C).float() - 0.5).sum() + torch.randn((), generator=gen, device=X.device) * s_b
                C = C * math.exp(-a.adaclip_lr * (float(cnt) / exp_b + 0.5 - a.adaclip))
                if a.adaclip_norm:
                    C = min(max(C, 1e-2 * a.clip), 1e2 * a.clip)
        else:
            sel = torch.randint(0, n, (a.batch,), generator=gen, device=X.device)
            out = functional_call(model, p, (X[sel].float(),))
            loss = lossf(out, y[sel])
            gr = torch.autograd.grad(loss, [p[k] for k in names])
            p = {k: (p[k] - a.lr * (g * mvec[k] if mvec is not None else g)).detach().requires_grad_(True)
                 for k, g in zip(names, gr)}
    return flat({k: v.detach() for k, v in p.items()}) - flat(gparams), C


def qsgd(v, levels, bucket, gen):
    """Unbiased stochastic quantisation (Alistarh et al., NeurIPS 2017) with L2-normalised buckets.
    Post-processing of the DP release, so the privacy guarantee is unchanged."""
    n = v.numel(); pad = (-n) % bucket
    vb = F.pad(v, (0, pad)).view(-1, bucket)
    nrm = vb.norm(dim=1, keepdim=True).clamp_min(1e-12)
    r = vb.abs() / nrm * levels
    lo = r.floor()
    qv = lo + (torch.rand(r.shape, generator=gen, device=v.device) < (r - lo)).float()
    return (torch.sign(vb) * qv * nrm / levels).view(-1)[:n]


def qsgd_bits(s):
    """Bits per coordinate (1 sign + magnitude) matched to a retention ratio s at 32-bit values."""
    return int(min(32, max(2, round(32 * s))))

def public_mask(d, s, prev_global_delta, rng_t, explore=0.1):
    """Data-independent mask: top-(1-explore) by |previous GLOBAL update| (public) + random rest."""
    k = max(1, int(round(s * d)))
    m = torch.zeros(d, dtype=torch.bool, device=prev_global_delta.device)
    k_exp = int(round(explore * k)) if prev_global_delta.abs().sum() > 0 else k
    if k - k_exp > 0:
        m[torch.topk(prev_global_delta.abs(), k - k_exp).indices] = True
    free = torch.nonzero(~m).squeeze(1)
    pick = free[torch.randperm(len(free), generator=rng_t, device="cpu").to(free.device)[:k_exp]]
    m[pick] = True
    return m


def _rdp_poisson_gaussian(q, sigma, steps, orders):
    """RDP of the Poisson-subsampled Gaussian mechanism, integer orders (Mironov et al. 2019).
    Exact for integer alpha; dependency-free so Kaggle needs no pip install."""
    from scipy.special import gammaln, logsumexp
    out = []
    for a in orders:
        if q >= 1.0:
            out.append(steps * a / (2 * sigma ** 2)); continue
        i = np.arange(a + 1)
        logc = gammaln(a + 1) - gammaln(i + 1) - gammaln(a - i + 1)
        terms = logc + i * math.log(q) + (a - i) * math.log1p(-q) + (i * i - i) / (2 * sigma ** 2)
        out.append(steps * logsumexp(terms) / (a - 1))
    return np.array(out)


def epsilon(q, sigma, steps, delta, orders=tuple(range(2, 257))):
    o = np.array(orders, dtype=float)
    rdp = _rdp_poisson_gaussian(q, sigma, steps, orders)
    eps = rdp - (math.log(delta) + np.log(o)) / (o - 1) + np.log((o - 1) / o)  # Balle et al. 2020 conversion
    return float(np.nanmin(eps))


def solve_sigma(eps, delta, q, steps):
    """Smallest sigma with epsilon(q, sigma, steps, delta) <= eps (bisection, conservative integer orders)."""
    lo, hi = 0.3, 2.0
    while epsilon(q, hi, steps, delta) > eps:
        hi *= 2
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if epsilon(q, mid, steps, delta) > eps else (lo, mid)
    return hi

# ----------------------------------------------------------------------------------------------
# 6. One FL run
# ----------------------------------------------------------------------------------------------
def run_id(a):
    keys = ["method", "dp", "mask", "eps", "T", "N", "alpha_dir", "part", "tau", "smin", "smax",
            "alpha", "gamma_ref", "clip", "scenario", "s_fixed", "s_budget", "seed", "width", "lr", "lr_dp"]
    s = "_".join(f"{k}={getattr(a, k)}" for k in keys)
    cfg = {k: getattr(a, k) for k in keys}
    # options added later enter the hash ONLY when non-default -> old run IDs unchanged
    ext = {k: getattr(a, k, v) for k, v in EXTRA_DEFAULTS.items() if getattr(a, k, v) != v}
    if ext:
        s += "_" + "_".join(f"{k}={v}" for k, v in sorted(ext.items()))
        cfg.update(ext)
    return hashlib.md5(s.encode()).hexdigest()[:10], cfg


EXTRA_DEFAULTS = dict(dataset="ptbxl", task="single", explore=0.1, adaclip=0.0, adaclip_lr=0.2, adaclip_infl=0.05, qbucket=512,
                      adaclip_norm=0, agg="weighted", arch="cnn", bw_user=10e6, labelset="super")


def fl_run(a, data, log=print):
    rid, cfg = run_id(a)
    os.makedirs(a.results, exist_ok=True)
    os.makedirs(a.ckpt, exist_ok=True)
    out_path = os.path.join(a.results, f"{rid}.json")
    if os.path.exists(out_path) and not a.force:
        log(f"[skip] {rid} already done")
        return json.load(open(out_path))
    dev = data["Xtr"].device
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    gen = torch.Generator(device=dev); gen.manual_seed(a.seed)
    cpu_gen = torch.Generator(); cpu_gen.manual_seed(a.seed + 7)
    rng = np.random.default_rng(a.seed)

    assert data["kind"] == a.dataset, f"--dataset {a.dataset} but the npz holds {data['kind']}"
    if a.dataset == "cinc":   # one federated client per real institution
        assert a.task == "multi", "CinC-2021 is multi-label: use --task multi"
        assert a.N == len(data["site_names"]), f"set --N {len(data['site_names'])} (number of institutions)"
        parts = [np.where(data["site_tr"] == k)[0] for k in range(a.N)]
    else:
        parts = dirichlet_split(data["ytr"].cpu().numpy(), a.N, a.alpha_dir, a.seed)
    nk = np.array([len(p) for p in parts])
    cnt = np.bincount(data["ytr"].cpu().numpy(), minlength=5)
    multi = a.task == "multi"
    if multi and getattr(a, "labelset", "super") == "diag":   # swap in the 44-label targets for this run
        assert "Ydtr" in data, "--labelset diag needs a ptbxl_ml.npz built by the NEW `cgfl.py prep_ml` (with Yd)"
        data = {**data, "Ytr": data["Ydtr"], "Yva": data["Ydva"], "Yte": data["Ydte"]}
    if not multi:
        w_cls = torch.tensor(cnt.sum() / (5 * cnt), dtype=torch.float32, device=dev)
    if multi:   # BCE targets; pos_weight = neg/pos per label plays the role of the class weights
        assert "Ytr" in data, "--task multi needs --labels ptbxl_ml.npz (build it with `cgfl.py prep_ml`)"
        pos = data["Ytr"].sum(0)
        tot = torch.as_tensor(data["site_cls"][data["site_tr"]].sum(0), device=dev).float() \
            if a.dataset == "cinc" else torch.full_like(pos, len(data["Ytr"]))
        w_cls = ((tot - pos) / pos.clamp_min(1)).float()
        if getattr(a, "labelset", "super") == "diag":   # 44 labels, some with <50 positives: cap neg/pos at 50
            w_cls = w_cls.clamp_max(50.0)               # (otherwise single rare labels dominate the clipped gradient)
    Ytr = data["Ytr"] if multi else data["ytr"]
    Yva, Yte = (data["Yva"], data["Yte"]) if multi else (data["yva"], data["yte"])

    n_cls = data["Ytr"].shape[1] if multi else 5
    model = make_model(a, n_cls).to(dev)
    gparams = {k: v.detach().clone() for k, v in model.named_parameters()}
    d = sum(v.numel() for v in gparams.values())

    s_bar = calibrate(a, a.N, a.seed)
    ch = Channel(a.N, a.seed, a.scenario)
    m_sel = max(1, int(round(a.part * a.N)))
    # participation schedule is drawn up-front from a public seed (data-independent): exact accounting
    sched = [rng.choice(a.N, m_sel, replace=False) for _ in range(a.T)]
    part_cnt = np.bincount(np.concatenate(sched), minlength=a.N)

    sigma, q = np.zeros(a.N), np.zeros(a.N)
    if a.dp == "record":
        cache = {}
        for k in range(a.N):
            q[k] = min(a.batch / nk[k], 0.5)
            key = (round(q[k], 5), int(part_cnt[k] * a.tau))
            if key not in cache:
                cache[key] = solve_sigma(a.eps, a.delta, *key)
            sigma[k] = cache[key]

    start, hist, prev_gdelta = 0, [], torch.zeros(d, device=dev)
    clip_k = np.full(a.N, float(a.clip))   # per-site clip norm (changes only with --adaclip)
    ck = os.path.join(a.ckpt, f"{rid}.pt")
    if os.path.exists(ck) and not a.force:
        st = torch.load(ck, map_location=dev, weights_only=False)
        gparams, start, hist, prev_gdelta = st["g"], st["t"], st["hist"], st["prev"]
        clip_k = st.get("clip_k", clip_k)
        torch.set_rng_state(st["cpu_rng"]); gen.set_state(st["gen"]); cpu_gen.set_state(st["cpu_gen"])
        ch.g = st["ch_rng"]; rng = st["rng"]
        log(f"[resume] {rid} from round {start}")

    bpv, bidx = 32, int(math.ceil(math.log2(d)))  # bits per value / per Top-k index (18 for d < 2^18)
    # NOTE: runs made before this fix stored payloads with 32-bit indices; the paper rescales those
    # index-carrying (Top-k) runs by (32 + b_idx) / 64, which is exactly what this line now computes.
    t0 = time.time()
    for t in range(start, a.T):
        snr = ch.snr_db()
        sel = sched[t]
        s_all = retention(a.method, snr, a, rng, s_bar)
        bw_user = a.bw_user
        agg = torch.zeros(d, device=dev)
        cov = torch.zeros(d, device=dev) if a.agg == "coord" else None   # weighted coverage per coordinate
        bits, lat = [], []
        for k in sel:
            Xk, yk = data["Xtr"][parts[k]], Ytr[parts[k]]
            s = float(s_all[k])
            quant = a.method in ("qsgd", "qsgd_cap")
            use_public = a.mask == "public" and a.method != "fedavg" and not quant
            mask = public_mask(d, s, prev_gdelta, cpu_gen, a.explore) if use_public else None
            cm = torch.as_tensor(data["site_cls"][k], dtype=torch.float32, device=dev) if a.dataset == "cinc" else None
            delta, clip_k[k] = local_train(model, gparams, Xk, yk, w_cls, a, mask, sigma[k], q[k], gen, clip_k[k], cm)
            if quant:   # dense DP update, stochastically quantised at the matched bit width
                bq = qsgd_bits(s)
                delta = qsgd(delta, 2 ** (bq - 1) - 1, a.qbucket, gen)
            elif a.method != "fedavg" and not use_public:  # Top-k on the (already DP) update = post-processing
                kk = max(1, int(round(s * d)))
                keep = torch.topk(delta.abs(), kk).indices
                sp = torch.zeros_like(delta); sp[keep] = delta[keep]; delta = sp
            wk = nk[k] / nk[sel].sum()
            agg += wk * delta
            if cov is not None:   # support of the transmitted update (public mask / Top-k set / dense)
                if use_public:
                    cov += wk * mask.float()
                elif a.method != "fedavg" and not quant:
                    cov += wk * (delta != 0).float()
                else:
                    cov += wk
            if quant:
                b = d * bq + math.ceil(d / a.qbucket) * bpv
            else:
                b = s * d * (bpv + (0 if (use_public or a.method == "fedavg") else bidx))
            bits.append(b)
            lat.append(b / capacity_bits(snr[k], bw_user))
        if cov is not None:   # post-processing of the released updates + public weights/masks: DP unchanged
            agg = torch.where(cov > 0, agg / cov.clamp_min(1e-12), torch.zeros_like(agg))
        gnew = flat(gparams) + a.server_lr * agg
        prev_gdelta = agg
        gparams = unflat(gnew.clone(), gparams)
        rec = dict(t=t + 1, mb=float(np.sum(bits) / 8e6), lat_s=float(np.max(lat)),
                   s_mean=float(np.mean(s_all[sel])))
        if (t + 1) % a.eval_every == 0 or t + 1 == a.T:
            for kname, v in gparams.items():
                dict(model.named_parameters())[kname].data.copy_(v)
            rec.update({f"val_{k}": v for k, v in evaluate(model, data["Xva"], Yva, multi=multi,
                                                          ann=data.get("ann_va")).items()})
            if a.adaclip > 0:
                rec["clip_mean"] = float(clip_k.mean())
            log(f"  {rid} r{t+1:4d} valAUC={rec['val_auc']:.3f} bacc={rec['val_bacc']:.3f} "
                f"s={rec['s_mean']:.2f} {time.time()-t0:.0f}s")
        hist.append(rec)
        if (t + 1) % a.ckpt_every == 0:
            torch.save(dict(g=gparams, t=t + 1, hist=hist, prev=prev_gdelta, cpu_rng=torch.get_rng_state(),
                            gen=gen.get_state(), cpu_gen=cpu_gen.get_state(), ch_rng=ch.g, rng=rng,
                            clip_k=clip_k), ck)

    for kname, v in gparams.items():
        dict(model.named_parameters())[kname].data.copy_(v)
    test = evaluate(model, data["Xte"], Yte, multi=multi, ann=data.get("ann_te"))
    if a.dataset == "cinc":   # per-institution test AUC (fairness across real sites)
        sm = lambda k: torch.as_tensor(data["site_te"] == k, device=dev)
        test["site_auc"] = {nm: evaluate(model, data["Xte"][sm(k)], Yte[sm(k)], multi=True,
                                         ann=data["ann_te"][data["site_te"] == k])["auc"]
                            for k, nm in enumerate(data["site_names"])}
        test["site_nk"] = dict(zip(data["site_names"], nk.tolist()))
    maj = float(cnt.max() / cnt.sum())
    res = dict(id=rid, cfg=cfg, d=d, s_bar=s_bar, sigma_mean=float(sigma.mean()), sigma_max=float(sigma.max()),
               test=test, comm_mb=float(sum(h["mb"] for h in hist)),
               wallclock_uplink_s=float(sum(h["lat_s"] for h in hist)), hist=hist,
               sanity_acc_below_majority=bool(not multi and test["acc"] < maj and test["auc"] > .75),
               runtime_s=time.time() - t0)
    json.dump(res, open(out_path, "w"))
    if os.path.exists(ck):
        os.remove(ck)
    log(f"[done] {rid} testAUC={test['auc']:.3f} acc={test['acc']:.3f} bacc={test['bacc']:.3f} "
        f"comm={res['comm_mb']:.1f}MB uplink={res['wallclock_uplink_s']:.2f}s  ({res['runtime_s']/60:.1f} min)")
    if res["sanity_acc_below_majority"]:
        log("  !! accuracy < majority-class rate while AUC is high: check pred_hist (the 13.9% bug pattern)")
    return res

# ----------------------------------------------------------------------------------------------
# 7. Experiment grid: staged, budget-aware. Each entry = overrides of the defaults.
# ----------------------------------------------------------------------------------------------
TUNE_CFGS = [(tau, T) for tau in (2, 5, 10) for T in (100, 200)]   # (tau, T) candidates, eps = 3


def select_best(results, base):
    """Pick (tau, T) by mean FINAL VALIDATION AUC of the capacity map over seeds 1-3 (test set untouched).
    Ties within 0.002 -> fewer local steps, then fewer rounds."""
    import argparse as _ap
    sc = {}
    for tau, T in TUNE_CFGS:
        v = []
        for s in (1, 2, 3):
            rid, _ = run_id(_ap.Namespace(**{**base, **dict(method="capacity", dp="record", mask="public",
                                                         tau=tau, T=T, seed=s)}))
            f = os.path.join(results, f"{rid}.json")
            if os.path.exists(f):
                v.append(json.load(open(f))["hist"][-1]["val_auc"])
        if len(v) == 3:
            sc[(tau, T)] = float(np.mean(v))
    if not sc:
        return None, sc
    top = max(sc.values())
    best = min((c for c in sc if sc[c] >= top - 0.002), key=lambda c: (c[0], c[1]))
    return best, sc


RES_LR, RES_LR_DP = [0.02, 0.05, 0.1], [0.5, 1.0, 2.0, 4.0]   # ResNet learning-rate candidates


def select_lr(results, base):
    """ResNet learning rates: best final VALIDATION AUC of the 1-seed, T=100 tuning runs (test set untouched)."""
    import argparse as _ap
    best = {}
    for dp, key, cands, extra in [("none", "lr", RES_LR, dict(mask="topk")), ("record", "lr_dp", RES_LR_DP, dict(mask="public"))]:
        sc = {}
        for v in cands:
            o = dict(arch="resnet", method="fixed", dp=dp, tau=5, T=100, seed=1, **extra, **{key: v})
            rid, _ = run_id(_ap.Namespace(**{**base, **o}))
            f = os.path.join(results or "", f"{rid}.json")
            if os.path.exists(f):
                sc[v] = json.load(open(f))["hist"][-1]["val_auc"]
        best[key] = (max(sc, key=sc.get) if len(sc) == len(cands) else None, sc)
    return best


def build_grid(stage, results=None, base=None):
    S5, S3 = [1, 2, 3, 4, 5], [1, 2, 3]
    G = []
    if stage == "best":   # tune (tau, T) on VALIDATION AUC (3 seeds); finished runs are reused
        for (tau, T), s in itertools.product(TUNE_CFGS, S3):
            G.append(dict(method="capacity", dp="record", mask="public", tau=tau, T=T, seed=s))
    if stage == "best_confirm":   # all DP methods, 5 seeds, at the selected (tau, T)
        pick = select_best(results, base)[0] if results else None
        tau, T = pick or (5, 100)
        for s in S5:
            for m in ["fixed", "capacity"]:
                G.append(dict(method=m, dp="record", mask="public", tau=tau, T=T, seed=s))
            G.append(dict(method="fedavg", dp="record", tau=tau, T=T, seed=s))
    if stage == "multilabel":   # standard superdiagnostic multi-label task, main comparison
        for s in S5:
            G.append(dict(method="fedavg", dp="none", task="multi", seed=s))
            G.append(dict(method="fedavg", dp="record", task="multi", seed=s))
            for m in ["fixed", "capacity"]:
                G.append(dict(method=m, dp="none", mask="topk", task="multi", seed=s))
                G.append(dict(method=m, dp="record", mask="public", task="multi", seed=s))
    if stage == "diag":   # fine-grained PTB-XL task (44 diagnostic statements), same design as "multilabel"
        for s in S5:
            G.append(dict(method="fedavg", dp="none", task="multi", labelset="diag", seed=s))
            G.append(dict(method="fedavg", dp="record", task="multi", labelset="diag", seed=s))
            for m in ["fixed", "capacity"]:
                G.append(dict(method=m, dp="none", mask="topk", task="multi", labelset="diag", seed=s))
                G.append(dict(method=m, dp="record", mask="public", task="multi", labelset="diag", seed=s))
    if stage == "cinc":   # second dataset, REAL institutions as clients (all take part every round)
        nsite = 6
        try:
            nsite = len(np.load(base["data"])["site_names"]) if base and str(base.get("data", "")).endswith(".npz") else 6
        except Exception:
            pass
        C = dict(dataset="cinc", task="multi", N=nsite, part=1.0)
        for s in S5:
            G.append(dict(method="fedavg", dp="none", seed=s, **C))
            G.append(dict(method="fedavg", dp="record", seed=s, **C))
            for m in ["fixed", "capacity"]:
                G.append(dict(method=m, dp="none", mask="topk", seed=s, **C))
            for m in ["fixed", "random", "capacity", "qsgd_cap"]:
                G.append(dict(method=m, dp="record", mask="public", seed=s, **C))
    if stage == "external":   # external baselines at the default operating point (eps = 3)
        for s in S5:
            G.append(dict(method="fixed", dp="record", mask="public", explore=1.0, seed=s))     # Fed-SMP-style
            G.append(dict(method="capacity", dp="record", mask="public", explore=1.0, seed=s))  # random mask + capacity
            G.append(dict(method="fedavg", dp="record", adaclip=0.5, adaclip_norm=1, seed=s))   # DP-FedAvg + adaptive clip
            G.append(dict(method="capacity", dp="record", mask="public", adaclip=0.5, adaclip_norm=1, seed=s))  # CGFL + adaptive clip
            G.append(dict(method="qsgd", dp="record", seed=s))                                  # QSGD, matched bits
            G.append(dict(method="qsgd_cap", dp="record", seed=s))                              # QSGD, capacity bits
        for s in S3:
            G.append(dict(method="capacity", dp="record", mask="public", explore=0.3, seed=s))  # mask exploration share
    if stage == "agg":   # per-coordinate normalised aggregation (tests the coordinate-dilution hypothesis)
        for s in S5:
            for m in ["fixed", "capacity"]:
                G.append(dict(method=m, dp="record", mask="public", agg="coord", seed=s))           # default point
                G.append(dict(method=m, dp="record", mask="public", agg="coord", tau=5, seed=s))    # tuned point
            G.append(dict(method="capacity", dp="none", mask="topk", agg="coord", seed=s))         # non-private
    if stage == "agg_cinc":   # normalised aggregation on CinC-2021 (real institutions as clients); needs --data cinc2021.npz
        for s, m in itertools.product(S5, ["fixed", "capacity"]):
            G.append(dict(method=m, dp="record", mask="public", agg="coord", seed=s,
                          dataset="cinc", task="multi", N=6, part=1.0))
    if stage == "resnet_tune":   # ResNet step 1: learning rates for the larger model (1 seed, T=100, tuned tau=5)
        for lr in RES_LR:
            G.append(dict(arch="resnet", method="fixed", dp="none", mask="topk", lr=lr, tau=5, T=100, seed=1))
        for lr in RES_LR_DP:
            G.append(dict(arch="resnet", method="fixed", dp="record", mask="public", lr_dp=lr, tau=5, T=100, seed=1))
    if stage == "resnet":   # ResNet step 2: main comparison with the ~2.4 M-parameter ResNet (tuned tau=5, T=200)
        sel = select_lr(results, base) if results else {}
        lr = (sel.get("lr") or (None,))[0]
        lr_dp = (sel.get("lr_dp") or (None,))[0]
        if lr is None or lr_dp is None:
            print("!! resnet: run --stage resnet_tune first (learning rates not selected yet)")
            return G
        R = dict(arch="resnet", tau=5, lr=lr, lr_dp=lr_dp)
        for s in S5:
            G.append(dict(method="fedavg", dp="record", seed=s, **R))
            for m in ["fixed", "capacity"]:
                G.append(dict(method=m, dp="record", mask="public", seed=s, **R))
            G.append(dict(method="capacity", dp="record", mask="public", agg="coord", seed=s, **R))
        for s in S3:
            G.append(dict(method="fedavg", dp="none", seed=s, **R))
            G.append(dict(method="capacity", dp="none", mask="topk", seed=s, **R))
    if stage == "tune":  # run FIRST: pick --lr (non-DP) and --lr_dp (DP) on 1 seed, short horizon
        for lr in [0.02, 0.05, 0.1]:
            G.append(dict(method="fixed", dp="none", mask="topk", lr=lr, T=100, seed=1))
        for lr in [0.5, 1.0, 2.0, 4.0]:
            G.append(dict(method="fixed", dp="record", mask="public", lr_dp=lr, T=100, seed=1))
    if stage in ("main", "all"):  # Table: all methods, 5 seeds, eps=3, plus non-private references
        for s in S5:
            G.append(dict(method="fedavg", dp="none", seed=s))
            for m in ["fixed", "random", "inverse", "logistic", "capacity"]:
                G.append(dict(method=m, dp="none", mask="topk", seed=s))
                G.append(dict(method=m, dp="record", mask="public", seed=s))
            G.append(dict(method="capacity", dp="record", mask="topk", seed=s))   # sparsify-after-noise ablation
            G.append(dict(method="fedavg", dp="record", seed=s))                  # DP-FedAvg, no compression
    if stage in ("eps_T", "all"):  # disentangle DP noise vs too-few rounds
        for s, e, T, m in itertools.product(S3, [1, 3, 8], [100, 200, 400], ["fixed", "capacity"]):
            G.append(dict(method=m, dp="record", mask="public", eps=e, T=T, seed=s))
    if stage in ("ablation", "all"):  # one-at-a-time sensitivity around the default
        oat = dict(smin=[0.05, 0.2], smax=[0.6, 0.95], clip=[0.5, 2.0], tau=[5, 20], part=[0.3, 1.0],
                   alpha=[0.15, 1.0], gamma_ref=[12.0, 21.0])
        for s, (k, vals) in itertools.product(S3, oat.items()):
            for v in vals:
                G.append(dict(method="capacity" if k not in ("alpha", "gamma_ref") else "logistic",
                              dp="record", mask="public", seed=s, **{k: v}))
    if stage in ("scale", "all"):  # scalability + heterogeneity
        for s, m in itertools.product(S3, ["fixed", "capacity"]):
            for N in [50, 100]:
                G.append(dict(method=m, dp="record", mask="public", N=N, seed=s))
            for ad in [0.1, 1.0]:
                G.append(dict(method=m, dp="record", mask="public", alpha_dir=ad, seed=s))
    if stage in ("channel", "all"):  # channel realism
        for s, m, sc in itertools.product(S3, ["fixed", "capacity"], ["blockage", "interf", "blockage_interf"]):
            G.append(dict(method=m, dp="record", mask="public", scenario=sc, seed=s))
    return G


def defaults():
    p = argparse.ArgumentParser(add_help=False)
    A = p.add_argument
    A("--method", default="capacity"); A("--dp", default="record", choices=["none", "record"])
    A("--mask", default="public", choices=["public", "topk"])
    A("--eps", type=float, default=3.0); A("--delta", type=float, default=1e-5)
    A("--T", type=int, default=200); A("--N", type=int, default=20); A("--alpha_dir", type=float, default=0.3)
    A("--part", type=float, default=0.5); A("--tau", type=int, default=10); A("--batch", type=int, default=64)
    A("--max_phys_batch", type=int, default=256)
    A("--lr", type=float, default=0.05, help="local SGD lr, non-DP runs")
    A("--lr_dp", type=float, default=0.5, help="local DP-SGD lr (clipped grads are smaller -> larger lr)"); A("--server_lr", type=float, default=1.0)
    A("--clip", type=float, default=1.0); A("--width", type=int, default=32)
    A("--smin", type=float, default=0.1); A("--smax", type=float, default=0.9)
    A("--alpha", type=float, default=0.4); A("--gamma_ref", type=float, default=16.5)
    A("--s_fixed", type=float, default=-1, help="fixed ratio; <=0 -> matched to s_bar")
    A("--s_budget", type=float, default=-1, help="override matched mean retention")
    A("--explore", type=float, default=0.1); A("--bw_user", type=float, default=10e6)
    A("--scenario", default="los", choices=["los", "blockage", "interf", "blockage_interf"])
    A("--seed", type=int, default=1); A("--eval_every", type=int, default=10); A("--ckpt_every", type=int, default=20)
    A("--data", default="ptbxl100.npz"); A("--synthetic", action="store_true")
    A("--results", default="results"); A("--ckpt", default="ckpt"); A("--device", default=None)
    A("--force", action="store_true")
    A("--task", default="single", choices=["single", "multi"]); A("--labels", default=None)
    A("--labelset", default="super", choices=["super", "diag"],
      help="multi-label PTB-XL targets: 5 diagnostic superclasses or the 44 diagnostic statements")
    A("--adaclip", type=float, default=0.0, help="target unclipped quantile (0 = fixed clip)")
    A("--adaclip_lr", type=float, default=0.2); A("--adaclip_infl", type=float, default=0.05)
    A("--qbucket", type=int, default=512)
    A("--adaclip_norm", type=int, default=0, help="1 = normalise the DP-SGD step by C/C0 (stable adaptive clipping)")
    A("--dataset", default="ptbxl", choices=["ptbxl", "cinc"])
    A("--agg", default="weighted", choices=["weighted", "coord"],
      help="coord = per-coordinate normalised aggregation: each coordinate is averaged only over the sites that sent it")
    A("--arch", default="cnn", choices=["cnn", "resnet"])
    return p


def main():
    ap = argparse.ArgumentParser(parents=[defaults()])
    ap.add_argument("cmd", choices=["prep", "prep_ml", "prep_cinc", "run", "grid", "bench", "summarize", "count", "select"])
    ap.add_argument("--stage", default="main"); ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", default="/kaggle/working/ptbxl100.npz"); ap.add_argument("--root", default=None)
    ap.add_argument("--budget_h", type=float, default=30.0)
    ap.add_argument("--target_auc", type=float, default=-1)
    a = ap.parse_args()
    if a.cmd == "prep":
        return prep_ptbxl(a.out, a.root)
    if a.cmd == "prep_ml":
        return prep_ml(a.out, a.root)
    if a.cmd == "prep_cinc":
        return prep_cinc(a.out, a.root)
    if a.cmd == "select" and a.stage.startswith("resnet"):
        for key, (b, sc) in select_lr(a.results, vars(a)).items():
            print(f"  {key}: candidates {sc} -> selected {b}")
        return
    if a.cmd == "select":
        best, sc = select_best(a.results, vars(a))
        for c, v in sorted(sc.items()):
            print(f"  tau={c[0]:2d} T={c[1]:3d}  mean val AUC (capacity, seeds 1-3) = {v:.4f}")
        print("selected (tau, T) =", best); return
    if a.cmd == "summarize":
        return summarize(a.results, a.stage, a.target_auc, base=vars(a))
    if a.cmd == "count":
        g = build_grid(a.stage, a.results, vars(a)); print(f"{a.stage}: {len(g)} runs"); return
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data(a.data, a.synthetic, dev, a.labels)
    if a.cmd == "run":
        return fl_run(a, data)
    if a.cmd == "bench":  # time a few rounds of the most expensive config, project the full grid
        b = argparse.Namespace(**vars(a)); b.T, b.eval_every, b.force = 5, 5, True
        b.results, b.ckpt = "/tmp/bench_res", "/tmp/bench_ck"
        r = fl_run(b, data, log=lambda *x: None)
        per_round = r["runtime_s"] / 5
        print(f"{per_round:.2f} s/round (DP, N={a.N}, part={a.part}, tau={a.tau}) on {dev}")
        tot = 0
        done = set(os.path.basename(f)[:-5] for f in glob.glob(os.path.join(a.results, "*.json")))
        for st in ["main", "eps_T", "ablation", "scale", "channel", "best", "best_confirm", "multilabel", "external", "cinc",
                   "agg", "agg_cinc", "resnet_tune", "resnet", "diag"]:
            g = [o for o in build_grid(st, a.results, vars(a))
                 if run_id(argparse.Namespace(**{**vars(a), **o}))[0] not in done]   # only runs still to do
            h = sum(per_round * o.get("T", a.T) * (o.get("N", a.N) / a.N) *
                    (o.get("tau", a.tau) / a.tau) * (o.get("part", a.part) / a.part) *
                    (0.3 if o.get("dp", a.dp) == "none" else 1) for o in g) / 3600
            tot += h
            print(f"  {st:12s} {len(g):4d} runs to do  ~{h:5.1f} GPU-h  (~{h/2:5.1f} quota-h on T4x2)")
        print(f"  TOTAL ~{tot:.1f} GPU-h  -> ~{tot/2/a.budget_h:.1f} weeks of a {a.budget_h:.0f} h/week quota on T4x2")
        return
    # grid
    i, n = map(int, a.shard.split("/"))
    grid = build_grid(a.stage, a.results, vars(a))[i::n]
    base = vars(a).copy()
    print(f"stage={a.stage} shard {i}/{n}: {len(grid)} runs on {dev}", flush=True)
    for j, o in enumerate(grid):
        cfg = argparse.Namespace(**{**base, **o})
        print(f"--- [{j+1}/{len(grid)}] {o}", flush=True)
        fl_run(cfg, data, log=lambda *x: print(*x, flush=True))


def grid_ids(stage, base=None):
    base = base or vars(defaults().parse_args([]))
    out = {}
    for o in build_grid(stage, base.get("results"), base):
        rid, _ = run_id(argparse.Namespace(**{**base, **o}))
        out[rid] = o
    return out


def summarize(results, stage, target_auc, base=None):
    """Stage-aware summary: mean +/- std, 95% CI, uplink time-to-target, paired Wilcoxon (same seeds)."""
    import pandas as pd
    from scipy import stats
    base0 = base or vars(defaults().parse_args([]))
    ids = grid_ids(stage, base0)
    rows = []
    for rid, o in ids.items():
        f = os.path.join(results, f"{rid}.json")
        if not os.path.exists(f):
            continue
        r = json.load(open(f))
        o = {"method": base0["method"], "dp": base0["dp"], "mask": base0["mask"], **o}
        if o["method"] == "fedavg":
            o["mask"] = "-"
        rows.append(dict(o, auc=r["test"]["auc"], bacc=r["test"]["bacc"], f1=r["test"].get("f1", np.nan),
                         acc=r["test"]["acc"], comm_mb=r["comm_mb"], uplink_s=r["wallclock_uplink_s"],
                         hist=r["hist"]))
    print(f"stage={stage}: {len(rows)}/{len(ids)} runs finished")
    if not rows:
        return
    df = pd.DataFrame(rows)
    if target_auc <= 0:   # default target: 95% of the non-private FedAvg validation AUC
        ref = [h[-1]["val_auc"] for h, m, d in zip(df["hist"], df["method"], df["dp"]) if m == "fedavg" and d == "none"]
        target_auc = 0.95 * float(np.mean(ref)) if ref else 0.80

    def t2t(hist):
        cum = 0.0
        for h in hist:
            cum += h["lat_s"]
            if h.get("val_auc", -1) >= target_auc:
                return cum
        return np.nan
    df["t2t_s"] = df["hist"].apply(t2t)
    keys = [c for c in df.columns if c not in ("seed", "hist", "auc", "bacc", "f1", "acc", "comm_mb", "uplink_s", "t2t_s")]
    df[keys] = df[keys].astype(str)          # NaN -> "nan" so one-at-a-time ablation rows are not dropped
    g = df.groupby(keys, dropna=False)
    agg = g.agg(n=("auc", "size"), auc=("auc", "mean"), auc_sd=("auc", "std"), bacc=("bacc", "mean"),
                f1=("f1", "mean"), comm_mb=("comm_mb", "mean"), uplink_s=("uplink_s", "mean"),
                t2t_s=("t2t_s", "mean"), reached=("t2t_s", lambda x: int(x.notna().sum())))
    agg["auc_ci95"] = g["auc"].apply(lambda x: stats.t.ppf(.975, len(x) - 1) * x.std() / math.sqrt(len(x))
                                     if len(x) > 1 else np.nan)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(f"time-to-target uses val AUC >= {target_auc:.3f}")
    print(agg.round(4).to_string())
    if stage == "main":   # paired tests vs the proposed method, DP + public-mask setting
        sub = df[(df.dp == "record") & (df["mask"] == "public")]
        base = sub[sub.method == "capacity"].set_index("seed")
        for m in sorted(set(sub.method) - {"capacity"}):
            o = sub[sub.method == m].set_index("seed")
            c = base.index.intersection(o.index)
            if len(c) >= 5:
                for met in ["auc", "uplink_s"]:
                    p = stats.wilcoxon(base.loc[c, met], o.loc[c, met]).pvalue
                    print(f"capacity vs {m:9s} {met:9s} Wilcoxon p={p:.3f} (n={len(c)}; n=5 -> min p=0.0625)")
    df.drop(columns="hist").to_csv(os.path.join(results, f"runs_{stage}.csv"), index=False)
    agg.to_csv(os.path.join(results, f"table_{stage}.csv"))


if __name__ == "__main__":
    main()
