#!/usr/bin/env python3
"""Re-derive the wireless baselines (Table 2) and the gradient-proxy statistics of Section VI from the code.

Usage: python wireless_verify.py <cgfl.py dir> <results dir> <ptbxl_ml.npz or ptbxl100.npz> [out.json]

No GPU and no training: for each of the 53 seed-matched fixed/capacity pairs it regenerates, with the run's
own seeds, the channel realisations, the participation schedule and the retention ratios, then
  1. checks that the regenerated uplink time equals the logged one (validates the regeneration);
  2. computes latency-equalising bandwidth allocation (BWA) for both policies and deadline dropping;
  3. computes the data-only proxy: pi-weighted covariance between each hospital's realised mean retention
     and the total-variation distance of its label distribution from the global one, and the within-round
     coefficient of variation of the retention ratios; then their Spearman correlation with the AUC cost.
"""
import sys, os, json, argparse
import numpy as np
from scipy import stats

CODE, RES, DATA = sys.argv[1:4]
OUT = sys.argv[4] if len(sys.argv) > 4 else None
sys.path.insert(0, CODE)
import cgfl  # noqa: E402

BASE = vars(cgfl.defaults().parse_args(["--lr_dp", "0.5"]))
D = np.load(DATA)
y_all, fold = D["y"], D["fold"]
ytr = y_all[fold <= 8]
DP = dict(dp="record", mask="public")


def load(o):
    rid, _ = cgfl.run_id(argparse.Namespace(**{**BASE, **o}))
    return json.load(open(os.path.join(RES, rid + ".json")))


def regenerate(o):
    """Channel SNRs, schedule, retention ratios and bits per round, exactly as fl_run draws them."""
    a = argparse.Namespace(**{**BASE, **o})
    r = load(o)
    d = r["d"]
    s_bar = cgfl.calibrate(a, a.N, a.seed)
    ch = cgfl.Channel(a.N, a.seed, a.scenario)
    rng = np.random.default_rng(a.seed)
    m_sel = max(1, int(round(a.part * a.N)))
    sched = [rng.choice(a.N, m_sel, replace=False) for _ in range(a.T)]
    rounds = []
    for t in range(a.T):
        snr = ch.snr_db()
        s_all = cgfl.retention(a.method, snr, a, rng, s_bar)
        sel = sched[t]
        bits = s_all[sel] * d * 32.0
        rounds.append(dict(sel=sel, snr=snr[sel], s=s_all[sel], bits=bits, s_all=s_all))
    return a, r, rounds


def lat_equal(rd, W):
    return rd["bits"] / cgfl.capacity_bits(rd["snr"], W)


def lat_bwa(rd, W):
    """Min-max latency when the scheduled sites share m*W Hz; rate on band b = b log2(1 + g W / b)."""
    g = 10 ** (rd["snr"] / 10)
    bits = rd["bits"]; m = len(bits)
    total = m * W

    def band_needed(L):
        need = bits / L
        if np.any(need >= g * W / np.log(2) * (1 - 1e-12)):
            return np.inf
        lo, hi = np.full(m, 1e-3), np.full(m, W)
        while np.any(hi * np.log2(1 + g * W / hi) < need):     # grow until enough
            grow = hi * np.log2(1 + g * W / hi) < need
            hi[grow] *= 2
            if hi.max() > 1e15:
                return np.inf
        for _ in range(80):
            mid = (lo + hi) / 2
            ok = mid * np.log2(1 + g * W / mid) >= need
            hi = np.where(ok, mid, hi); lo = np.where(ok, lo, mid)
        return hi.sum()

    L_hi = lat_equal(rd, W).max()            # equal bands are feasible
    L_lo = (bits / (g * W / np.log(2))).max()
    for _ in range(80):
        L = np.sqrt(L_lo * L_hi)
        if band_needed(L) <= total:
            L_hi = L
        else:
            L_lo = L
    return L_hi


groups = [("Main (N=20, LoS, 5 seeds)", [({}, s) for s in range(1, 6)])]
los = [({}, s) for s in range(1, 6)] + [(dict(eps=e, T=T), s) for e in [1, 3, 8] for T in [100, 200, 400] for s in [1, 2, 3]] + \
      [(dict(alpha_dir=al), s) for al in [0.1, 1.0] for s in [1, 2, 3]]
groups += [("All LoS, N=20", los), ("N=50", [(dict(N=50), s) for s in [1, 2, 3]]), ("N=100", [(dict(N=100), s) for s in [1, 2, 3]]),
           ("Blockage", [(dict(scenario="blockage"), s) for s in [1, 2, 3]]),
           ("Interference", [(dict(scenario="interf"), s) for s in [1, 2, 3]]),
           ("Blockage + interference", [(dict(scenario="blockage_interf"), s) for s in [1, 2, 3]])]
all53 = los + groups[2][1] + groups[3][1] + groups[4][1] + groups[5][1] + groups[6][1]

cache, maxdev = {}, 0.0


def pair(o, s):
    key = (json.dumps(o, sort_keys=True), s)
    if key in cache:
        return cache[key]
    out = {}
    for m in ["fixed", "capacity"]:
        a, r, rounds = regenerate(dict(method=m, **DP, **o, seed=s))
        W = a.bw_user
        eq = np.array([lat_equal(rd, W).max() for rd in rounds])
        global maxdev
        maxdev = max(maxdev, abs(eq.sum() - r["wallclock_uplink_s"]) / r["wallclock_uplink_s"])
        out[m] = dict(a=a, r=r, rounds=rounds, eq=eq, bwa=np.array([lat_bwa(rd, W) for rd in rounds]))
    # deadline dropping: fixed ratio on equal bands, deadline = CGFL's round time in the same round
    drops = tot = 0
    for rf, lc in zip(out["fixed"]["rounds"], out["capacity"]["eq"]):
        lf = lat_equal(rf, out["fixed"]["a"].bw_user)
        drops += int((lf > lc * (1 + 1e-9)).sum()); tot += len(lf)
    out["drop"] = (drops, tot)
    cache[key] = out
    return out


rows = []
print(f"{'Setting':28s} {'n':>3s} {'Fixed':>9s} {'Fix+BWA':>9s} {'CGFL':>9s} {'CGFL+BWA':>9s} {'Drop':>5s}")
for name, items in groups:
    P = [pair(o, s) for o, s in items]
    f = np.mean([p["fixed"]["eq"].sum() for p in P]); fb = np.mean([p["fixed"]["bwa"].sum() for p in P])
    c = np.mean([p["capacity"]["eq"].sum() for p in P]); cb = np.mean([p["capacity"]["bwa"].sum() for p in P])
    dr = sum(p["drop"][0] for p in P) / sum(p["drop"][1] for p in P)
    rows.append(dict(setting=name, n=len(P), fixed=f, fixed_bwa=fb, cgfl=c, cgfl_bwa=cb, drop=dr))
    print(f"{name:28s} {len(P):3d} {f:9.1f} {fb:9.1f} {c:9.1f} {cb:9.1f} {100*dr:4.0f}%")
print(f"\nregenerated vs. logged uplink time: max relative deviation {maxdev:.2e}")

P = [pair(o, s) for o, s in all53]
bwa_gain = np.array([1 - p["fixed"]["bwa"].sum() / p["fixed"]["eq"].sum() for p in P])
vs_bwa = np.array([1 - p["capacity"]["eq"].sum() / p["fixed"]["bwa"].sum() for p in P])
comb = np.array([1 - p["capacity"]["bwa"].sum() / p["capacity"]["eq"].sum() for p in P])
print(f"BWA shortens the fixed ratio by at most {100*bwa_gain.max():.1f}% (mean {100*bwa_gain.mean():.1f}%)")
print(f"CGFL faster than Fixed+BWA in {(vs_bwa > 0).sum()}/{len(P)} runs, by {100*vs_bwa.mean():.1f}% on average")
print(f"CGFL+BWA vs CGFL: {100*comb.mean():.1f}% on average")

# ---------------- data-only proxy (Section VI)
cov_proxy, cv, dauc, cfgkey = [], [], [], []
for (o, s), p in zip(all53, P):
    a = p["capacity"]["a"]
    parts = cgfl.dirichlet_split(ytr, a.N, a.alpha_dir, a.seed)
    nk = np.array([len(q) for q in parts]); pi = nk / nk.sum()
    glob = np.bincount(ytr, minlength=5) / len(ytr)
    Dk = np.array([0.5 * np.abs(np.bincount(ytr[q], minlength=5) / len(q) - glob).sum() for q in parts])
    S = np.zeros(a.N); C = np.zeros(a.N)
    for rd in p["capacity"]["rounds"]:
        S[rd["sel"]] += rd["s"]; C[rd["sel"]] += 1
    sk = np.where(C > 0, S / np.maximum(C, 1), np.nan)
    ok = ~np.isnan(sk)
    w = pi[ok] / pi[ok].sum()
    cov_proxy.append(np.sum(w * (sk[ok] - np.sum(w * sk[ok])) * (Dk[ok] - np.sum(w * Dk[ok]))))
    cv.append(np.mean([rd["s"].std() / rd["s"].mean() for rd in p["capacity"]["rounds"]]))
    dauc.append(p["capacity"]["r"]["test"]["auc"] - p["fixed"]["r"]["test"]["auc"])
    cfgkey.append(json.dumps(o, sort_keys=True))
cov_proxy, cv, dauc = map(np.array, (cov_proxy, cv, dauc))
r1 = stats.spearmanr(cov_proxy, dauc)
keys = sorted(set(cfgkey))
agg = lambda v: np.array([np.mean([x for x, k in zip(v, cfgkey) if k == kk]) for kk in keys])
r2 = stats.spearmanr(agg(cov_proxy), agg(dauc))
r3 = stats.spearmanr(cv, dauc)
print(f"\nproxy (cov of s_k and D_k) vs AUC cost: rho={r1.statistic:.2f}, p={r1.pvalue:.2f} over {len(dauc)} pairs; "
      f"rho={r2.statistic:.2f}, p={r2.pvalue:.2f} over {len(keys)} configurations")
print(f"covariance positive in {(cov_proxy > 0).mean()*100:.0f}% of runs")
print(f"dispersion (mean within-round CV) vs AUC cost: rho={r3.statistic:.2f}, p={r3.pvalue:.4f}")
if OUT:
    json.dump(dict(table=rows, max_dev=maxdev, bwa_gain_max=bwa_gain.max(), vs_bwa_mean=vs_bwa.mean(),
                   vs_bwa_all=int((vs_bwa > 0).sum()), comb_mean=comb.mean(),
                   proxy=dict(rho53=r1.statistic, p53=r1.pvalue, rho_cfg=r2.statistic, p_cfg=r2.pvalue,
                              n_cfg=len(keys), pos_share=float((cov_proxy > 0).mean())),
                   dispersion=dict(rho=r3.statistic, p=r3.pvalue)), open(OUT, "w"), indent=1, default=float)
