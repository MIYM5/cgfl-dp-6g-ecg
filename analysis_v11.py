#!/usr/bin/env python3
"""Statistics of the v1.1 experiments, computed from results/*.json (and *_probs.npz where needed).

Usage: python analysis_v11.py <cgfl.py dir> <results dir> <ptbxl_ml.npz> [out.json]

Prints and writes (out.json) every number reported for the v1.1 experiments:
  protocol   20-seed comparisons (PTB-XL default / tuned / 44 labels, CinC-2021 patient-wise split):
             mean +- sd, paired differences with 95% CIs, one-sided non-inferiority tests (margin 0.01 AUC,
             paired t-test with a Wilcoxon check, Holm correction over the four settings), uplink saving
  tune       candidates and the values selected on validation AUC
  remedies   screening of the remedies for the accuracy cost (3 seeds) + the fixed-ratio momentum control
  baselines  Wei et al. 2024 and Cheng et al. 2022 adaptations vs. fixed ratio / capacity / noise-aware
  csi        imperfect CSI with HARQ: AUC, uplink time, retransmissions and dropped updates
  maskq      gradient energy kept by the public mask vs. a random mask
  health     44-label F1 at 0.5 vs. validation-tuned thresholds, rare-label AUC, per-hospital AUC (PTB-XL)
  e2e        end-to-end time per run (computation + uplink + downlink broadcast) and bandwidth sweep
"""
import sys, os, json, argparse, math
import numpy as np
from scipy import stats

CODE, RES, LAB = sys.argv[1:4]
OUT = sys.argv[4] if len(sys.argv) > 4 else None
sys.path.insert(0, CODE)
sys.argv = sys.argv[:1]
import cgfl  # noqa: E402

BASE = vars(cgfl.defaults().parse_args(["--lr_dp", "0.5"]))
DP = dict(dp="record", mask="public")
SET = cgfl.SETTINGS
DELTA = 0.01
out = {}


def get(o):
    f = os.path.join(RES, cgfl.run_id(argparse.Namespace(**{**BASE, **o}))[0] + ".json")
    return json.load(open(f)) if os.path.exists(f) else None


def runs(o, seeds):
    return [r for r in (get({**o, "seed": s}) for s in seeds) if r]


def ms(R, key=lambda r: r["test"]["auc"]):
    v = np.array([key(r) for r in R])
    return dict(n=len(v), mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0)


def paired(a_o, b_o, seeds):
    P = [(get({**a_o, "seed": s}), get({**b_o, "seed": s})) for s in seeds]
    P = [(x, y) for x, y in P if x and y]
    d = np.array([x["test"]["auc"] - y["test"]["auc"] for x, y in P])
    n = len(d); m = d.mean(); se = d.std(ddof=1) / math.sqrt(n); t = stats.t.ppf(0.975, n - 1)
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(d, n).mean() for _ in range(10000)])
    up = 1 - np.mean([x["wallclock_uplink_s"] for x, _ in P]) / np.mean([y["wallclock_uplink_s"] for _, y in P])
    return dict(n=n, diff=float(m), ci_t=[float(m - t * se), float(m + t * se)],
                ci_boot=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                p_ni_t=float(stats.t.sf((m + DELTA) / se, n - 1)),
                p_ni_w=float(stats.wilcoxon(d + DELTA, alternative="greater").pvalue),
                p_two=float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0,
                uplink_saving=float(up))


def holm(ps):
    order = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for i, j in enumerate(order):
        run = max(run, (len(ps) - i) * ps[j]); adj[j] = min(1.0, run)
    return adj.tolist()


pick = cgfl.v11_select(RES, BASE)
NA = pick["noiseaware"]
out["tune"] = {fam: dict(cands=[{k: v for k, v in o.items() if k not in ("dp", "mask")} for o in c],
                         val=[(None if None in (v := [get({**o, "seed": s}) for s in (1, 2)]) else
                               float(np.mean([r["hist"][-1]["val_auc"] for r in v]))) for o in c],
                         pick={k: v for k, v in pick[fam].items() if k not in ("dp", "mask")})
               for fam, c in cgfl.REV_TUNE.items()}

# ---------------------------------------------------------------- 20-seed protocol
S20 = range(1, 21)
SETTINGS20 = {"PTB-XL default": {}, "PTB-XL tuned": dict(tau=5), "PTB-XL 44 labels": dict(task="multi", labelset="diag"),
              "CinC-2021": cgfl.CINC}
prot = {}
for name, st in SETTINGS20.items():
    M = {"DP-FedAvg": dict(method="fedavg", dp="record", **st), "Fixed ratio": dict(method="fixed", **DP, **st),
         "Capacity map": dict(method="capacity", **DP, **st), "Noise-aware": {**NA, **st}}
    row = {m: {**ms(runs(o, S20)), "uplink": ms(runs(o, S20), key=lambda r: r["wallclock_uplink_s"]),
               "mb": ms(runs(o, S20), key=lambda r: r["comm_mb"])} for m, o in M.items()}
    row["cap_vs_fixed"] = paired(M["Capacity map"], M["Fixed ratio"], S20)
    row["na_vs_fixed"] = paired(M["Noise-aware"], M["Fixed ratio"], S20)
    row["na_vs_cap"] = paired(M["Noise-aware"], M["Capacity map"], S20)
    row["na_vs_fedavg"] = paired(M["Noise-aware"], M["DP-FedAvg"], S20)
    prot[name] = row
for key in ("cap_vs_fixed", "na_vs_fixed"):
    adj = holm([prot[n][key]["p_ni_t"] for n in SETTINGS20])
    for n, p in zip(SETTINGS20, adj):
        prot[n][key]["p_ni_holm"] = p
out["protocol"] = prot

# ---------------------------------------------------------------- remedies (screening, 3 seeds)
S3 = (1, 2, 3)
rem = {}
for st in ("default", "harsh", "N100"):
    cap = runs(dict(method="capacity", **DP, **SET[st]), S3)
    rem[st] = {"Fixed ratio": ms(runs(dict(method="fixed", **DP, **SET[st]), S3)),
               "Fixed ratio + momentum": ms(runs({**pick["mom"], "method": "fixed", **SET[st]}, S3)),
               "Capacity map": ms(cap)}
    for fam, lab in (("damped", "Damped normalisation"), ("mom", "Capacity + momentum"),
                     ("sizefloor", "Size-aware floor"), ("noiseaware", "Noise-aware")):
        R = runs({**pick[fam], **SET[st]}, S3)
        rem[st][lab] = {**ms(R), "gain": float(np.mean([a["test"]["auc"] - b["test"]["auc"] for a, b in zip(R, cap)]))}
out["remedies"] = rem
out["remedy_selected"] = cgfl.v11_best_variant(RES, BASE, pick)

# ---------------------------------------------------------------- new baselines (5 seeds)
S5 = range(1, 6)
bl = {}
for st in ("default", "tuned", "harsh"):
    bl[st] = {}
    for lab, o in (("Fixed ratio", dict(method="fixed", **DP)), ("Capacity map", dict(method="capacity", **DP)),
                   ("Noise-aware", NA), ("Wei et al.", pick["wei"]), ("Cheng et al.", pick["cheng"])):
        R = runs({**o, **SET[st]}, S5)
        bl[st][lab] = {**ms(R), "uplink": ms(R, key=lambda r: r["wallclock_uplink_s"]),
                       "mb": ms(R, key=lambda r: r["comm_mb"])}
bl["CinC-2021"] = {}
for lab, o in (("Fixed ratio", dict(method="fixed", **DP)), ("Capacity map", dict(method="capacity", **DP)),
               ("Noise-aware", NA), ("Wei et al.", pick["wei"]), ("Cheng et al.", pick["cheng"])):
    R = runs({**o, **cgfl.CINC}, S5)
    bl["CinC-2021"][lab] = {**ms(R), "uplink": ms(R, key=lambda r: r["wallclock_uplink_s"]),
                            "mb": ms(R, key=lambda r: r["comm_mb"])}
out["baselines"] = bl

# ---------------------------------------------------------------- imperfect CSI
csi = {}
for e in (0.0, 0.05, 0.1):
    for m in ("fixed", "capacity"):
        o = dict(method=m, seed=1, **DP) if e == 0 else dict(method=m, csi_err=e, csi_backoff_db=1.0, **DP)
        R = runs({k: v for k, v in o.items() if k != "seed"}, S5)
        csi[f"{e}/{m}"] = {**ms(R), "uplink": ms(R, key=lambda r: r["wallclock_uplink_s"]),
                           "retx": float(np.mean([sum(h.get("retx", 0) for h in r["hist"]) for r in R])),
                           "dropped": float(np.mean([sum(h.get("dropped", 0) for h in r["hist"]) for r in R])),
                           "sent": float(np.mean([r["cfg"]["T"] * max(1, int(round(r["cfg"]["part"] * r["cfg"]["N"])))
                                                  for r in R]))}
out["csi"] = csi

# ---------------------------------------------------------------- mask quality
mq = {}
for st in ("default", "tuned"):
    R = runs(dict(method="capacity", mq_every=10, **DP, **SET[st]), S3)
    for s_ in ("0.05", "0.1", "0.2", "0.28", "0.4", "0.6", "0.9"):
        pub = [np.mean([h["mq"][s_][0] for h in r["hist"] if "mq" in h]) for r in R]
        rnd = [np.mean([h["mq"][s_][1] for h in r["hist"] if "mq" in h]) for r in R]
        mq.setdefault(st, {})[s_] = dict(public=float(np.mean(pub)), random=float(np.mean(rnd)))
out["maskq"] = mq

# ---------------------------------------------------------------- healthcare
L = np.load(LAB); tr = L["fold"] <= 8
RARE = np.where(L["Yd"][tr].sum(0) < 100)[0]
hl = {"n_rare": int(len(RARE))}
for lab, o in (("DP-FedAvg", dict(method="fedavg", dp="record")), ("Fixed ratio", dict(method="fixed", **DP)),
               ("Capacity map", dict(method="capacity", **DP)), ("Noise-aware", NA)):
    R = runs({**o, "task": "multi", "labelset": "diag"}, S20)
    Rt = [r for r in R if "f1_tuned" in r["test"]]
    hl[lab] = dict(auc=ms(R), f1_05=ms(R, key=lambda r: r["test"]["f1"]),
                   f1_tuned=ms(Rt, key=lambda r: r["test"]["f1_tuned"]), n_tuned=len(Rt),
                   f1_05_same=ms(Rt, key=lambda r: r["test"]["f1"]),
                   rare=ms(R, key=lambda r: np.mean([r["test"]["label_auc"][str(c)] for c in RARE
                                                     if str(c) in r["test"]["label_auc"]])),
                   frequent=ms(R, key=lambda r: np.mean([v for c, v in r["test"]["label_auc"].items()
                                                         if int(c) not in set(RARE)])))
for lab, o in (("Fixed ratio", dict(method="fixed", **DP)), ("Capacity map", dict(method="capacity", **DP)),
               ("Noise-aware", NA), ("DP-FedAvg", dict(method="fedavg", dp="record"))):
    R = [r for r in runs(o, S20) if "hosp_auc" in r["test"]]
    worst = [np.nanmin(r["test"]["hosp_auc"]) for r in R]
    spread = [np.nanmax(r["test"]["hosp_auc"]) - np.nanmin(r["test"]["hosp_auc"]) for r in R]
    hl.setdefault("hospital", {})[lab] = dict(n=len(R), worst=float(np.mean(worst)), spread=float(np.mean(spread)),
                                              mean=float(np.mean([np.nanmean(r["test"]["hosp_auc"]) for r in R])))
out["health"] = hl

# ---------------------------------------------------------------- end-to-end time and bandwidth sweep
T_C = 0.38           # measured local computation per round, tau = 10 masked DP-SGD steps, NVIDIA T4 (s)
P_BS, W_DL = 40.0, 100e6    # downlink broadcast: 40 dBm over the full 100 MHz carrier, at the weakest scheduled rate


def regen(r):
    """Channel SNRs, schedule and per-client payload bits of a fixed / capacity / DP-FedAvg run, as fl_run draws them."""
    a = argparse.Namespace(**{**BASE, **r["cfg"]})
    s_bar = cgfl.calibrate(a, a.N, a.seed)
    ch = cgfl.Channel(a.N, a.seed, a.scenario)
    rng = np.random.default_rng(a.seed)
    m = max(1, int(round(a.part * a.N)))
    sched = [rng.choice(a.N, m, replace=False) for _ in range(a.T)]
    if a.method == "noiseaware":   # same public calibration as fl_run (sizes logged with the run)
        nk = np.asarray(r["test"]["part_sizes"], float)
        pc = np.bincount(np.concatenate(sched), minlength=a.N)
        q = np.minimum(a.batch / nk, 0.5)
        sig = np.array([cgfl.solve_sigma(a.eps, a.delta, round(q[k], 5), int(pc[k] * a.tau)) for k in range(a.N)])
        Rcal = cgfl.capacity_bits(cgfl.calib_snr(a, a.N, a.seed), a.bw_user)
        a._na_ssharp = cgfl.noise_sharp(nk / nk.sum(), sig, q, nk, a.na_sref, a.na_a)
        a._na_coef = cgfl.bisect_coef(lambda c: np.clip(np.minimum(a._na_ssharp, c * Rcal), a.smin, a.smax), s_bar)
    rounds = []
    for t in range(a.T):
        snr = ch.snr_db()
        s_all = np.ones(a.N) if a.method == "fedavg" else cgfl.retention(a.method, snr, a, rng, s_bar)
        sel = sched[t]
        rounds.append((snr[sel], s_all[sel] * r["d"] * 32.0))
    return rounds


def e2e(r, W_u=10e6, tc=T_C):
    """Uplink at sub-band W_u with the same transmit power (noise power scales with W_u), downlink broadcast of the
    dense model, and local computation; summed over rounds. At W_u = 10 MHz the uplink equals the logged value."""
    up = dl = 0.0
    for snr, bits in regen(r):
        up += np.max(bits / cgfl.capacity_bits(snr + 10 * np.log10(10e6 / W_u), W_u))
        g_dl = snr.min() + (P_BS - 23.0) - 10 * np.log10(W_DL / 10e6)
        dl += 32.0 * r["d"] / float(cgfl.capacity_bits(np.array([g_dl]), W_DL)[0])
    return dict(up=up, dl=dl, comp=tc * len(r["hist"]), logged=r["wallclock_uplink_s"])


sweep, dev = {}, 0.0
for W in (1e6, 2e6, 5e6, 10e6, 20e6):
    row = {}
    for lab, o in (("Fixed ratio", dict(method="fixed", **DP)), ("Capacity map", dict(method="capacity", **DP)),
                   ("Noise-aware", NA), ("DP-FedAvg", dict(method="fedavg", dp="record"))):
        E = [e2e(r, W) for r in runs(o, S20)]
        if W == 10e6:
            dev = max(dev, max(abs(e["up"] - e["logged"]) / e["logged"] for e in E))
        row[lab] = dict(uplink=float(np.mean([e["up"] for e in E])), downlink=float(np.mean([e["dl"] for e in E])),
                        comp=float(np.mean([e["comp"] for e in E])),
                        total=float(np.mean([e["up"] + e["dl"] + e["comp"] for e in E])))
    for lab in ("Capacity map", "Noise-aware", "DP-FedAvg"):
        row[lab]["saving_total"] = 1 - row[lab]["total"] / row["Fixed ratio"]["total"]
        row[lab]["saving_uplink"] = 1 - row[lab]["uplink"] / row["Fixed ratio"]["uplink"]
    sweep[f"{W/1e6:g} MHz"] = row
print(f"regenerated vs logged uplink at 10 MHz: max relative deviation {dev:.1e}")
out["e2e"] = sweep

# ---------------------------------------------------------------- print
def f(x): return f"{x:+.4f}"
print("== tuning picks"); [print(f"  {k:10s} {v['pick']}  val={v['val']}") for k, v in out["tune"].items()]
print("\n== 20-seed protocol")
for n, row in prot.items():
    print(f"  {n}")
    for m in ("DP-FedAvg", "Fixed ratio", "Capacity map", "Noise-aware"):
        print(f"     {m:13s} AUC {row[m]['mean']:.4f}+-{row[m]['sd']:.4f} (n={row[m]['n']})  uplink {row[m]['uplink']['mean']:9.1f}s  MB {row[m]['mb']['mean']:7.1f}")
    for k in ("cap_vs_fixed", "na_vs_fixed", "na_vs_cap", "na_vs_fedavg"):
        p = row[k]
        print(f"     {k:13s} {f(p['diff'])} CI_t[{f(p['ci_t'][0])},{f(p['ci_t'][1])}] boot[{f(p['ci_boot'][0])},{f(p['ci_boot'][1])}]"
              f" NI p_t={p['p_ni_t']:.4f} p_w={p['p_ni_w']:.4f}" + (f" holm={p['p_ni_holm']:.4f}" if "p_ni_holm" in p else "")
              + f" two-sided p={p['p_two']:.3f} uplink saved {100*p['uplink_saving']:.1f}%")
print("\n== remedies"); [print(f"  {st}: " + "; ".join(f"{k} {v['mean']:.4f}" + (f" ({v['gain']:+.4f})" if 'gain' in v else "") for k, v in r.items())) for st, r in rem.items()]
print("  selected:", out["remedy_selected"])
print("\n== baselines"); [print(f"  {st:9s} " + "; ".join(f"{k} {v['mean']:.4f}+-{v['sd']:.4f} {v['uplink']['mean']:.1f}s {v['mb']['mean']:.0f}MB n={v['n']}" for k, v in r.items())) for st, r in bl.items()]
print("\n== CSI"); [print(f"  {k:14s} AUC {v['mean']:.4f} uplink {v['uplink']['mean']:.1f}s retx {v['retx']:.0f} dropped {v['dropped']:.0f} of {v['sent']:.0f}") for k, v in csi.items()]
print("\n== mask quality"); [print(f"  {st}: " + " ".join(f"s={k}: {v['public']:.3f}/{v['random']:.3f}" for k, v in r.items())) for st, r in mq.items()]
print("\n== health", json.dumps({k: v for k, v in hl.items()}, default=float)[:3000])
print("\n== e2e"); [print(f"  {W:7s} " + "; ".join(f"{k} up {v['uplink']:.1f} dl {v['downlink']:.1f} comp {v['comp']:.1f} total {v['total']:.1f}" + (f" (save {100*v['saving_total']:.1f}% / up {100*v['saving_uplink']:.1f}%)" if 'saving_total' in v else "") for k, v in r.items())) for W, r in sweep.items()]
if OUT:
    json.dump(out, open(OUT, "w"), indent=1, default=float)
